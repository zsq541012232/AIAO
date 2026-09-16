"""
ZernikeNet Learn-Merge 持续更新流水线
========================================
解决「适配器无法直接融合进旧模型权重」的问题，支持多轮持续更新。

核心思想：每轮分两个阶段
  Phase 1 - Learn（学习）:
    冻结旧模型 → 训练旁路适配器（快，几分钟）
    Teacher = 旧模型 + 适配器（在新数据上的最优解）

  Phase 2 - Merge（融合）:
    将 (旧模型 + 适配器) 作为 Teacher
    Student = 旧模型权重解冻，用三重损失训练：
      ① task_loss:     新数据真实标签
      ② distill_loss: 蒸馏 Teacher 的输出（吸收适配器知识）
      ③ anchor_loss:  L2 锚定旧权重（保护旧知识）
    融合完毕 → 丢弃适配器 → 模型权重已更新 → 干净的起点

多轮流程:
  Round 1: old_weight → learn adapter_1 → merge → model_r1.pth
  Round 2: model_r1  → learn adapter_2 → merge → model_r2.pth
  Round 3: model_r2  → learn adapter_3 → merge → model_r3.pth
  ...（无限轮次，每轮只需几分钟，无适配器堆积）

优点:
  - 每轮只需几分钟（适配器训练 + 少量蒸馏轮次）
  - 旧知识通过蒸馏 + L2锚定双重保护
  - 适配器用完即弃，不堆积
  - 模型权重持续更新，知识逐轮积累
"""

import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict

from zernike_adapter import ZernikeSideAdapter, SideAdapterTrainer


class LearnMergePipeline:
    """
    Learn-Merge 两阶段持续更新流水线。

    使用方式：
        pipeline = LearnMergePipeline(
            model_class=ZernikeUNet,
            model_kwargs={'num_outputs': 35, 'in_channels': 2},
            criterion=criterion,
            device=device,
        )

        # 第 1 轮
        pipeline.run_round(
            train_loader=new_loader_1,
            val_loader=val_loader_1,
            old_weight_path='./weights/model_best.pth',
            round_id=1,
        )

        # 第 2 轮（自动使用上一轮融合后的模型）
        pipeline.run_round(
            train_loader=new_loader_2,
            val_loader=val_loader_2,
            round_id=2,
        )

        # 推理时直接用融合后的模型（不需要适配器）
        model = pipeline.get_merged_model()
    """

    def __init__(self, model_class, model_kwargs, criterion, device,
                 work_dir='./weights', log_dir='./results'):
        """
        Args:
            model_class:   模型类（如 ZernikeUNet）
            model_kwargs:  模型初始化参数 dict
            criterion:     任务损失函数
            device:        torch.device
            work_dir:      权重保存目录
            log_dir:       日志保存目录
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.criterion = criterion
        self.device = device
        self.work_dir = work_dir
        self.log_dir = log_dir

        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # 当前模型权重（每轮 Merge 后更新）
        self.current_model = None
        self.current_weight_path = None
        self.round_history = []

        print(f">>> [LearnMergePipeline] 初始化完成")
        print(f"    模型: {model_class.__name__}")
        print(f"    参数: {model_kwargs}")

    def _create_model(self, load_path=None):
        """创建模型并加载权重。"""
        model = self.model_class(**self.model_kwargs)
        if load_path and os.path.exists(load_path):
            model.load_state_dict(
                torch.load(load_path, map_location='cpu', weights_only=False)
            )
        model = model.to(self.device)
        return model

    def _freeze_model(self, model):
        """冻结模型所有参数。"""
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    def _unfreeze_model(self, model):
        """解冻模型所有参数。"""
        for param in model.parameters():
            param.requires_grad = True
        model.train()

    def run_round(self, train_loader, val_loader, round_id,
                  old_weight_path=None,
                  learn_epochs=30, learn_lr=5e-4,
                  merge_epochs=10, merge_lr=1e-4,
                  distill_lambda=2.0, anchor_lambda=10.0,
                  warmup_epochs=3, l2_reg=0.01,
                  verbose=True):
        """
        执行一轮 Learn-Merge。

        Args:
            train_loader:     新数据训练集
            val_loader:       新数据验证集
            round_id:         轮次编号
            old_weight_path:  旧权重路径（仅第 1 轮需要指定，
                              后续轮自动使用上一轮的融合模型）
            learn_epochs:     Phase 1 适配器训练轮数
            learn_lr:          Phase 1 学习率
            merge_epochs:     Phase 2 蒸馏融合轮数
            merge_lr:         Phase 2 学习率（要比首次训练低很多）
            distill_lambda:   蒸馏损失权重（建议 1.0~5.0）
            anchor_lambda:    L2 锚定权重（建议 1.0~50.0，越大越保守）
            warmup_epochs:    适配器预热轮数
            l2_reg:           适配器修正量 L2 正则化

        Returns:
            round_result dict
        """
        print(f"\n{'='*70}")
        print(f"  Learn-Merge Round {round_id}")
        print(f"{'='*70}")

        # 确定本轮起始权重
        if round_id == 1 and old_weight_path:
            start_weight = old_weight_path
        elif self.current_weight_path:
            start_weight = self.current_weight_path
        else:
            raise ValueError("第 1 轮必须指定 old_weight_path")

        # ==================== Phase 1: Learn ====================
        print(f"\n--- Phase 1: Learn (训练适配器) ---")

        # 加载旧模型并冻结
        old_model = self._create_model(start_weight)
        self._freeze_model(old_model)
        old_model = old_model.to(self.device)

        # 创建适配器
        adapter = ZernikeSideAdapter(
            num_outputs=self.model_kwargs.get('num_outputs', 35),
            in_channels=self.model_kwargs.get('in_channels', 2),
            feat_dim=128,
            alpha_init=0.0,
            l2_reg=l2_reg,
        )
        adapter = adapter.to(self.device)

        # 训练适配器
        adapter_path = os.path.join(self.work_dir, f'adapter_round{round_id}.pth')
        adapter_log = os.path.join(self.log_dir, f'adapter_round{round_id}.csv')

        learn_trainer = SideAdapterTrainer(
            frozen_model=old_model,
            adapter=adapter,
            criterion=self.criterion,
            device=self.device,
            save_path=adapter_path,
            log_path=adapter_log,
        )

        learn_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=learn_epochs,
            lr=learn_lr,
            weight_decay=1e-2,
            warmup_epochs=warmup_epochs,
        )

        # 记录 Phase 1 结果
        learn_alpha = adapter.alpha.item()
        print(f"\n    Phase 1 完成: alpha={learn_alpha:.4f}, "
              f"adapter saved: {adapter_path}")

        # ==================== Phase 2: Merge ====================
        print(f"\n--- Phase 2: Merge (蒸馏融合进权重) ---")

        # Teacher = 旧模型 + 适配器（全部冻结）
        # 用 Teacher 的输出作为蒸馏目标
        old_model.eval()
        adapter.eval()
        for p in old_model.parameters():
            p.requires_grad = False
        for p in adapter.parameters():
            p.requires_grad = False

        # Student = 旧模型权重的副本（解冻，可训练）
        # 从旧权重重新加载（保证起点一致）
        student = self._create_model(start_weight)
        self._unfreeze(student)

        # 保存旧权重快照用于 L2 锚定
        old_params_snapshot = {
            name: param.data.clone().detach()
            for name, param in student.named_parameters()
        }

        # 优化器
        merge_optimizer = optim.AdamW(
            student.parameters(), lr=merge_lr, weight_decay=1e-2
        )
        merge_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            merge_optimizer, T_max=merge_epochs, eta_min=merge_lr * 0.1
        )

        merge_log = os.path.join(self.log_dir, f'merge_round{round_id}.csv')
        best_merge_val = float('inf')
        merge_history = defaultdict(list)

        for epoch in range(merge_epochs):
            t0 = time.time()
            student.train()
            run_task, run_distill, run_anchor, run_total = 0.0, 0.0, 0.0, 0.0

            pbar = tqdm(train_loader, desc=f"Merge {epoch+1}/{merge_epochs}")
            for imgs, targets in pbar:
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                merge_optimizer.zero_grad()

                # Teacher 输出（无梯度）
                with torch.no_grad():
                    old_out = old_model(imgs)
                    correction = adapter(imgs, old_out)
                    teacher_output = old_out + correction

                # Student 输出
                student_output = student(imgs)

                # ① 任务损失
                task_loss = self.criterion(student_output, targets)

                # ② 蒸馏损失：Student 拟合 Teacher 输出
                distill_loss = F.mse_loss(student_output, teacher_output)

                # ③ L2 锚定损失：参数不偏离旧值太远
                anchor_loss = 0.0
                for name, param in student.named_parameters():
                    if name in old_params_snapshot:
                        anchor_loss = anchor_loss + (
                            (param - old_params_snapshot[name].to(param.device)) ** 2
                        ).sum()

                # 总损失
                total_loss = (
                    task_loss
                    + distill_lambda * distill_loss
                    + anchor_lambda * anchor_loss * 1e-4  # 缩放因子防数值过大
                )

                total_loss.backward()
                merge_optimizer.step()

                run_task += task_loss.item()
                run_distill += distill_loss.item()
                run_anchor += anchor_loss.item()
                run_total += total_loss.item()

                pbar.set_postfix(
                    task=f"{task_loss.item():.4f}",
                    dist=f"{distill_loss.item():.4f}",
                    anch=f"{anchor_loss.item():.2f}",
                )

            merge_scheduler.step()

            n = len(train_loader)
            avg_task = run_task / n
            avg_distill = run_distill / n
            avg_anchor = run_anchor / n

            # 验证
            student.eval()
            val_loss = 0.0
            val_teacher_loss = 0.0  # Teacher 的 val loss
            all_preds, all_trues, all_teacher = [], [], []

            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = imgs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    s_out = student(imgs)
                    old_out = old_model(imgs)
                    corr = adapter(imgs, old_out)
                    t_out = old_out + corr

                    val_loss += self.criterion(s_out, targets).item()
                    val_teacher_loss += self.criterion(t_out, targets).item()

                    all_preds.append(s_out.cpu().numpy())
                    all_teacher.append(t_out.cpu().numpy())
                    all_trues.append(targets.cpu().numpy())

            avg_val = val_loss / len(val_loader)
            avg_val_teacher = val_teacher_loss / len(val_loader)

            # 符号错误率
            v_preds = np.concatenate(all_preds, axis=0)
            v_trues = np.concatenate(all_trues, axis=0)
            sign_err = np.mean(np.sign(v_preds) * np.sign(v_trues) < 0)

            # Student 与 Teacher 输出差异（衡量融合程度）
            v_teacher = np.concatenate(all_teacher, axis=0)
            merge_gap = np.mean((v_preds - v_teacher) ** 2)

            t1 = time.time()
            print(f"    Merge Epoch {epoch+1}: "
                  f"task={avg_task:.4f} | distill={avg_distill:.4f} | "
                  f"anchor={avg_anchor:.2f} | "
                  f"val={avg_val:.6f} | teacher_val={avg_val_teacher:.6f} | "
                  f"merge_gap={merge_gap:.6f} | "
                  f"sign_err={sign_err:.1%} | {t1-t0:.0f}s")

            merge_history['epoch'].append(epoch + 1)
            merge_history['task_loss'].append(avg_task)
            merge_history['distill_loss'].append(avg_distill)
            merge_history['anchor_loss'].append(avg_anchor)
            merge_history['val_loss'].append(avg_val)
            merge_history['val_teacher_loss'].append(avg_val_teacher)
            merge_history['merge_gap'].append(merge_gap)
            merge_history['sign_err'].append(sign_err)

            pd.DataFrame(merge_history).to_csv(merge_log, index=False)

            if avg_val < best_merge_val:
                best_merge_val = avg_val
                merged_path = os.path.join(self.work_dir, f'model_merged_r{round_id}.pth')
                torch.save(student.state_dict(), merged_path)
                print(f"    ★ Best merged model: {merged_path}")

        # ==================== 更新当前模型 ====================
        self.current_weight_path = merged_path
        self.current_model = student
        self.current_model.eval()

        # 清理 GPU 显存：释放旧模型和适配器
        del old_model, adapter
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        round_result = {
            'round': round_id,
            'start_weight': start_weight,
            'merged_weight': merged_path,
            'learn_alpha': learn_alpha,
            'merge_best_val': best_merge_val,
            'learn_epochs': learn_epochs,
            'merge_epochs': merge_epochs,
        }
        self.round_history.append(round_result)

        print(f"\n>>> Round {round_id} 完成!")
        print(f"    融合后模型: {merged_path}")
        print(f"    最佳 Val Loss: {best_merge_val:.6f}")
        print(f"    适配器已丢弃，模型权重已更新")
        print(f"    下一轮可直接使用此模型作为起点\n")

        return round_result

    def get_merged_model(self):
        """获取当前融合后的模型（用于推理，不需要适配器）。"""
        if self.current_model is None:
            raise RuntimeError("尚未执行任何轮次，请先调用 run_round()")
        self.current_model.eval()
        return self.current_model

    def save_round_summary(self):
        """保存多轮汇总报告。"""
        if not self.round_history:
            print("无轮次记录")
            return

        summary_path = os.path.join(self.log_dir, 'learn_merge_summary.csv')
        df = pd.DataFrame(self.round_history)
        df.to_csv(summary_path, index=False)
        print(f">>> 多轮汇总报告: {summary_path}")
        print(df.to_string(index=False))

    def plot_merge_curves(self, round_id):
        """绘制某一轮的 Merge 训练曲线。"""
        import matplotlib.pyplot as plt

        merge_log = os.path.join(self.log_dir, f'merge_round{round_id}.csv')
        if not os.path.exists(merge_log):
            print(f"日志文件不存在: {merge_log}")
            return

        df = pd.read_csv(merge_log)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. 损失曲线
        axes[0].plot(df['epoch'], df['task_loss'], label='Task', color='#1f77b4')
        axes[0].plot(df['epoch'], df['distill_loss'], label='Distill', color='#ff7f0e')
        axes[0].plot(df['epoch'], df['anchor_loss'], label='Anchor (×100)', color='#2ca02c')
        axes[0].set_title(f'Round {round_id} Merge Losses')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True, linestyle=':')

        # 2. Val Loss: Student vs Teacher
        axes[1].plot(df['epoch'], df['val_loss'], label='Student (merged)', color='#1f77b4', lw=2)
        axes[1].plot(df['epoch'], df['val_teacher_loss'], label='Teacher (old+adapter)',
                     color='#ff7f0e', lw=2, ls='--')
        axes[1].set_title(f'Round {round_id} Val: Student vs Teacher')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, linestyle=':')

        # 3. Merge Gap（Student 趋近 Teacher 的程度）
        axes[2].plot(df['epoch'], df['merge_gap'], label='Merge Gap', color='#9467bd', lw=2.5)
        axes[2].set_title(f'Round {round_id} Merge Gap (Student→Teacher)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('MSE(student, teacher)')
        axes[2].legend()
        axes[2].grid(True, linestyle=':')

        plt.suptitle(f'Learn-Merge Round {round_id}', fontsize=14, y=1.02)
        plt.tight_layout()
        save_fig = merge_log.replace('.csv', '_curves.png')
        plt.savefig(save_fig, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Merge curves saved: {save_fig}")
