"""
ZernikeNet 旁路适配器 (Side Adapter)
=====================================
专为「旧模型训练数天、新数据量少只需几分钟」的场景设计。

核心思想：
  - 旧模型参数 100% 冻结 → 旧知识零遗忘
  - 旁路加一个小型可训练适配器 → 只学增量修正
  - 最终输出 = 旧模型输出 + α × 适配器修正

三大关键优化：
  1. 零初始化末层 + 可学习缩放因子 α（初始=0）
     → 训练开始时修正量恰好为零，模型行为与旧模型完全一致
     → 随训练逐步学习微小的增量修正，平滑过渡
  2. 条件式适配器：输入 = 原始图像 + 旧模型预测值
     → 适配器能感知「旧模型预测了什么」，做出针对性的修正
     → 比无条件适配器更精准，修正量更小
  3. 修正量 L2 正则化
     → 鼓励适配器只做最小必要修正，进一步保护旧知识
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


class ZernikeSideAdapter(nn.Module):
    """
    条件式旁路适配器。

    输入：原始图像 [B, C, H, W] + 旧模型预测 [B, num_outputs]
    输出：修正量 [B, num_outputs]

    最终预测 = old_output + alpha * adapter_correction

    参数量极小（约旧模型的 1~2%），训练只需几分钟。
    """

    def __init__(self, num_outputs=35, in_channels=2,
                 feat_dim=128, alpha_init=0.0, l2_reg=0.01):
        """
        Args:
            num_outputs: Zernike 系数数量（与旧模型一致）
            in_channels:  输入图像通道数（与旧模型一致）
            feat_dim:     适配器内部特征维度（越大容量越大，建议 64~256）
            alpha_init:   缩放因子初始值（0.0 = 从"无修正"开始，逐步学习）
            l2_reg:       修正量 L2 正则化系数（鼓励最小修正，建议 0.001~0.1）
        """
        super().__init__()

        self.num_outputs = num_outputs
        self.l2_reg = l2_reg

        # 可学习缩放因子（初始为 0 → 开始时修正量为零）
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # 轻量 CNN 编码器：从原始图像提取新数据特征
        # 比旧 U-Net 小得多（约 1/50 参数量）
        self.encoder = nn.Sequential(
            # Block 1: 64 channels
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: 64 channels (进一步降采样)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: 128 channels
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4: 全局特征
            nn.Conv2d(128, feat_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),

            # 全局平均池化 → [B, feat_dim]
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # 条件式修正头：输入 = 图像特征 + 旧模型预测
        # concat(feat_dim, num_outputs) → 修正量
        correction_head = nn.Sequential(
            nn.Linear(feat_dim + num_outputs, feat_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, num_outputs),
        )

        # 零初始化最后一层 → 初始修正量为 0
        nn.init.zeros_(correction_head[-1].weight)
        nn.init.zeros_(correction_head[-1].bias)

        self.correction_head = correction_head

        # 统计参数量
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"    [SideAdapter] 初始化完成")
        print(f"      适配器参数量: {total_params:,} (仅旧模型的 ~{total_params // 1000}K)")
        print(f"      alpha_init={alpha_init}, l2_reg={l2_reg}")
        print(f"      末层已零初始化 → 训练开始时修正量=0，行为与旧模型完全一致")

    def forward(self, img, old_output):
        """
        前向传播。

        Args:
            img:         原始输入图像 [B, C, H, W]
            old_output:  旧模型预测值  [B, num_outputs]（已 detach，无梯度）

        Returns:
            correction: 修正量 [B, num_outputs]
        """
        # 提取图像特征
        feat = self.encoder(img)  # [B, feat_dim]

        # 拼接图像特征 + 旧模型预测（条件式）
        combined = torch.cat([feat, old_output], dim=1)  # [B, feat_dim + num_outputs]

        # 生成修正量
        correction = self.correction_head(combined)  # [B, num_outputs]

        # 缩放（alpha 初始为 0，训练中逐渐增大）
        correction = self.alpha * correction

        return correction

    def get_l2_penalty(self, correction):
        """
        修正量 L2 正则化：鼓励适配器只做最小必要修正。

        loss_extra = l2_reg * mean(correction^2)
        """
        return self.l2_reg * torch.mean(correction ** 2)


class SideAdapterTrainer:
    """
    旁路适配器训练器。

    使用方式：
        # 1. 加载旧模型并冻结
        old_model = ZernikeUNet(num_outputs=35, in_channels=2)
        old_model.load_state_dict(torch.load('model_best.pth'))
        for p in old_model.parameters():
            p.requires_grad = False
        old_model.eval()

        # 2. 创建适配器
        adapter = ZernikeSideAdapter(num_outputs=35, in_channels=2)

        # 3. 训练
        trainer = SideAdapterTrainer(old_model, adapter, criterion, device)
        trainer.train(train_loader, val_loader, epochs=30, lr=5e-4)
    """

    def __init__(self, frozen_model, adapter, criterion, device,
                 save_path='./weights/adapter_best.pth',
                 log_path='./results/adapter_log.csv'):
        """
        Args:
            frozen_model: 已冻结的旧模型（requires_grad=False, eval模式）
            adapter:      ZernikeSideAdapter 实例
            criterion:    任务损失函数（如 ConsistentUnderCorrectLoss）
            device:       torch.device
            save_path:    最佳适配器权重保存路径
            log_path:     训练日志路径
        """
        self.frozen_model = frozen_model
        self.adapter = adapter
        self.criterion = criterion
        self.device = device
        self.save_path = save_path
        self.log_path = log_path

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # 确认旧模型已冻结
        frozen_count = sum(1 for p in frozen_model.parameters() if not p.requires_grad)
        total_count = sum(1 for p in frozen_model.parameters())
        print(f"    [Trainer] 旧模型已冻结: {frozen_count}/{total_count} 参数")

        # 确认适配器可训练
        adapter_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        print(f"    [Trainer] 适配器可训练参数: {adapter_params:,}")

    def _forward(self, imgs):
        """
        完整前向传播：old_output + adapter_correction

        Returns:
            final_output, old_output, correction
        """
        # 旧模型前向（无梯度，不更新参数）
        with torch.no_grad():
            old_output = self.frozen_model(imgs)

        # 适配器生成修正量
        correction = self.adapter(imgs, old_output)

        # 最终输出 = 旧模型 + 适配器修正
        final_output = old_output + correction

        return final_output, old_output, correction

    def train(self, train_loader, val_loader, epochs=30,
              lr=5e-4, weight_decay=1e-2, warmup_epochs=3):
        """
        在新数据上训练适配器。

        Args:
            epochs:         训练轮数（新数据量少，建议 20~50）
            lr:             学习率（建议 1e-4 ~ 1e-3）
            weight_decay:   权重衰减
            warmup_epochs:  预热轮数（前几轮学习率从 0 线性升到 lr，
                            防止适配器一开始就剧烈修改）
        """
        # 只优化适配器参数
        optimizer = optim.AdamW(
            self.adapter.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Cosine 退火 + Warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val_loss = float('inf')
        history = defaultdict(list)

        print(f"\n>>> [SideAdapter Training] 开始训练")
        print(f"    Epochs={epochs}, LR={lr}, Warmup={warmup_epochs}")
        print(f"    最终输出 = old_model(x) + alpha * adapter(x, old_output)")
        print(f"    alpha 初始值 = {self.adapter.alpha.item():.4f} (从0开始, 逐步学习)\n")

        for epoch in range(epochs):
            t0 = time.time()
            self.adapter.train()

            running_loss = 0.0
            running_task = 0.0
            running_l2 = 0.0
            running_alpha = 0.0
            running_corr_mag = 0.0  # 修正量平均幅度

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for imgs, targets in pbar:
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                final_output, old_output, correction = self._forward(imgs)

                # 任务损失
                task_loss = self.criterion(final_output, targets)

                # L2 正则化（鼓励最小修正）
                l2_loss = self.adapter.get_l2_penalty(correction)

                # 总损失
                total_loss = task_loss + l2_loss

                total_loss.backward()
                optimizer.step()
                scheduler.step()

                # 记录
                running_loss += total_loss.item()
                running_task += task_loss.item()
                running_l2 += l2_loss.item()
                running_alpha += self.adapter.alpha.item()
                running_corr_mag += correction.abs().mean().item()

                pbar.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    task=f"{task_loss.item():.4f}",
                    α=f"{self.adapter.alpha.item():.4f}",
                    |δ|=f"{correction.abs().mean().item():.5f}",
                )

            n = len(train_loader)
            avg_loss = running_loss / n
            avg_task = running_task / n
            avg_l2 = running_l2 / n
            avg_alpha = running_alpha / n
            avg_corr = running_corr_mag / n

            # === Validation ===
            self.adapter.eval()
            val_loss = 0.0
            val_old_loss = 0.0  # 旧模型单独的 val loss（衡量遗忘程度）
            all_preds, all_trues, all_old_preds = [], [], []

            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = imgs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    final_output, old_output, correction = self._forward(imgs)

                    val_loss += self.criterion(final_output, targets).item()
                    val_old_loss += self.criterion(old_output, targets).item()

                    all_preds.append(final_output.cpu().numpy())
                    all_old_preds.append(old_output.cpu().numpy())
                    all_trues.append(targets.cpu().numpy())

            avg_val = val_loss / len(val_loader)
            avg_val_old = val_old_loss / len(val_loader)

            # 符号一致性评估
            v_preds = np.concatenate(all_preds, axis=0)
            v_old = np.concatenate(all_old_preds, axis=0)
            v_trues = np.concatenate(all_trues, axis=0)

            sign_err_new = np.mean(np.sign(v_preds) * np.sign(v_trues) < 0)
            sign_err_old = np.mean(np.sign(v_old) * np.sign(v_trues) < 0)

            # 适配器改善了多少（相对旧模型）
            improvement = (avg_val_old - avg_val) / avg_val_old * 100 if avg_val_old > 0 else 0

            t1 = time.time()
            print(f"\n    Epoch {epoch+1}/{epochs} [{t1-t0:.0f}s]")
            print(f"      Train:  total={avg_loss:.6f} | task={avg_task:.6f} | "
                  f"l2={avg_l2:.6f}")
            print(f"      Val:    adapted={avg_val:.6f} | old={avg_val_old:.6f} | "
                  f"improvement={improvement:+.1f}%")
            print(f"      Alpha={avg_alpha:.4f} | |correction|={avg_corr:.5f} | "
                  f"SignErr: adapted={sign_err_new:.1%} vs old={sign_err_old:.1%}")

            # 记录历史
            history['epoch'].append(epoch + 1)
            history['train_total'].append(avg_loss)
            history['train_task'].append(avg_task)
            history['train_l2'].append(avg_l2)
            history['val_adapted'].append(avg_val)
            history['val_old'].append(avg_val_old)
            history['improvement_pct'].append(improvement)
            history['alpha'].append(avg_alpha)
            history['correction_mag'].append(avg_corr)
            history['sign_err_adapted'].append(sign_err_new)
            history['sign_err_old'].append(sign_err_old)

            pd.DataFrame(history).to_csv(self.log_path, index=False)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                self._save_checkpoint(epoch, avg_val, avg_val_old)
                print(f"      ★ Best model saved (val={avg_val:.6f})")

            print()

        print(f">>> [Done] Best val loss: {best_val_loss:.6f}")
        self._plot_history(history)
        return history

    def _save_checkpoint(self, epoch, val_loss, val_old_loss):
        """保存适配器权重（不保存旧模型，体积小）。"""
        torch.save({
            'epoch': epoch + 1,
            'adapter_state_dict': self.adapter.state_dict(),
            'val_loss': val_loss,
            'val_old_loss': val_old_loss,
        }, self.save_path)

    def _plot_history(self, history):
        """绘制训练曲线。"""
        import matplotlib.pyplot as plt

        df = pd.DataFrame(history)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 损失对比：adapted vs old
        axes[0, 0].plot(df['epoch'], df['val_adapted'], label='Adapted (new)', color='#1f77b4', lw=2)
        axes[0, 0].plot(df['epoch'], df['val_old'], label='Old (frozen)', color='#ff7f0e', lw=2, ls='--')
        axes[0, 0].fill_between(df['epoch'], df['val_old'], df['val_adapted'],
                                 alpha=0.15, color='#1f77b4', label='Improvement')
        axes[0, 0].set_title('Val Loss: Adapted vs Old Model')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle=':')

        # 2. 改善百分比
        axes[0, 1].plot(df['epoch'], df['improvement_pct'], color='#2ca02c', lw=2.5)
        axes[0, 1].axhline(y=0, color='gray', ls='--', alpha=0.5)
        axes[0, 1].set_title('Improvement over Old Model (%)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Improvement (%)')
        axes[0, 1].grid(True, linestyle=':')

        # 3. Alpha 和修正量幅度
        ax1 = axes[1, 0]
        ax1.plot(df['epoch'], df['alpha'], color='#d62728', lw=2.5, label='Alpha (scale)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Alpha', color='#d62728')
        ax1.tick_params(axis='y', labelcolor='#d62728')

        ax2 = ax1.twinx()
        ax2.plot(df['epoch'], df['correction_mag'], color='#9467bd', lw=2.5, ls='--', label='|correction|')
        ax2.set_ylabel('|correction|', color='#9467bd')
        ax2.tick_params(axis='y', labelcolor='#9467bd')

        axes[1, 0].set_title('Alpha & Correction Magnitude')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax1.grid(True, linestyle=':')

        # 4. 符号错误率对比
        axes[1, 1].plot(df['epoch'], df['sign_err_adapted'], label='Adapted', color='#1f77b4', lw=2.5)
        axes[1, 1].plot(df['epoch'], df['sign_err_old'], label='Old', color='#ff7f0e', lw=2.5, ls='--')
        axes[1, 1].set_title('Sign Error Ratio: Adapted vs Old')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Sign Error Ratio')
        axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        axes[1, 1].legend()
        axes[1, 1].grid(True, linestyle=':')

        plt.suptitle('ZernikeNet Side Adapter Training', fontsize=14, y=1.01)
        plt.tight_layout()
        save_fig = self.log_path.replace('.csv', '_curves.png')
        plt.savefig(save_fig, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Training curves saved: {save_fig}")


def load_adapted_model(old_model_class, old_weight_path, adapter_weight_path,
                       num_outputs=35, in_channels=2, device='cpu'):
    """
    推理时加载：旧模型 + 已训练好的适配器。

    Args:
        old_model_class:       旧模型类（如 ZernikeUNet）
        old_weight_path:       旧模型权重路径
        adapter_weight_path:   适配器权重路径
        num_outputs:           Zernike 系数数量
        in_channels:           输入通道数
        device:                推理设备

    Returns:
        frozen_model, adapter (都已加载权重并设为 eval 模式)
    """
    # 加载旧模型
    old_model = old_model_class(num_outputs=num_outputs, in_channels=in_channels)
    old_model.load_state_dict(
        torch.load(old_weight_path, map_location=device, weights_only=False)
    )
    for p in old_model.parameters():
        p.requires_grad = False
    old_model.eval()
    old_model = old_model.to(device)

    # 加载适配器
    adapter = ZernikeSideAdapter(num_outputs=num_outputs, in_channels=in_channels)
    checkpoint = torch.load(adapter_weight_path, map_location=device, weights_only=False)
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    adapter.eval()
    adapter = adapter.to(device)

    print(f">>> [Inference] Old model + Adapter loaded")
    print(f"    Old weight: {old_weight_path}")
    print(f"    Adapter weight: {adapter_weight_path}")
    print(f"    Alpha = {adapter.alpha.item():.4f}")

    return old_model, adapter


def adapted_inference(old_model, adapter, imgs):
    """
    推理函数：old_model(x) + alpha * adapter(x, old_output)

    Args:
        old_model: 冻结的旧模型
        adapter:   已训练的适配器
        imgs:      输入图像 [B, C, H, W]

    Returns:
        final_output [B, num_outputs]
    """
    with torch.no_grad():
        old_output = old_model(imgs)
        correction = adapter(imgs, old_output)
        final_output = old_output + correction
    return final_output
