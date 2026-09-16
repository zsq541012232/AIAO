"""
ZernikeNet 持续学习模块 (Continual Learning Module)
=====================================================
解决「在新数据上继续训练但不忘旧知识」的问题（灾难性遗忘）。

提供四种主流策略，可单独使用或组合使用：
  1. EWC  (Elastic Weight Consolidation)   —— 基于 Fisher 信息矩阵的正则化
  2. LwF  (Learning without Forgetting)    —— 旧模型知识蒸馏
  3. ReplayBuffer (Experience Replay)      —— 回放旧数据样本
  4. Adapter (Side / Adapter Network)      —— 冻结主干 + 旁路适配器

所有策略均兼容现有 model.py / data_utils.py / train.py 架构。
"""

import os
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


# ============================================================
#  1. EWC: Elastic Weight Consolidation
# ============================================================
#  原理：在旧任务上训练完毕后，计算每个参数的 Fisher 信息（对损失的敏感度）。
#  新任务训练时，对重要参数施加二次惩罚，阻止它们偏离旧最优值太远。
#  优点：不需要保存旧数据，只需保存旧权重 + Fisher 对角矩阵。

class EWC:
    """
    Elastic Weight Consolidation 正则化器。

    使用方式：
        # 1. 在旧数据上训练完毕后，计算 Fisher 信息
        ewc = EWC(model, lambda_reg=1000)
        ewc.compute_fisher(model, old_data_loader, criterion, device)

        # 2. 在新数据训练时，将 EWC 惩罚项加入总损失
        loss = task_loss + ewc.penalty(model)
    """

    def __init__(self, lambda_reg: float = 1000.0):
        """
        Args:
            lambda_reg: EWC 正则化强度。越大越保守（更不忘旧），太小则效果弱。
                        建议范围: 100 ~ 10000，从 1000 开始调。
        """
        self.lambda_reg = lambda_reg
        self.fisher_info = None       # dict: param_name -> Fisher 对角矩阵
        self.old_params = None         # dict: param_name -> 旧最优参数值（detached）

    def compute_fisher(self, model: nn.Module, data_loader, criterion, device,
                       max_batches: int = 100):
        """
        在旧数据上计算 Fisher 信息矩阵的对角近似。

        Args:
            model:           已在旧数据上训练好的模型
            data_loader:     旧数据的 DataLoader（只需一小部分即可）
            criterion:       训练时使用的损失函数
            device:          cuda / cpu
            max_batches:     最多采样多少个 batch 计算 Fisher（节省时间）
        """
        print(">>> [EWC] Computing Fisher Information Matrix...")
        model.eval()

        # 初始化 Fisher 矩阵为零
        fisher = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        count = 0
        for imgs, targets in data_loader:
            if count >= max_batches:
                break
            imgs = imgs.to(device)
            targets = targets.to(device)

            model.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()

            # 累积梯度的平方作为 Fisher 对角近似
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            count += 1

        # 取平均
        for name in fisher:
            fisher[name] /= count

        # 归一化（防止数值过大）
        total_fisher = sum(f.mean().item() for f in fisher.values()) / len(fisher)
        if total_fisher > 0:
            scale = 1.0 / total_fisher
            for name in fisher:
                fisher[name] *= scale

        self.fisher_info = fisher
        self.old_params = {name: param.data.clone().detach()
                           for name, param in model.named_parameters()
                           if param.requires_grad}

        print(f"    [EWC] Fisher computed on {count} batches, "
              f"lambda_reg={self.lambda_reg}")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        计算 EWC 正则化损失：lambda * Σ F_i * (θ_i - θ*_i)^2

        在新任务训练时调用此函数并加入总损失。
        """
        if self.fisher_info is None or self.old_params is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher_info:
                # (θ - θ*)^2 * F
                penalty = self.fisher_info[name].to(param.device) * \
                          (param.data - self.old_params[name].to(param.device)) ** 2
                loss = loss + penalty.sum()

        return self.lambda_reg * loss

    def save(self, path: str):
        """保存 Fisher 矩阵和旧参数到磁盘。"""
        if self.fisher_info is None:
            print("    [EWC] Warning: Fisher not computed, nothing to save.")
            return
        save_dict = {
            'fisher_info': self.fisher_info,
            'old_params': self.old_params,
            'lambda_reg': self.lambda_reg,
        }
        torch.save(save_dict, path)
        print(f"    [EWC] Saved to {path}")

    def load(self, path: str, device='cpu'):
        """从磁盘加载 Fisher 矩阵和旧参数。"""
        save_dict = torch.load(path, map_location=device, weights_only=False)
        self.fisher_info = save_dict['fisher_info']
        self.old_params = save_dict['old_params']
        self.lambda_reg = save_dict['lambda_reg']
        print(f"    [EWC] Loaded from {path}, lambda_reg={self.lambda_reg}")


# ============================================================
#  2. LwF: Learning without Forgetting
# ============================================================
#  原理：冻结旧模型作为 Teacher，在新数据上前向传播得到旧模型的"软标签"。
#  新模型（Student）在新数据上学习时，同时拟合：
#    - 新数据的真实标签（task loss）
#    - 旧模型在新数据上的输出（distillation loss）
#  优点：不需要旧数据！只需旧权重文件。
#  注意：对回归任务，distillation loss 用 MSE 即可。

class LwF:
    """
    Learning without Forgetting 知识蒸馏器。

    使用方式：
        # 1. 加载旧模型作为 Teacher（自动冻结）
        lwf = LwF(teacher_model, old_weight_path, lambda_distill=1.0, device=device)

        # 2. 在新数据训练时，计算蒸馏损失
        distill_loss = lwf.distillation_loss(student_output, imgs)
        loss = task_loss + lambda * distill_loss
    """

    def __init__(self, teacher_model: nn.Module, old_weight_path: str,
                 lambda_distill: float = 1.0, temperature: float = 1.0):
        """
        Args:
            teacher_model:     旧模型结构（与 Student 相同的类）
            old_weight_path:   旧权重文件路径 (.pth)
            lambda_distill:    蒸馏损失权重。建议 0.5~5.0，从 1.0 开始调。
            temperature:       温度参数（回归任务一般保持 1.0）
        """
        self.lambda_distill = lambda_distill
        self.temperature = temperature
        self.teacher = None

        if os.path.exists(old_weight_path):
            teacher_model.load_state_dict(
                torch.load(old_weight_path, map_location='cpu', weights_only=False)
            )
            self.teacher = teacher_model
            # 冻结 Teacher 所有参数
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()
            print(f">>> [LwF] Teacher loaded from {old_weight_path}, "
                  f"frozen, lambda_distill={lambda_distill}")
        else:
            print(f"    [LwF] Warning: weight file not found {old_weight_path}, "
                  f"LwF disabled.")

    def distillation_loss(self, student_output: torch.Tensor,
                         imgs: torch.Tensor) -> torch.Tensor:
        """
        计算知识蒸馏损失。

        Args:
            student_output: Student 模型在新数据上的输出 [B, num_modes]
            imgs:            新数据的输入图像 [B, C, H, W]

        Returns:
            distillation loss（标量）
        """
        if self.teacher is None:
            return torch.tensor(0.0, device=student_output.device)

        with torch.no_grad():
            teacher_output = self.teacher(imgs)

        # 回归任务蒸馏：直接用 MSE
        # 也可加温度缩放：MSE(s*T, t*T) / T^2
        T = self.temperature
        loss = F.mse_loss(student_output * T, teacher_output * T) / (T * T)

        return self.lambda_distill * loss

    def to(self, device):
        if self.teacher is not None:
            self.teacher = self.teacher.to(device)
        return self


# ============================================================
#  3. ReplayBuffer: Experience Replay
# ============================================================
#  原理：保留一小部分旧数据样本，在新任务训练时混入回放。
#  优点：效果最好（最直接地防止遗忘），实现最简单。
#  缺点：需要存储旧数据（但只需少量，如 5%~10%）。

class ReplayBuffer:
    """
    经验回放缓冲区。

    使用方式：
        # 1. 从旧数据中采样一部分存入 Buffer
        buffer = ReplayBuffer(max_size=500)
        buffer.populate_from_loader(old_data_loader, max_samples=500)

        # 2. 新任务训练时，每个 batch 混入回放样本
        for imgs, targets in new_loader:
            replay_imgs, replay_targets = buffer.sample(batch_size=16)
            mixed_imgs = torch.cat([imgs, replay_imgs])
            mixed_targets = torch.cat([targets, replay_targets])
            ...  # 正常训练
    """

    def __init__(self, max_size: int = 500):
        """
        Args:
            max_size: 缓冲区最大容量（样本数）。建议 200~2000。
                      越大效果越好但占用内存/显存越多。
        """
        self.max_size = max_size
        self.imgs = []       # list of tensors
        self.targets = []    # list of tensors

    def populate_from_loader(self, data_loader, max_samples: int = 500):
        """
        从 DataLoader 中随机采样存入缓冲区。

        Args:
            data_loader:  旧数据的 DataLoader
            max_samples:   最多采样多少个样本
        """
        print(f">>> [ReplayBuffer] Populating from data loader...")
        all_imgs = []
        all_targets = []

        for imgs, targets in data_loader:
            for i in range(imgs.shape[0]):
                all_imgs.append(imgs[i])
                all_targets.append(targets[i])
                if len(all_imgs) >= max_samples:
                    break
            if len(all_imgs) >= max_samples:
                break

        # 随机采样 max_size 个
        if len(all_imgs) > self.max_size:
            indices = random.sample(range(len(all_imgs)), self.max_size)
        else:
            indices = list(range(len(all_imgs)))

        self.imgs = [all_imgs[i] for i in indices]
        self.targets = [all_targets[i] for i in indices]
        print(f"    [ReplayBuffer] Stored {len(self.imgs)} samples.")

    def sample(self, batch_size: int, device='cpu'):
        """
        从缓冲区随机采样一个 mini-batch。

        Returns:
            (replay_imgs, replay_targets) tensors
        """
        if len(self.imgs) == 0:
            raise RuntimeError("ReplayBuffer is empty!")

        batch_size = min(batch_size, len(self.imgs))
        indices = random.sample(range(len(self.imgs)), batch_size)

        replay_imgs = torch.stack([self.imgs[i] for i in indices]).to(device)
        replay_targets = torch.stack([self.targets[i] for i in indices]).to(device)

        return replay_imgs, replay_targets

    def save(self, path: str):
        """保存缓冲区到磁盘。"""
        save_dict = {
            'imgs': torch.stack(self.imgs),
            'targets': torch.stack(self.targets),
            'max_size': self.max_size,
        }
        torch.save(save_dict, path)
        print(f"    [ReplayBuffer] Saved {len(self.imgs)} samples to {path}")

    def load(self, path: str):
        """从磁盘加载缓冲区。"""
        save_dict = torch.load(path, map_location='cpu', weights_only=False)
        self.imgs = [save_dict['imgs'][i] for i in range(save_dict['imgs'].shape[0])]
        self.targets = [save_dict['targets'][i] for i in range(save_dict['targets'].shape[0])]
        self.max_size = save_dict['max_size']
        print(f"    [ReplayBuffer] Loaded {len(self.imgs)} samples from {path}")


# ============================================================
#  4. Adapter: 旁路适配器模块
# ============================================================
#  原理：完全冻结旧模型的所有参数，在旁路添加一个小型可训练适配器网络。
#  适配器学习新数据特有的修正量，最终输出 = 旧模型输出 + 适配器修正。
#  优点：旧知识 100% 保留（参数完全冻结），且适配器小、训练快。
#  缺点：表达能力受限（只能做增量修正）。

class AdapterModule(nn.Module):
    """
    旁路适配器：在冻结的旧模型旁边添加轻量可训练网络。

    结构：
        input -> [Frozen Old Model] -> old_output
              -> [Adapter Network]  -> adapter_output
        final_output = old_output + alpha * adapter_output

    使用方式：
        # 1. 加载旧模型并冻结
        old_model = ZernikeUNet(num_outputs=35, in_channels=2)
        old_model.load_state_dict(torch.load('model_best.pth'))
        for p in old_model.parameters():
            p.requires_grad = False
        old_model.eval()

        # 2. 创建适配器
        adapter = AdapterModule(num_outputs=35, in_channels=2, alpha=1.0)
        adapter.to(device)

        # 3. 训练时只训练 adapter 参数
        optimizer = optim.AdamW(adapter.parameters(), lr=1e-3)

        # 4. 前向传播
        old_out = old_model(imgs)        # 不计算梯度
        adapter_out = adapter(imgs)      # 计算梯度
        final_out = old_out + adapter_out
        loss = criterion(final_out, targets)
    """

    def __init__(self, num_outputs: int = 35, in_channels: int = 2,
                 alpha: float = 1.0):
        """
        Args:
            num_outputs: Zernike 系数数量
            in_channels: 输入图像通道数
            alpha:       适配器输出的缩放因子。初始化为 0 则从"无修正"开始，
                         随训练逐渐增大。建议 1.0 或 0.1。
        """
        super().__init__()
        self.alpha = alpha

        # 轻量适配器网络（小型 UNet 风格）
        # 比旧模型小得多，只学增量修正
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_outputs),
        )

        # 初始化最后一层为接近零（从"无修正"开始，逐步学习增量）
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        print(f"    [Adapter] Initialized (alpha={alpha}), "
              f"last layer zero-initialized for smooth start.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回适配器修正量（需与旧模型输出相加）。

        Args:
            x: 输入图像 [B, C, H, W]

        Returns:
            adapter_output [B, num_outputs]
        """
        return self.alpha * self.adapter(x)


# ============================================================
#  5. ContinualLearningTrainer: 统一持续学习训练器
# ============================================================
#  整合上述四种策略，提供一键式持续学习训练接口。

class ContinualLearningTrainer:
    """
    统一持续学习训练器，支持组合使用 EWC + LwF + Replay + Adapter。

    使用方式（完整示例见 train_continual.py）：

        trainer = ContinualLearningTrainer(
            model=model,
            criterion=criterion,
            device=device,
            old_weight_path='./weights/model_best.pth',
            strategies=['lwf', 'ewc'],   # 选择策略
            lwf_lambda=1.0,
            ewc_lambda=1000.0,
        )

        # 如果用 EWC，需先在旧数据上计算 Fisher
        trainer.prepare_ewc(old_data_loader)

        # 如果用 Replay，需先填充缓冲区
        trainer.prepare_replay(old_data_loader, max_samples=500)

        # 在新数据上训练
        trainer.train(new_train_loader, new_val_loader, epochs=30, lr=1e-4)
    """

    def __init__(self, model, criterion, device,
                 old_weight_path=None,
                 strategies=None,
                 lwf_lambda=1.0,
                 ewc_lambda=1000.0,
                 replay_buffer_size=500,
                 replay_batch_ratio=0.25):
        """
        Args:
            model:              Student 模型（新任务上要训练的）
            criterion:          任务损失函数（如 ConsistentUnderCorrectLoss）
            device:             torch.device
            old_weight_path:    旧权重文件路径
            strategies:         使用的策略列表，如 ['ewc', 'lwf', 'replay']
            lwf_lambda:          LwF 蒸馏损失权重
            ewc_lambda:          EWC 正则化强度
            replay_buffer_size:  Replay 缓冲区大小
            replay_batch_ratio:  每个 batch 中回放样本占比（0~1）
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.strategies = strategies or []
        self.replay_batch_ratio = replay_batch_ratio

        # 初始化各策略
        self.ewc = None
        self.lwf = None
        self.replay_buffer = None
        self.adapter = None

        # EWC
        if 'ewc' in self.strategies:
            self.ewc = EWC(lambda_reg=ewc_lambda)

        # LwF（需要旧权重路径）
        if 'lwf' in self.strategies and old_weight_path:
            # 创建 Teacher（深拷贝当前模型结构）
            teacher = copy.deepcopy(model)
            self.lwf = LwF(teacher, old_weight_path, lambda_distill=lwf_lambda)
            self.lwf.to(device)

        # Replay
        if 'replay' in self.strategies:
            self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

        print(f">>> [ContinualLearningTrainer] Strategies: {self.strategies}")

    def prepare_ewc(self, old_data_loader, max_batches=100):
        """在旧数据上计算 Fisher 信息矩阵（EWC 前置步骤）。"""
        if self.ewc is None:
            print("    [CL Trainer] EWC not enabled, skipping.")
            return
        self.ewc.compute_fisher(self.model, old_data_loader,
                                self.criterion, self.device, max_batches)

    def prepare_replay(self, old_data_loader, max_samples=500):
        """从旧数据填充回放缓冲区。"""
        if self.replay_buffer is None:
            print("    [CL Trainer] Replay not enabled, skipping.")
            return
        self.replay_buffer.populate_from_loader(old_data_loader, max_samples)

    def compute_total_loss(self, imgs, targets, outputs):
        """
        计算总损失 = task_loss + ewc_penalty + lwf_distill_loss

        Args:
            imgs:     输入图像 [B, C, H, W]
            targets:  真实标签 [B, num_modes]
            outputs:  模型输出 [B, num_modes]

        Returns:
            total_loss, loss_dict (各部分损失值)
        """
        loss_dict = {}

        # 1. 任务损失
        task_loss = self.criterion(outputs, targets)
        loss_dict['task'] = task_loss.item()
        total_loss = task_loss

        # 2. EWC 正则化
        if self.ewc is not None and self.ewc.fisher_info is not None:
            ewc_loss = self.ewc.penalty(self.model)
            loss_dict['ewc'] = ewc_loss.item()
            total_loss = total_loss + ewc_loss

        # 3. LwF 蒸馏
        if self.lwf is not None and self.lwf.teacher is not None:
            lwf_loss = self.lwf.distillation_loss(outputs, imgs)
            loss_dict['lwf'] = lwf_loss.item()
            total_loss = total_loss + lwf_loss

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict

    def train(self, train_loader, val_loader, epochs=30,
              lr=1e-4, weight_decay=1e-2,
              scheduler_type='cosine',
              save_path='./weights/continual_best.pth',
              log_path='./results/continual_log.csv'):
        """
        在新数据上执行持续学习训练。

        Args:
            train_loader:   新数据训练集 DataLoader
            val_loader:      新数据验证集 DataLoader
            epochs:          训练轮数（建议比首次训练少，如 20~30）
            lr:              学习率（建议比首次训练低，如 1e-4）
            weight_decay:    权重衰减
            scheduler_type:  'cosine' 或 'onecycle'
            save_path:       最佳模型保存路径
            log_path:        训练日志 CSV 保存路径
        """
        import torch.optim as optim
        from tqdm import tqdm
        import pandas as pd
        import time

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # 优化器（只优化 requires_grad=True 的参数）
        optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay
        )

        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr * 0.01
            )
        else:
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr,
                steps_per_epoch=len(train_loader),
                epochs=epochs, pct_start=0.1
            )

        best_val_loss = float('inf')
        history = defaultdict(list)

        print(f">>> [CL Trainer] Starting continual learning training...")
        print(f"    Epochs={epochs}, LR={lr}, Strategies={self.strategies}")

        for epoch in range(epochs):
            t0 = time.time()
            self.model.train()
            running_loss = 0.0
            running_task = 0.0
            running_ewc = 0.0
            running_lwf = 0.0

            pbar = tqdm(train_loader, desc=f"CL Epoch {epoch+1}/{epochs}")
            for imgs, targets in pbar:
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # === Replay: 混入旧数据 ===
                if self.replay_buffer is not None and len(self.replay_buffer.imgs) > 0:
                    replay_bs = max(1, int(imgs.shape[0] * self.replay_batch_ratio))
                    r_imgs, r_targets = self.replay_buffer.sample(
                        replay_bs, device=self.device
                    )
                    imgs = torch.cat([imgs, r_imgs], dim=0)
                    targets = torch.cat([targets, r_targets], dim=0)

                optimizer.zero_grad()
                outputs = self.model(imgs)

                total_loss, loss_dict = self.compute_total_loss(
                    imgs, targets, outputs
                )

                total_loss.backward()
                optimizer.step()

                if scheduler_type != 'cosine':
                    scheduler.step()

                running_loss += loss_dict['total']
                running_task += loss_dict.get('task', 0)
                running_ewc += loss_dict.get('ewc', 0)
                running_lwf += loss_dict.get('lwf', 0)

                pbar.set_postfix(
                    loss=f"{loss_dict['total']:.4f}",
                    task=f"{loss_dict.get('task', 0):.4f}",
                    ewc=f"{loss_dict.get('ewc', 0):.4f}",
                    lwf=f"{loss_dict.get('lwf', 0):.4f}"
                )

            if scheduler_type == 'cosine':
                scheduler.step()

            avg_train_loss = running_loss / len(train_loader)
            avg_task = running_task / len(train_loader)

            # === Validation ===
            self.model.eval()
            val_loss = 0.0
            all_preds, all_trues = [], []

            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = imgs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
                    all_preds.append(outputs.cpu().numpy())
                    all_trues.append(targets.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)

            # 符号一致性评估
            v_preds = np.concatenate(all_preds, axis=0)
            v_trues = np.concatenate(all_trues, axis=0)
            sign_match = np.sign(v_preds) * np.sign(v_trues)
            mismatch = sign_match < 0
            mismatch_ratio = np.mean(mismatch)
            mean_sign_prod = np.mean(v_preds * v_trues)

            t1 = time.time()
            print(f"    Epoch {epoch+1}: Train={avg_train_loss:.6f}, "
                  f"Val={avg_val_loss:.6f}, Task={avg_task:.4f}, "
                  f"EWC={running_ewc/len(train_loader):.4f}, "
                  f"LwF={running_lwf/len(train_loader):.4f}, "
                  f"SignErr={mismatch_ratio:.1%}, "
                  f"Time={t1-t0:.1f}s")

            # 记录
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['task_loss'].append(avg_task)
            history['ewc_loss'].append(running_ewc / len(train_loader))
            history['lwf_loss'].append(running_lwf / len(train_loader))
            history['sign_err'].append(mismatch_ratio)
            history['mean_sign_prod'].append(mean_sign_prod)

            pd.DataFrame(history).to_csv(log_path, index=False)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"    -> Best model saved: {save_path}")

        print(f">>> [CL Trainer] Done. Best val loss: {best_val_loss:.6f}")
        return history
