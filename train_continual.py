"""
ZernikeNet 持续学习训练脚本
=============================
在已有的训练权重基础上，用新数据继续训练，同时防止灾难性遗忘。

使用前请确保：
  1. 已有旧训练权重文件 (如 ./weights/model_best.pth)
  2. 新数据目录结构同旧数据 (imgIF*.jpg, imgPoDF*.jpg, Zernike*.csv)
  3. continual_learning.py 在同目录下

运行方式：
    python train_continual.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

from data_utils import split_dataset, get_indices_from_dir, ZernikeDataset
from model import ZernikeUNet, ConsistentUnderCorrectLoss
from continual_learning import (
    EWC, LwF, ReplayBuffer, AdapterModule, ContinualLearningTrainer
)

torch.backends.cudnn.benchmark = True


def train_continual():
    """
    持续学习训练主函数。
    演示如何在新数据上继续训练，同时使用 LwF + EWC + Replay 防止遗忘。
    """

    # ==========================================
    # --- 1. 参数配置 ---
    # ==========================================
    # 新数据目录（你的新一批数据）
    new_data_dir = "../dataset/new_data/imgData-rr-z48"

    # 旧权重路径（在旧数据上训练得到的）
    old_weight_path = './weights/model_best.pth'

    # 旧数据目录（如果要用 EWC 计算 Fisher 或填充 Replay Buffer，需要旧数据路径）
    # 如果没有旧数据了，可以设为 None，此时只用 LwF 策略
    old_data_dir = "../dataset/def-onf-if/imgData-rr-z48"

    # 输出路径
    output_weight_path = './weights/continual_best.pth'
    output_log_path = './results/continual_log.csv'

    # 模型参数（需与旧训练一致）
    num_modes = 35
    in_channels = 2
    prefixes = ["imgIF", "imgPoDF"]

    # 持续学习超参数
    epochs = 30                  # 新任务训练轮数（建议比首次少）
    batch_size = 32
    lr = 1e-4                    # 持续学习建议用更小的学习率
    weight_decay = 1e-2

    # === 持续学习策略配置 ===
    # 可选策略: 'lwf', 'ewc', 'replay', 'adapter'
    # 推荐组合: ['lwf', 'ewc'] 或 ['lwf', 'replay'] 或全部使用
    strategies = ['lwf', 'ewc']

    # LwF 蒸馏损失权重
    lwf_lambda = 1.0             # 建议 0.5~5.0

    # EWC 正则化强度
    ewc_lambda = 1000.0          # 建议 100~10000

    # Replay Buffer 配置
    replay_buffer_size = 500     # 缓冲区大小
    replay_batch_ratio = 0.25    # 每个batch中回放样本占比

    # Adapter 配置（如果使用 adapter 策略）
    use_adapter = False
    adapter_alpha = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")
    print(f">>> Strategies: {strategies}")

    # ==========================================
    # --- 2. 数据加载 ---
    # ==========================================
    print(">>> Loading new dataset...")
    train_idx, val_idx, _ = split_dataset(new_data_dir)

    new_train_dataset = ZernikeDataset(new_data_dir, train_idx, prefixes, num_modes)
    new_val_dataset = ZernikeDataset(new_data_dir, val_idx, prefixes, num_modes)

    new_train_loader = torch.utils.data.DataLoader(
        new_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    new_val_loader = torch.utils.data.DataLoader(
        new_val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # 旧数据 DataLoader（用于 EWC Fisher 计算和 Replay Buffer 填充）
    old_data_loader = None
    if old_data_dir and os.path.exists(old_data_dir):
        if 'ewc' in strategies or 'replay' in strategies:
            print(">>> Loading old dataset for EWC/Replay preparation...")
            old_indices = get_indices_from_dir(old_data_dir)
            # 只取一部分旧数据（加速）
            sample_size = min(len(old_indices), 2000)
            old_sample_indices = old_indices[:sample_size]
            old_dataset = ZernikeDataset(old_data_dir, old_sample_indices,
                                         prefixes, num_modes)
            old_data_loader = torch.utils.data.DataLoader(
                old_dataset, batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=True
            )
            print(f"    Old data: {len(old_sample_indices)} samples loaded.")
    else:
        print("    Old data directory not found. EWC/Replay will be skipped "
              "(LwF can still work).")

    # ==========================================
    # --- 3. 模型初始化 ---
    # ==========================================
    print(">>> Initializing model and loading old weights...")
    model = ZernikeUNet(num_outputs=num_modes, in_channels=in_channels).to(device)

    # 加载旧权重
    if os.path.exists(old_weight_path):
        model.load_state_dict(
            torch.load(old_weight_path, map_location=device, weights_only=False)
        )
        print(f"    Old weights loaded from {old_weight_path}")
    else:
        print(f"    WARNING: Old weights not found at {old_weight_path}!")
        print(f"    Training from scratch (no continual learning benefit).")

    # ==========================================
    # --- 4. 损失函数 ---
    # ==========================================
    criterion = ConsistentUnderCorrectLoss(
        mse_weight=1.0, margin=0.0, sign_penalty=8.0, over_weight=3.5
    ).to(device)
    print(f"    Loss: ConsistentUnderCorrectLoss")

    # ==========================================
    # --- 5. Adapter 模式（可选）---
    # ==========================================
    if use_adapter:
        print(">>> [Adapter Mode] Freezing old model, adding adapter...")
        # 冻结旧模型所有参数
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        # 创建适配器
        adapter = AdapterModule(
            num_outputs=num_modes, in_channels=in_channels,
            alpha=adapter_alpha
        ).to(device)

        # 训练时: final_output = old_output + adapter_output
        # 需要自定义训练循环（见下方 adapter 模式）
        train_with_adapter(model, adapter, criterion, new_train_loader,
                          new_val_loader, epochs, lr, weight_decay, device,
                          output_weight_path, output_log_path)
        return

    # ==========================================
    # --- 6. 持续学习训练器 ---
    # ==========================================
    print(">>> Setting up Continual Learning Trainer...")
    trainer = ContinualLearningTrainer(
        model=model,
        criterion=criterion,
        device=device,
        old_weight_path=old_weight_path if 'lwf' in strategies else None,
        strategies=strategies,
        lwf_lambda=lwf_lambda,
        ewc_lambda=ewc_lambda,
        replay_buffer_size=replay_buffer_size,
        replay_batch_ratio=replay_batch_ratio,
    )

    # ==========================================
    # --- 7. 准备阶段 ---
    # ==========================================
    # EWC: 在旧数据上计算 Fisher 信息矩阵
    if 'ewc' in strategies and old_data_loader is not None:
        trainer.prepare_ewc(old_data_loader, max_batches=100)
        # 保存 Fisher 矩阵以备后用
        trainer.ewc.save('./weights/ewc_fisher.pth')

    # Replay: 从旧数据填充缓冲区
    if 'replay' in strategies and old_data_loader is not None:
        trainer.prepare_replay(old_data_loader, max_samples=replay_buffer_size)
        # 保存缓冲区
        trainer.replay_buffer.save('./weights/replay_buffer.pth')

    # ==========================================
    # --- 8. 训练 ---
    # ==========================================
    history = trainer.train(
        train_loader=new_train_loader,
        val_loader=new_val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        scheduler_type='cosine',  # 持续学习推荐用 cosine
        save_path=output_weight_path,
        log_path=output_log_path,
    )

    # ==========================================
    # --- 9. 可视化 ---
    # ==========================================
    plot_continual_history(history, output_log_path)

    print(f"\n>>> Done! Best model: {output_weight_path}")
    print(f">>> Training log: {output_log_path}")


def train_with_adapter(frozen_model, adapter, criterion, train_loader,
                       val_loader, epochs, lr, weight_decay, device,
                       save_path, log_path):
    """
    Adapter 模式训练：
    - 旧模型完全冻结，只训练 adapter
    - final_output = old_model(imgs) + adapter(imgs)
    """
    from collections import defaultdict
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    optimizer = optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

    best_val = float('inf')
    history = defaultdict(list)

    print(f">>> [Adapter Training] epochs={epochs}, lr={lr}")

    for epoch in range(epochs):
        t0 = time.time()
        adapter.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Adapter Epoch {epoch+1}/{epochs}")
        for imgs, targets in pbar:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 旧模型输出（无梯度）
            with torch.no_grad():
                old_output = frozen_model(imgs)

            # 适配器输出
            adapter_output = adapter(imgs)

            # 最终输出 = 旧 + 适配器
            final_output = old_output + adapter_output

            loss = criterion(final_output, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train = running_loss / len(train_loader)

        # Validation
        adapter.eval()
        val_loss = 0.0
        all_preds, all_trues = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                old_out = frozen_model(imgs)
                adapter_out = adapter(imgs)
                final = old_out + adapter_out
                loss = criterion(final, targets)
                val_loss += loss.item()
                all_preds.append(final.cpu().numpy())
                all_trues.append(targets.cpu().numpy())

        avg_val = val_loss / len(val_loader)
        v_preds = np.concatenate(all_preds, axis=0)
        v_trues = np.concatenate(all_trues, axis=0)
        sign_err = np.mean(np.sign(v_preds) * np.sign(v_trues) < 0)

        t1 = time.time()
        print(f"    Epoch {epoch+1}: Train={avg_train:.6f}, Val={avg_val:.6f}, "
              f"SignErr={sign_err:.1%}, Time={t1-t0:.1f}s")

        history['epoch'].append(epoch+1)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['sign_err'].append(sign_err)
        pd.DataFrame(history).to_csv(log_path, index=False)

        if avg_val < best_val:
            best_val = avg_val
            # 保存 adapter 权重
            torch.save({
                'adapter_state_dict': adapter.state_dict(),
                'frozen_model_path': save_path.replace('continual', 'adapter'),
            }, save_path.replace('continual_best', 'adapter_best'))
            print(f"    -> Best adapter saved.")

    print(f">>> [Adapter Training] Done. Best val: {best_val:.6f}")


def plot_continual_history(history, save_path):
    """绘制持续学习训练曲线。"""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 损失曲线
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Total', color='#1f77b4')
    axes[0].plot(df['epoch'], df['val_loss'], label='Val', color='#ff7f0e')
    if 'task_loss' in df.columns:
        axes[0].plot(df['epoch'], df['task_loss'], label='Task Loss', color='#2ca02c', ls='--')
    axes[0].set_title('Loss Curves (Continual Learning)')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()
    axes[0].grid(True, linestyle=':')

    # 各策略损失
    if 'ewc_loss' in df.columns or 'lwf_loss' in df.columns:
        if 'ewc_loss' in df.columns:
            axes[1].plot(df['epoch'], df['ewc_loss'], label='EWC Penalty', color='#d62728')
        if 'lwf_loss' in df.columns:
            axes[1].plot(df['epoch'], df['lwf_loss'], label='LwF Distill', color='#9467bd')
        axes[1].set_title('Continual Learning Losses')
        axes[1].set_xlabel('Epochs')
        axes[1].legend()
        axes[1].grid(True, linestyle=':')

    # 符号一致性
    if 'sign_err' in df.columns:
        axes[2].plot(df['epoch'], df['sign_err'], label='Sign Error Ratio',
                     color='red', lw=2.5)
        axes[2].set_title('Sign Error Ratio')
        axes[2].set_xlabel('Epochs')
        axes[2].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.1%}')
        )
        axes[2].legend()
        axes[2].grid(True, linestyle=':')

    plt.suptitle('ZernikeNet Continual Learning', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig = save_path.replace('.csv', '_curves.png')
    plt.savefig(save_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Training curves saved: {save_fig}")


if __name__ == "__main__":
    train_continual()
