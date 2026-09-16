"""
ZernikeNet 旁路适配器训练脚本
===============================
场景：旧模型训练数天已完成，新数据量少只需几分钟。

原理：
  - 旧模型参数 100% 冻结 → 旧知识零遗忘
  - 旁路适配器只学增量修正 → 训练极快
  - 最终输出 = old_model(x) + alpha * adapter(x, old_output)

使用方式：
  1. 修改下方路径配置（新数据目录 + 旧权重路径）
  2. python train_adapter.py
  3. 推理时用 load_adapted_model() 加载旧模型+适配器
"""

import os
import torch
from data_utils import split_dataset, ZernikeDataset
from model import ZernikeUNet, ConsistentUnderCorrectLoss
from zernike_adapter import ZernikeSideAdapter, SideAdapterTrainer, load_adapted_model

torch.backends.cudnn.benchmark = True


def train_adapter():
    # ==========================================
    # --- 1. 路径与参数配置 ---
    # ==========================================

    # === 新数据目录 ===
    new_data_dir = "../dataset/new_data/imgData-rr-z48"

    # === 旧权重路径 ===
    old_weight_path = './weights/model_best.pth'

    # === 输出路径 ===
    adapter_save_path = './weights/adapter_best.pth'
    log_path = './results/adapter_log.csv'

    # === 模型参数（必须与旧训练一致）===
    num_modes = 35
    in_channels = 2
    prefixes = ["imgIF", "imgPoDF"]

    # === 训练参数 ===
    epochs = 30              # 新数据量少，20~50 轮足够
    batch_size = 32
    lr = 5e-4                # 适配器小网络，可用稍高学习率
    weight_decay = 1e-2
    warmup_epochs = 3        # 前几轮预热，防止适配器一开始就剧烈修改

    # === 适配器参数 ===
    feat_dim = 128           # 适配器内部特征维度
    alpha_init = 0.0         # 缩放因子初始值（0 = 从"无修正"开始）
    l2_reg = 0.01            # 修正量 L2 正则化（鼓励最小修正）

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")

    # ==========================================
    # --- 2. 加载新数据 ---
    # ==========================================
    print(">>> Loading new dataset...")
    train_idx, val_idx, _ = split_dataset(new_data_dir)

    train_dataset = ZernikeDataset(new_data_dir, train_idx, prefixes, num_modes)
    val_dataset = ZernikeDataset(new_data_dir, val_idx, prefixes, num_modes)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    print(f"    Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # ==========================================
    # --- 3. 加载旧模型并冻结 ---
    # ==========================================
    print(">>> Loading old model and freezing ALL parameters...")
    old_model = ZernikeUNet(num_outputs=num_modes, in_channels=in_channels)

    if os.path.exists(old_weight_path):
        old_model.load_state_dict(
            torch.load(old_weight_path, map_location=device, weights_only=False)
        )
        print(f"    Old weights loaded: {old_weight_path}")
    else:
        raise FileNotFoundError(f"旧权重文件不存在: {old_weight_path}")

    # 冻结所有参数
    for param in old_model.parameters():
        param.requires_grad = False
    old_model.eval()
    old_model = old_model.to(device)
    print(f"    Old model frozen (requires_grad=False for all params)")

    # ==========================================
    # --- 4. 创建适配器 ---
    # ==========================================
    print(">>> Creating side adapter...")
    adapter = ZernikeSideAdapter(
        num_outputs=num_modes,
        in_channels=in_channels,
        feat_dim=feat_dim,
        alpha_init=alpha_init,
        l2_reg=l2_reg,
    )
    adapter = adapter.to(device)

    # ==========================================
    # --- 5. 损失函数（与旧训练一致）---
    # ==========================================
    criterion = ConsistentUnderCorrectLoss(
        mse_weight=1.0, margin=0.0, sign_penalty=8.0, over_weight=3.5
    ).to(device)

    # ==========================================
    # --- 6. 训练 ---
    # ==========================================
    trainer = SideAdapterTrainer(
        frozen_model=old_model,
        adapter=adapter,
        criterion=criterion,
        device=device,
        save_path=adapter_save_path,
        log_path=log_path,
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
    )

    # ==========================================
    # --- 7. 完成 ---
    # ==========================================
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"  适配器权重: {adapter_save_path}")
    print(f"  训练日志:   {log_path}")
    print(f"  最终 alpha = {adapter.alpha.item():.4f}")
    print(f"{'='*60}")
    print(f"\n推理时使用:")
    print(f"  from zernike_adapter import load_adapted_model, adapted_inference")
    print(f"  old_model, adapter = load_adapted_model(")
    print(f"      ZernikeUNet, '{old_weight_path}', '{adapter_save_path}',")
    print(f"      num_outputs={num_modes}, in_channels={in_channels}, device=device)")
    print(f"  output = adapted_inference(old_model, adapter, imgs)")


if __name__ == "__main__":
    train_adapter()
