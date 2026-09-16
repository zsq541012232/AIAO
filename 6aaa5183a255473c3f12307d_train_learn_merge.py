"""
ZernikeNet Learn-Merge 多轮持续更新训练脚本
============================================
场景：持续来多批新数据，每批数据量少，需要在旧模型基础上快速更新，
      同时保留绝大部分旧知识，且权重能逐轮融合（不堆积适配器）。

每轮流程：
  Phase 1 - Learn:  冻结旧模型 → 训练适配器（快，几分钟）
  Phase 2 - Merge:  旧模型+适配器作 Teacher → 蒸馏融合进权重（快，几分钟）
  → 丢弃适配器 → 模型权重已更新 → 下一轮的干净起点

使用方式：
  1. 修改下方路径配置
  2. python train_learn_merge.py
"""

import os
import torch
from data_utils import split_dataset, ZernikeDataset
from model import ZernikeUNet, ConsistentUnderCorrectLoss
from learn_merge import LearnMergePipeline

torch.backends.cudnn.benchmark = True


def train_learn_merge():
    # ==========================================
    # --- 1. 路径与参数配置 ---
    # ==========================================

    # === 旧权重（首次训练数天得到的）===
    old_weight_path = './weights/model_best.pth'

    # === 多批新数据目录（按顺序来）===
    # 每来一批新数据就加一个目录，脚本会按顺序逐轮处理
    new_data_rounds = [
        "../dataset/new_data_batch1/imgData-rr-z48",
        # "../dataset/new_data_batch2/imgData-rr-z48",   # 第二批新数据来了再取消注释
        # "../dataset/new_data_batch3/imgData-rr-z48",   # 第三批...
    ]

    # === 模型参数（必须与旧训练一致）===
    num_modes = 35
    in_channels = 2
    prefixes = ["imgIF", "imgPoDF"]
    batch_size = 32

    # === Phase 1: Learn 参数 ===
    learn_epochs = 30        # 适配器训练轮数
    learn_lr = 5e-4         # 适配器学习率
    warmup_epochs = 3        # 适配器预热

    # === Phase 2: Merge 参数 ===
    merge_epochs = 10        # 蒸馏融合轮数（少即可，因为有 Teacher 引导）
    merge_lr = 1e-4          # 融合学习率（要低，防止破坏旧权重）
    distill_lambda = 2.0    # 蒸馏损失权重（越大越贴近 Teacher = 适配器学到的知识）
    anchor_lambda = 10.0    # L2 锚定权重（越大越保守，旧知识保护越强）

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")

    # ==========================================
    # --- 2. 初始化流水线 ---
    # ==========================================
    criterion = ConsistentUnderCorrectLoss(
        mse_weight=1.0, margin=0.0, sign_penalty=8.0, over_weight=3.5
    ).to(device)

    pipeline = LearnMergePipeline(
        model_class=ZernikeUNet,
        model_kwargs={
            'num_outputs': num_modes,
            'in_channels': in_channels,
        },
        criterion=criterion,
        device=device,
        work_dir='./weights',
        log_dir='./results',
    )

    # ==========================================
    # --- 3. 逐轮执行 Learn-Merge ---
    # ==========================================
    for round_id, data_dir in enumerate(new_data_rounds, start=1):
        print(f"\n{'#'*70}")
        print(f"#  处理第 {round_id} 批新数据: {data_dir}")
        print(f"{'#'*70}")

        if not os.path.exists(data_dir):
            print(f"    目录不存在，跳过: {data_dir}")
            continue

        # 加载这批新数据
        train_idx, val_idx, _ = split_dataset(data_dir)
        train_dataset = ZernikeDataset(data_dir, train_idx, prefixes, num_modes)
        val_dataset = ZernikeDataset(data_dir, val_idx, prefixes, num_modes)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        # 执行本轮 Learn-Merge
        pipeline.run_round(
            train_loader=train_loader,
            val_loader=val_loader,
            round_id=round_id,
            old_weight_path=old_weight_path if round_id == 1 else None,
            # Phase 1
            learn_epochs=learn_epochs,
            learn_lr=learn_lr,
            warmup_epochs=warmup_epochs,
            # Phase 2
            merge_epochs=merge_epochs,
            merge_lr=merge_lr,
            distill_lambda=distill_lambda,
            anchor_lambda=anchor_lambda,
        )

        # 绘制本轮 Merge 曲线
        pipeline.plot_merge_curves(round_id)

    # ==========================================
    # --- 4. 汇总报告 ---
    # ==========================================
    pipeline.save_round_summary()

    # ==========================================
    # --- 5. 最终模型 ---
    # ==========================================
    final_model = pipeline.get_merged_model()
    final_path = os.path.join('./weights', 'model_final_merged.pth')
    torch.save(final_model.state_dict(), final_path)

    print(f"\n{'='*70}")
    print(f"全部轮次完成!")
    print(f"  最终融合模型: {final_path}")
    print(f"  总轮数: {len(new_data_rounds)}")
    print(f"\n推理时直接加载此模型即可（不需要适配器）:")
    print(f"  model = ZernikeUNet(num_outputs={num_modes}, in_channels={in_channels})")
    print(f"  model.load_state_dict(torch.load('{final_path}'))")
    print(f"  output = model(imgs)  # 与之前用法完全一样")
    print(f"{'='*70}")


if __name__ == "__main__":
    train_learn_merge()
