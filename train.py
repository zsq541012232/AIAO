import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_utils import split_dataset, get_indices_from_dir, ZernikeDataset, ZernikeDatasetFixed3Channel
from model import (ConsistentUnderCorrectLoss, SignMarginShrinkLoss, ZernikeNet, ZernikeViT, ZernikeEffNet, SignWeightedMSELoss, SignMarginLoss,
                  ZernikeUNet)
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from generative_model import ZernikeToPSFGenerator
import model   

torch.backends.cudnn.benchmark = True


def train():
    best_val_loss = float('inf')
    save_dirs = ['weights', 'logs', 'results']
    for d in save_dirs:
        os.makedirs(d, exist_ok=True)

    # ==========================================
    # --- 1. 参数配置 ---
    # ==========================================
    use_fixed_3channel = False          # False 表示使用 Siamese 双通道模式
    
    train_dir = "../dataset/def-onf-if/imgData-rr-z48"

    weight_path = './weights/resnet34-333f7ec4.pth'   # resnet+cbam（仅供其他模型使用）
    num_modes = 35
    epochs = 50
    batch_size = 32

    prefixes = ["imgIF", "imgPoDF"]   # 当前使用的两个通道

    # ==================== Cycle Consistency ====================
    use_cycle_loss = False
    cycle_lambda = 0.05          # 可调，建议 0.01~0.1
    generative_weight_path = './weights/generative_best.pth'

    margin = 0.00
    sign_penalty = 8.0
    shrink_weight = 3.0    # ← 推荐从 2.0 开始尝试，最高可到 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using Device: {device} | Modes: {num_modes}")

    # ==========================================
    # --- 2. 数据处理与 DataLoader ---
    # ==========================================
    print(">>> Splitting dataset from single directory...")
    train_idx, val_idx, _ = split_dataset(train_dir)
    data_dir_train, data_dir_val = train_dir, train_dir

    print(">>> Loading datasets...")
    DatasetClass = ZernikeDatasetFixed3Channel if use_fixed_3channel else ZernikeDataset
    model_in_channels = 3 if use_fixed_3channel else len(prefixes)

    train_dataset = DatasetClass(data_dir_train, train_idx, prefixes, num_modes)
    val_dataset = DatasetClass(data_dir_val, val_idx, prefixes, num_modes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ==========================================
    # --- 3. 模型与损失函数初始化 ---
    # ==========================================
    print(">>> Initializing Model...")

    if use_fixed_3channel:
        raise NotImplementedError("3通道模式暂未适配 Siamese 模型")
    else:
        # model = ZernikeSiameseViTAttnResRoPE(num_outputs=num_modes).to(device)
        # model = ZernikeEffNet(num_outputs=num_modes, in_channels=model_in_channels, weight_path=None).to(device)
        # model = ZernikeNet(num_outputs=num_modes, in_channels=model_in_channels, weight_path=weight_path).to(device)
        # model = ZernikeSiameseResNetCBAM(num_outputs=num_modes, weight_path=weight_path).to(device)
        # model = ZernikeDualCrossNet(num_outputs=num_modes, weight_path=weight_path).to(device)
        model = ZernikeUNet(num_outputs=num_modes, in_channels=model_in_channels).to(device)
        # model = ZernikeMambaPure(num_outputs=num_modes, in_channels=model_in_channels, img_size=224).to(device)
        # model = ZernikeFusionMambaPure(num_outputs=num_modes, img_size=224).to(device)
        # model = ZernikeUNetMambaDeepFusion(num_outputs=num_modes, in_channels=model_in_channels, img_size=224).to(device)

    # 加载预训练生成式模型（冻结）
    gen_model = None
    if use_cycle_loss:
        gen_model = ZernikeToPSFGenerator(num_modes=num_modes).to(device)
        if os.path.exists(generative_weight_path):
            gen_model.load_state_dict(torch.load(generative_weight_path, map_location=device))
            print(f">>> 成功加载生成式模型: {generative_weight_path}")
        else:
            print(">>> Warning: 未找到生成式模型权重，cycle loss 将被禁用")
            use_cycle_loss = False
        # 冻结生成式模型
        for param in gen_model.parameters():
            param.requires_grad = False
        gen_model.eval()
        print("    ✅ 生成式模型已冻结，用于 cycle consistency")

    # criterion = nn.MSELoss().to(device)
    # print("    ✅ 已使用 nn.MSELoss")
    # criterion = SignWeightedMSELoss(penalty_weight=sign_penalty).to(device)
    # print("    ✅ 已使用 SignWeightedMSELoss（含符号惩罚）")
    # criterion = SignMarginLoss(mse_weight=1.0, margin=margin, sign_penalty=sign_penalty).to(device)
    # print(f"    ✅ 已使用 SignMarginLoss（margin={margin}, penalty={sign_penalty}）")
    # criterion = SignMarginShrinkLoss(mse_weight=1.0,margin=margin, sign_penalty=sign_penalty, shrink_weight=shrink_weight).to(device)
    # print(f"    ✅ 已使用 SignMarginShrinkLoss（margin={margin}, sign_penalty={sign_penalty}, shrink_weight={shrink_weight}）")
    criterion = ConsistentUnderCorrectLoss(mse_weight=1.0,margin=0.0,sign_penalty=8.0,over_weight=3.5).to(device)
    print(f"    ✅ 已使用 ConsistentUnderCorrectLoss（margin=0.0, sign_penalty=8.0, over_weight=3.5）")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                              steps_per_epoch=len(train_loader),
                                              epochs=epochs, pct_start=0.1)

    # 训练记录
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': [],
        'val_sign_err_item': [], 'val_avg_wrong_mag': [], 'val_severe_sign_err': [],
        'val_mean_sign_prod': [], 'val_norm_sign_err': [],
        'train_recon_loss': []   
    }

    print(">>> Starting GPU training loop...")
    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for imgs, targets in pbar:
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(imgs)                    # pred_zernike

            loss_main = criterion(outputs, targets)

            if use_cycle_loss:
                real_if = imgs[:, 0:1, :, :]         # 第0通道就是 imgIF（已log1p）
                gen_if = gen_model(outputs)          # 用预测的z生成图像
                recon_loss = nn.MSELoss()(gen_if, real_if)
                loss = loss_main + cycle_lambda * recon_loss
                current_recon = recon_loss.item()
            else:
                loss = loss_main
                current_recon = 0.0

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_running_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), recon=current_recon if use_cycle_loss else 0)

        avg_train_loss = train_running_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_running_loss = 0.0
        val_all_preds, val_all_trues = [], []

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
            for imgs, targets in vbar:
                imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item()

                val_all_preds.append(outputs.cpu().numpy())
                val_all_trues.append(targets.cpu().numpy())

        avg_val_loss = val_running_loss / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]

        # 符号一致性评估
        # ==================== 新指标计算 ====================
        v_preds = np.concatenate(val_all_preds, axis=0)
        v_trues = np.concatenate(val_all_trues, axis=0)

        sign_match = np.sign(v_preds) * np.sign(v_trues)
        mismatch = sign_match < 0
        mismatch_ratio = np.mean(mismatch)

        wrong_mag = np.abs(v_preds[mismatch])
        avg_wrong_mag = np.mean(wrong_mag) if len(wrong_mag) > 0 else 0.0

        threshold = 0.01
        severe_mask = mismatch & (np.abs(v_preds) > threshold)
        severe_ratio = np.mean(severe_mask)

        mean_sign_prod = np.mean(v_preds * v_trues)

        norm_sign_err = np.sum(np.abs(v_preds - v_trues)[mismatch]) / (np.sum(np.abs(v_trues)) + 1e-8)

        # 记录到 history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)
        history['val_sign_err_item'].append(mismatch_ratio)
        history['val_avg_wrong_mag'].append(avg_wrong_mag)
        history['val_severe_sign_err'].append(severe_ratio)
        history['val_mean_sign_prod'].append(mean_sign_prod)
        history['val_norm_sign_err'].append(norm_sign_err)
        history['train_recon_loss'].append(current_recon)

        epoch_end = time.time()
        print(f"    Epoch {epoch + 1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}, "
              f"SignErr={mismatch_ratio:.1%} | AvgWrongMag={avg_wrong_mag:.4f} | "
              f"Severe={severe_ratio:.1%} | MeanProd={mean_sign_prod:.4f} | "
              f"NormErr={norm_sign_err:.1%}, Time={epoch_end - epoch_start:.1f}s")


        pd.DataFrame(history).to_csv("./results/training_log.csv", index=False)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "./weights/model_best.pth")


    model_name = model.__class__.__name__
    loss_name = criterion.__class__.__name__
    plot_history(history, model_name, loss_name)




def plot_history(history, model_name, loss_name):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # 损失曲线
    axes[0].plot(history['epoch'], history['train_loss'], label='Train Loss', color='#1f77b4')
    axes[0].plot(history['epoch'], history['val_loss'], label='Val Loss', color='#ff7f0e')
    axes[0].set_title('Loss Curve ({loss_name})')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss Value')
    axes[0].legend()
    axes[0].grid(True, linestyle=':')

    # LR
    axes[1].plot(history['epoch'], history['lr'], label='Learning Rate', color='orange')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('LR')
    axes[1].legend()
    axes[1].grid(True, linestyle=':')

    # Sign Error Ratio 
    axes[2].plot(history['epoch'], history['val_sign_err_item'], label='Sign Error Ratio', color='red', lw=2.5)
    axes[2].plot(history['epoch'], history['val_severe_sign_err'], label='Severe Sign Error (>0.01)', color='#d62728', lw=2.5, linestyle='--')
    axes[2].set_title('Sign Error Ratio')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Ratio')
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    axes[2].legend()
    axes[2].grid(True, linestyle=':')

    # Avg Wrong Magnitude（核心新指标）
    axes[3].plot(history['epoch'], history['val_avg_wrong_mag'], label='Avg Wrong Magnitude', color='#9467bd', lw=2.5)
    axes[3].set_title('Average Magnitude of Sign-Wrong Predictions')
    axes[3].set_xlabel('Epochs')
    axes[3].set_ylabel('Magnitude (lower is not always better)')
    axes[3].legend()
    axes[3].grid(True, linestyle=':')

    # Mean Sign Product（全局符号一致性）
    axes[4].plot(history['epoch'], history['val_mean_sign_prod'], label='Mean Sign Product', color='#2ca02c', lw=2.5)
    axes[4].set_title('Mean Sign Product (higher = better)')
    axes[4].set_xlabel('Epochs')
    axes[4].set_ylabel('Product')
    axes[4].legend()
    axes[4].grid(True, linestyle=':')

    # Normalized Sign Error Contribution
    axes[5].plot(history['epoch'], history['val_norm_sign_err'], label='Norm Sign Error Contribution', color='#ff7f0e', lw=2.5)
    axes[5].set_title('Normalized Sign Error Contribution to Total Error')
    axes[5].set_xlabel('Epochs')
    axes[5].set_ylabel('Ratio')
    axes[5].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    axes[5].legend()
    axes[5].grid(True, linestyle=':')

    plt.suptitle(f'Zernike Training: {model_name} | Optimizer: AdamW | Loss: {loss_name}', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    save_path = f'./results/training_curves_{model_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"    ✅ 已生成可视化图表: {save_path}")




if __name__ == "__main__":
    train()
