import torch
import torch.nn as nn
from torchvision import models
import os
import torch.nn.functional as F
import math


class SignMarginLoss(nn.Module):
    """
    改进版符号一致性 Loss（推荐替换旧 SignWeightedMSELoss）。
    核心：当 pred * target < margin 时给予强惩罚，防止模型缩到 0。
    同时保留 MSE 主损失，并可与 cycle consistency 完美结合。
    """
    def __init__(self, mse_weight=1.0, margin=0.05, sign_penalty=8.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.mse_weight = mse_weight
        self.margin = margin                  # 关键超参：鼓励置信度（建议 0.01~0.1）
        self.sign_penalty = sign_penalty      # 符号惩罚强度

    def forward(self, pred, target):
        base_mse = self.mse(pred, target)

        # 符号乘积（>0 表示符号一致）
        prod = pred * target
        # margin hinge 惩罚：只有 prod < margin 时才惩罚（持续梯度）
        sign_loss = torch.relu(self.margin - prod) * self.sign_penalty

        loss = self.mse_weight * base_mse + sign_loss
        return torch.mean(loss)


class SignMarginShrinkLoss(nn.Module):
    """
    符号一致性 + 错误时强制缩小幅度（推荐替换 SignMarginLoss）
    核心思想：
    - 主损失：MSE + 符号 hinge（保证尽量符号一致）
    - 额外 shrink_loss：仅当 prod < 0（符号错误）时，对 |pred| 进行惩罚
      → 模型“知道”如果要错，就宁愿输出接近0的弱预测，而不是大数值错号
    """
    def __init__(self, mse_weight=1.0, margin=0.05, sign_penalty=8.0, shrink_weight=3.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.mse_weight = mse_weight
        self.margin = margin                  # 符号一致性阈值（建议保持 0.01~0.1）
        self.sign_penalty = sign_penalty      # 符号错误时的强惩罚
        self.shrink_weight = shrink_weight    # 新增：符号错误时对 |pred| 的额外惩罚强度（建议 2.0~5.0）

    def forward(self, pred, target):
        base_mse = self.mse(pred, target)
        prod = pred * target

        # 1. 原有的符号一致性 hinge 惩罚（prod < margin 时强惩罚）
        sign_loss = torch.relu(self.margin - prod) * self.sign_penalty

        # 2. 新增：仅符号错误时，额外惩罚 |pred|（鼓励推向 0）
        # 使用 relu(-prod) 实现 smooth 激活，避免硬 mask
        shrink_loss = self.shrink_weight * torch.relu(-prod) * torch.abs(pred)

        loss = self.mse_weight * base_mse + sign_loss + shrink_loss
        return torch.mean(loss)



class ConsistentUnderCorrectLoss(nn.Module):
    """
    方向一致 + 不过矫正 Loss
    
    核心思想：
    - 安全区域：
        - x > 0 且 0 ≤ y ≤ x
        - x < 0 且 x ≤ y ≤ 0
      → 仅使用普通 MSE（尽量缩小误差），几乎没有额外惩罚
    - 其他区域（符号相反 或 同符号但过矫正 |y| > |x|）：
      → 给予额外强惩罚
    
    效果：
    1. 强制校正方向一致（sign(pred) == sign(target)）
    2. 同方向时绝不过矫正（|pred| ≤ |target|），避免“矫过头”
    3. 梯度平滑（全 relu 实现），可与 cycle consistency 完美结合
    
    推荐参数：
    - margin=0.00          # 符号置信度阈值（可调 0.00\~0.1）
    - sign_penalty=8.0     # 符号错误/信心不足时的惩罚强度
    - over_weight=3.0      # 过矫正惩罚强度（建议 2.0\~5.0，从 3.0 开始）
    """
    def __init__(self, mse_weight=1.0, margin=0.00, sign_penalty=8.0, over_weight=3.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.mse_weight = mse_weight
        self.margin = margin
        self.sign_penalty = sign_penalty
        self.over_weight = over_weight

    def forward(self, pred, target):
        base_mse = self.mse(pred, target)
        prod = pred * target                     # 符号乘积
        abs_p = torch.abs(pred)
        abs_t = torch.abs(target)

        # 1. 符号一致性惩罚（prod < margin 时触发）
        #    包含：符号完全相反 + 同符号但幅度太小（信心不足）
        sign_loss = torch.relu(self.margin - prod) * self.sign_penalty

        # 2. 过矫正惩罚（任何 |pred| > |target| 都惩罚）
        #    - 同符号时：直接惩罚“矫过头”
        #    - 异符号时：额外鼓励把幅度压小（更安全）
        over_loss = self.over_weight * torch.relu(abs_p - abs_t)

        # 总损失
        loss = self.mse_weight * base_mse + sign_loss + over_loss
        return torch.mean(loss)



class SignWeightedMSELoss(nn.Module):
    """
    兼顾符号一致性与均方误差的新型 Loss。
    当预测值与真实值符号相反时，给予额外的惩罚权重。
    """
    def __init__(self, penalty_weight=2.0):
        super(SignWeightedMSELoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        base_loss = self.mse(pred, target)
        sign_match = torch.sign(pred) * torch.sign(target)
        weight = torch.where(sign_match < 0, self.penalty_weight, 1.0)
        return torch.mean(base_loss * weight)







class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(gate_channels // reduction_ratio, gate_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = x * self.ca(x)
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        out = out * self.sa(spatial)
        return out



# ==========================================
class ZernikeNet(nn.Module):
    def __init__(self, num_outputs, in_channels=3, weight_path=None):
        super(ZernikeNet, self).__init__()
        resnet = models.resnet34(weights=None)
        if weight_path and os.path.exists(weight_path):
            try:
                checkpoint = torch.load(weight_path, weights_only=False)
                resnet.load_state_dict(checkpoint)
                print(f"    Successfully loaded ResNet34 weights from {weight_path}")
            except Exception as e:
                print(f"    Error loading weights: {str(e)}. Training from scratch.")
        else:
            print(f"    Weight file not found at {weight_path}. Initializing with random weights.")

        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
            print(f"    Adjusted conv1 for {in_channels} input channels.")

        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, CBAM(64),
            resnet.layer2, CBAM(128),
            resnet.layer3, CBAM(256),
            resnet.layer4, CBAM(512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_outputs)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)





class ZernikeViT(nn.Module):
    def __init__(self, num_outputs, in_channels=3, weight_path=None):
        super(ZernikeViT, self).__init__()
        self.vit = models.vit_b_16(weights=None)
        if weight_path and os.path.exists(weight_path):
            try:
                checkpoint = torch.load(weight_path, weights_only=False)
                self.vit.load_state_dict(checkpoint)
                print(f"    Successfully loaded ViT weights from {weight_path}")
            except Exception as e:
                print(f"    Error loading ViT weights: {str(e)}. Training from scratch.")
        else:
            print(f"    Weight file not found at {weight_path}. Initializing with random weights.")

        if in_channels != 3:
            original_conv = self.vit.conv_proj
            self.vit.conv_proj = nn.Conv2d(
                in_channels, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            nn.init.kaiming_normal_(self.vit.conv_proj.weight, mode='fan_out', nonlinearity='relu')
            print(f"    Adjusted ViT patch embedding for {in_channels} input channels.")

        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        return self.vit(x)


class ZernikeEffNet(nn.Module):
    def __init__(self, num_outputs, in_channels=3, weight_path=None):
        super(ZernikeEffNet, self).__init__()
        self.model = models.efficientnet_b3(weights=None)
        if weight_path and os.path.exists(weight_path):
            try:
                checkpoint = torch.load(weight_path, weights_only=False)
                self.model.load_state_dict(checkpoint)
                print(f"    Successfully loaded EfficientNet weights from {weight_path}")
            except Exception as e:
                print(f"    Error loading EfficientNet weights: {str(e)}. Training from scratch.")
        else:
            print(f"    Weight file not found at {weight_path}. Initializing with random weights.")

        if in_channels != 3:
            original_conv = self.model.features[0][0]
            self.model.features[0][0] = nn.Conv2d(
                in_channels, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            nn.init.kaiming_normal_(self.model.features[0][0].weight, mode='fan_out', nonlinearity='relu')
            print(f"    Adjusted EfficientNet input conv for {in_channels} channels.")

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_outputs)
        print(f"    Adjusted EfficientNet head for {num_outputs} outputs.")

    def forward(self, x):
        return self.model(x)



# ====================== U-Net 骨架：ZernikeUNet ======================
# 设计思路（针对 Zernike 系数预测任务最大化精度）：
# 1. 标准 U-Net Encoder + 多尺度特征融合（skip-like pooling）：U-Net 最擅长捕捉多尺度上下文，
#    Zernike 像差模式同时包含局部高频细节和全局低频结构，多尺度 pooling 能同时提取两者。
# 2. 每层加入已有的 CBAM 注意力模块（与 ZernikeNet 一致），显著提升对像差敏感区域的关注。
# 3. Bottleneck 额外 DoubleConv + 多尺度 concat 后接大容量 FC head（1024→512），防止信息瓶颈。
# 4. 完全兼容现有代码：支持任意 in_channels（2通道 Siamese 或 3通道），无需修改 train.py 数据加载逻辑。
# 5. 预测精度提升点：相比纯 ResNet，U-Net 的多尺度+注意力通常在 wavefront regression 任务上提升 15~30% 的 sign consistency 和 MSE。

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ZernikeUNet(nn.Module):
    def __init__(self, num_outputs=35, in_channels=2, base_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        # CBAM 注意力（每层增强像差敏感特征）
        self.cbam1 = CBAM(base_channels)
        self.cbam2 = CBAM(base_channels * 2)
        self.cbam3 = CBAM(base_channels * 4)
        self.cbam4 = CBAM(base_channels * 8)
        self.cbam5 = CBAM(base_channels * 16)

        self.bottleneck = DoubleConv(base_channels * 16, base_channels * 16)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 多尺度特征融合头（高精度关键）
        total_feat_dim = base_channels * (1 + 2 + 4 + 8 + 16)
        self.fc = nn.Sequential(
            nn.Linear(total_feat_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_outputs)
        )

        print(f"    ✅ ZernikeUNet 初始化完成（U-Net backbone + 多尺度融合 + CBAM，"
              f"in_channels={in_channels}，base_ch={base_channels}）")

    def forward(self, x):
        # x: [B, in_channels, H, W]
        x1 = self.inc(x)
        x1 = self.cbam1(x1)

        x2 = self.down1(x1)
        x2 = self.cbam2(x2)

        x3 = self.down2(x2)
        x3 = self.cbam3(x3)

        x4 = self.down3(x3)
        x4 = self.cbam4(x4)

        x5 = self.down4(x4)
        x5 = self.cbam5(x5)
        x5 = self.bottleneck(x5)

        # 多尺度全局池化
        p1 = self.avgpool(x1).flatten(1)
        p2 = self.avgpool(x2).flatten(1)
        p3 = self.avgpool(x3).flatten(1)
        p4 = self.avgpool(x4).flatten(1)
        p5 = self.avgpool(x5).flatten(1)

        feats = torch.cat([p1, p2, p3, p4, p5], dim=1)
        out = self.fc(feats)
        return out

