# -*- coding: utf-8 -*-
"""
远场光斑数据随机增强脚本

对指定目录下的在焦/离焦光斑图像做随机的【平移、旋转、伸缩】增强，
泽尼克系数文件保持不变，仅复制。

目录中每组数据的文件格式：
    IFimg{编号}.bmp      在焦远场光斑图
    DFimg{编号}.bmp      离焦远场光斑图
    ZerCof{编号}.txt     泽尼克系数（15 阶）

编号规则：
    按编号从小到大依次增强，新数据紧接当前最大编号往后排。
    例如原有 1~100 组，则 1 号的增强结果编号为 101，100 号的为 200。
    （若编号不连续，则按“已存在数据组的排序位次”顺延。）

增强规则：
    - 默认同一编号的 IFimg 与 DFimg 各自独立采样随机参数，增强方式不同；
      设置 sync_augment=True（命令行 --sync）后，两张图使用完全相同的
      增强参数（同步平移/旋转/伸缩）。
    - 平移量按图像尺寸的【百分比】采样，与分辨率无关（256x256 时 5%≈12.8 像素）。
    - ZerCof 内容原样复制，不做任何修改。
    - 输出图像尺寸与原图一致，空出区域以黑色（0）填充。
    - 默认开启光斑保留校验：增强后光斑不得消失、不得整体越界，
      不合格的随机参数会被自动丢弃并重新采样。
"""

import os
import re
import shutil
import argparse

import cv2
import numpy as np


# ----------------------------------------------------------------------
# 增强参数默认范围（可按实际物理意义调整）
# ----------------------------------------------------------------------
ROTATION_RANGE = 10.0      # 旋转角度范围（度），在 [-10, 10] 内均匀采样
TRANSLATION_RANGE = 0.05   # 平移范围（图像宽/高的比例），5% 即 256 图上约 ±12.8 像素
SCALE_RANGE = (0.9, 1.1)   # 伸缩比例范围，在 [0.9, 1.1] 内均匀采样

# 光斑保留校验默认阈值
MIN_ENERGY_RETENTION = 0.85  # 增强后光斑能量 / 原光斑能量 的下限
CENTROID_MARGIN_RATIO = 0.08  # 光斑质心距画面边界的最小距离（占边长比例）
MAX_BORDER_RATIO = 0.05       # 光斑主体贴在图像最外圈的像素占比上限
MAX_RETRIES = 50              # 单张图随机参数不合格时的最大重采样次数
BORDER_WIDTH = 3              # “贴边”检查的边缘环宽（像素）

IF_PREFIX = "IFimg"
DF_PREFIX = "DFimg"
ZER_PREFIX = "ZerCof"
IMG_EXT = ".bmp"
TXT_EXT = ".txt"


def _scan_indices(folder):
    """扫描目录，返回三类文件各自包含的编号集合"""
    def _collect(prefix, ext):
        pattern = re.compile(r"^" + re.escape(prefix) + r"(\d+)" + re.escape(ext) + r"$")
        indices = set()
        for name in os.listdir(folder):
            m = pattern.match(name)
            if m:
                indices.add(int(m.group(1)))
        return indices

    if_idx = _collect(IF_PREFIX, IMG_EXT)
    df_idx = _collect(DF_PREFIX, IMG_EXT)
    zer_idx = _collect(ZER_PREFIX, TXT_EXT)
    return if_idx, df_idx, zer_idx


# ----------------------------------------------------------------------
# 光斑检测与保留校验
# ----------------------------------------------------------------------
def analyze_spot(img):
    """
    检测黑底亮斑图中的光斑主体（阈值分割后面积最大的亮连通域）。

    返回 dict:
        mask      光斑主体布尔掩码
        area      光斑像素数
        energy    光斑区域灰度能量（灰度值之和，float64）
        cx, cy    光斑能量加权质心（像素坐标）
        border_ratio  光斑落在图像最外圈环内的像素占比
    未检测到亮目标时返回 None。
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray_f = gray.astype(np.float64)
    h, w = gray.shape

    # Otsu 自动阈值分割亮目标（先轻微模糊抑制椒盐噪声）
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None

    # 忽略背景(label 0)，取面积最大的连通域作为光斑主体
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    if stats[k, cv2.CC_STAT_AREA] <= 1:
        return None
    mask = (labels == k)

    energy = float(gray_f[mask].sum())
    area = int(mask.sum())
    if energy <= 0:
        return None

    # 能量加权质心（比连通域几何质心更贴近亮斑中心）
    ys, xs = np.nonzero(mask)
    weights = gray_f[mask]
    cx = float((xs * weights).sum() / weights.sum())
    cy = float((ys * weights).sum() / weights.sum())

    b = BORDER_WIDTH
    border_band = np.zeros_like(mask)
    border_band[:b, :] = True
    border_band[-b:, :] = True
    border_band[:, :b] = True
    border_band[:, -b:] = True
    border_ratio = float(np.count_nonzero(mask & border_band) / area)

    return {"mask": mask, "area": area, "energy": energy,
            "cx": cx, "cy": cy, "border_ratio": border_ratio}


def check_spot_preserved(orig_img, aug_img,
                         min_energy_retention=MIN_ENERGY_RETENTION,
                         centroid_margin_ratio=CENTROID_MARGIN_RATIO,
                         max_border_ratio=MAX_BORDER_RATIO):
    """
    校验增强后光斑是否仍然保留在画面内。

    三项判据（全部满足才算合格）：
      1. 能量保留率 >= min_energy_retention（防止光斑整体移出/消失）；
      2. 光斑质心距画面四边 >= centroid_margin_ratio * 边长；
      3. 光斑主体贴边像素占比 <= max_border_ratio。

    返回 (ok: bool, info: dict)。
    """
    o = analyze_spot(orig_img)
    a = analyze_spot(aug_img)
    if o is None:
        return False, {"reason": "原图未检测到光斑"}
    if a is None:
        return False, {"reason": "增强后光斑消失",
                       "energy_retention": 0.0, "border_ratio": 1.0,
                       "cx": -1.0, "cy": -1.0}

    h, w = orig_img.shape[:2]
    energy_retention = a["energy"] / max(o["energy"], 1e-9)
    margin = centroid_margin_ratio * min(w, h)
    centroid_ok = (margin <= a["cx"] <= w - margin) and \
                  (margin <= a["cy"] <= h - margin)
    border_ok = a["border_ratio"] <= max_border_ratio
    energy_ok = energy_retention >= min_energy_retention

    info = {
        "energy_retention": energy_retention,
        "cx": a["cx"], "cy": a["cy"],
        "ncx": a["cx"] / w, "ncy": a["cy"] / h,  # 归一化质心 0~1
        "area_ratio": a["area"] / max(o["area"], 1),
        "border_ratio": a["border_ratio"],
        "energy_ok": energy_ok,
        "centroid_ok": centroid_ok,
        "border_ok": border_ok,
    }
    return energy_ok and centroid_ok and border_ok, info


def _violation_penalty(info):
    """将校验结果折算为连续的违规分数（0 为完全合格），用于重试耗尽时择优"""
    if "reason" in info:
        return 1e6
    p = 0.0
    if not info["energy_ok"]:
        p += MIN_ENERGY_RETENTION - info["energy_retention"]
    if not info["border_ok"]:
        p += info["border_ratio"] - MAX_BORDER_RATIO
    if not info["centroid_ok"]:
        # 质心越界距离（归一化坐标 0~1）
        ncx, ncy = info["ncx"], info["ncy"]
        margin = CENTROID_MARGIN_RATIO
        over = max(margin - ncx, ncx - (1 - margin),
                   margin - ncy, ncy - (1 - margin), 0.0)
        p += over
    return p


# ----------------------------------------------------------------------
# 随机增强
# ----------------------------------------------------------------------
def random_transform_params(rng,
                            rotation_range=ROTATION_RANGE,
                            translation_range=TRANSLATION_RANGE,
                            scale_range=SCALE_RANGE):
    """独立采样一组随机增强参数。tx/ty 为相对图像宽/高的比例（非像素）"""
    angle = rng.uniform(-rotation_range, rotation_range)
    tx_ratio = rng.uniform(-translation_range, translation_range)
    ty_ratio = rng.uniform(-translation_range, translation_range)
    scale = rng.uniform(scale_range[0], scale_range[1])
    return {"angle": angle, "tx_ratio": tx_ratio,
            "ty_ratio": ty_ratio, "scale": scale}


def augment_image(img, params):
    """
    对单张光斑图施加 绕中心旋转 + 伸缩 + 平移 的组合仿射变换。
    平移按图像尺寸百分比换算为像素；输出尺寸与输入一致，画布外补零。
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    # 绕图像中心旋转并伸缩
    mat = cv2.getRotationMatrix2D(center, params["angle"], params["scale"])
    # 叠加百分比平移
    mat[0, 2] += params["tx_ratio"] * w
    mat[1, 2] += params["ty_ratio"] * h

    augmented = cv2.warpAffine(
        img,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return augmented


def _safe_augment(rng, img, rotation_range, translation_range, scale_range,
                  ensure_spot, check_kwargs):
    """
    对单张图采样参数并增强；开启校验时不合格则重采样。
    返回 (augmented, params, checks: list[(ok, info)], retries)。
    """
    best = None  # (penalty, augmented, params, info)
    retries = 0
    attempts = (MAX_RETRIES + 1) if ensure_spot else 1

    for attempt in range(attempts):
        params = random_transform_params(rng, rotation_range,
                                         translation_range, scale_range)
        aug = augment_image(img, params)
        if not ensure_spot:
            return aug, params, [(True, {})], 0

        ok, info = check_spot_preserved(img, aug, **check_kwargs)
        if ok:
            return aug, params, [(ok, info)], attempt
        retries = attempt + 1
        penalty = _violation_penalty(info)
        if best is None or penalty < best[0]:
            best = (penalty, aug, params, info)

    # 重试耗尽：返回违规程度最小的一组（调用方负责告警）
    _, aug, params, info = best
    return aug, params, [(False, info)], retries


def augment_dataset(data_dir,
                    output_dir=None,
                    seed=None,
                    rotation_range=ROTATION_RANGE,
                    translation_range=TRANSLATION_RANGE,
                    scale_range=SCALE_RANGE,
                    sync_augment=False,
                    ensure_spot=True,
                    min_energy_retention=MIN_ENERGY_RETENTION,
                    centroid_margin_ratio=CENTROID_MARGIN_RATIO,
                    max_border_ratio=MAX_BORDER_RATIO,
                    overwrite=False):
    """
    对目录下的远场光斑数据执行随机增强。

    参数
    ----
    data_dir : str
        原始数据所在目录。
    output_dir : str | None
        增强结果输出目录；None 表示写回原目录（编号顺延，不覆盖原文件）。
    seed : int | None
        随机种子，传入后增强结果可复现。
    rotation_range : float
        随机旋转角度的绝对值上限（度）。
    translation_range : float
        随机平移比例上限（相对图像宽/高，0.05 表示 ±5%）。
    scale_range : (float, float)
        随机伸缩比例区间。
    sync_augment : bool
        True 时同组 IFimg/DFimg 使用完全相同的增强参数；
        False（默认）时两张图各自独立采样，增强方式不同。
    ensure_spot : bool
        是否开启光斑保留校验并重采样不合格参数，默认 True。
    min_energy_retention : float
        增强后/原光斑能量比下限，默认 0.85。
    centroid_margin_ratio : float
        光斑质心距边界的最小距离比例，默认 0.08。
    max_border_ratio : float
        光斑贴边像素占比上限，默认 0.05。
    overwrite : bool
        目标文件已存在时是否覆盖，默认 False（跳过并告警）。

    返回
    ----
    generated : list[tuple]
        实际生成的 (原编号, 新编号) 列表。
    """
    data_dir = os.path.abspath(data_dir)
    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    check_kwargs = {
        "min_energy_retention": min_energy_retention,
        "centroid_margin_ratio": centroid_margin_ratio,
        "max_border_ratio": max_border_ratio,
    }

    # 1. 扫描现有编号（先快照，避免把本次新生成的文件再次纳入）
    if_idx, df_idx, zer_idx = _scan_indices(data_dir)
    valid_idx = sorted(if_idx & df_idx & zer_idx)

    if not valid_idx:
        print(f"[警告] 在 {data_dir} 下未找到任何完整的数据组"
              f"（需同时包含 {IF_PREFIX}N{IMG_EXT}、"
              f"{DF_PREFIX}N{IMG_EXT}、{ZER_PREFIX}N{TXT_EXT}）。")
        return []

    # 仅有部分文件缺失的编号，给出提示
    missing = sorted((if_idx | df_idx | zer_idx) - set(valid_idx))
    if missing:
        print(f"[提示] 以下编号文件不完整，已跳过: {missing}")

    max_idx = max(valid_idx)
    rng = np.random.default_rng(seed)
    generated = []

    print(f"[信息] 原始完整数据组: {len(valid_idx)} 组，"
          f"编号范围 {valid_idx[0]}~{max_idx}")
    print(f"[信息] 增强结果编号范围: "
          f"{max_idx + 1}~{max_idx + len(valid_idx)}，输出目录: {output_dir}")
    print(f"[信息] 增强幅度: 旋转±{rotation_range}°, "
          f"平移±{translation_range * 100:.1f}%, 伸缩{scale_range[0]}~{scale_range[1]}; "
          f"IF/DF{'同步' if sync_augment else '独立'}增强; "
          f"光斑校验{'开启' if ensure_spot else '关闭'}")

    # 2. 按编号从小到大依次增强
    for pos, src_idx in enumerate(valid_idx, start=1):
        new_idx = max_idx + pos

        if_path = os.path.join(data_dir, f"{IF_PREFIX}{src_idx}{IMG_EXT}")
        df_path = os.path.join(data_dir, f"{DF_PREFIX}{src_idx}{IMG_EXT}")
        zer_path = os.path.join(data_dir, f"{ZER_PREFIX}{src_idx}{TXT_EXT}")

        out_if_path = os.path.join(output_dir, f"{IF_PREFIX}{new_idx}{IMG_EXT}")
        out_df_path = os.path.join(output_dir, f"{DF_PREFIX}{new_idx}{IMG_EXT}")
        out_zer_path = os.path.join(output_dir, f"{ZER_PREFIX}{new_idx}{TXT_EXT}")

        # 防覆盖保护
        targets = [out_if_path, out_df_path, out_zer_path]
        if not overwrite and any(os.path.exists(p) for p in targets):
            print(f"[跳过] 编号 {new_idx} 的目标文件已存在（源 {src_idx}），"
                  f"如需覆盖请加 --overwrite")
            continue

        # 以原始位深读取（灰度光斑图通常为 uint8 单通道）
        if_img = cv2.imread(if_path, cv2.IMREAD_UNCHANGED)
        df_img = cv2.imread(df_path, cv2.IMREAD_UNCHANGED)
        if if_img is None or df_img is None:
            print(f"[错误] 编号 {src_idx} 图像读取失败，跳过。")
            continue

        warnings = []
        if sync_augment:
            # 同步模式：两图共用同一组参数，必须同时通过两张图的光斑校验
            best = None
            retries = 0
            attempts = (MAX_RETRIES + 1) if ensure_spot else 1
            for attempt in range(attempts):
                params = random_transform_params(rng, rotation_range,
                                                 translation_range, scale_range)
                if_aug = augment_image(if_img, params)
                df_aug = augment_image(df_img, params)
                if not ensure_spot:
                    if_info, df_info = {}, {}
                    break
                ok_if, if_info = check_spot_preserved(if_img, if_aug, **check_kwargs)
                ok_df, df_info = check_spot_preserved(df_img, df_aug, **check_kwargs)
                if ok_if and ok_df:
                    retries = attempt
                    break
                retries = attempt + 1
                penalty = _violation_penalty(if_info) + _violation_penalty(df_info)
                if best is None or penalty < best[0]:
                    best = (penalty, if_aug, df_aug, params, if_info, df_info)
            else:
                _, if_aug, df_aug, params, if_info, df_info = best
                warnings.append(
                    f"同步参数重试 {MAX_RETRIES} 次仍未完全合格，"
                    f"已取最优结果（IF:{if_info} DF:{df_info}），"
                    f"请检查原图光斑是否本就贴近边界")
            if_params = df_params = params
        else:
            # 独立模式：两张图分别采样、分别校验
            if_aug, if_params, if_checks, if_retries = _safe_augment(
                rng, if_img, rotation_range, translation_range, scale_range,
                ensure_spot, check_kwargs)
            df_aug, df_params, df_checks, df_retries = _safe_augment(
                rng, df_img, rotation_range, translation_range, scale_range,
                ensure_spot, check_kwargs)
            if_info = if_checks[0][1]
            df_info = df_checks[0][1]
            retries = max(if_retries, df_retries)
            if if_retries > MAX_RETRIES - 1 and not if_checks[0][0]:
                warnings.append(f"IFimg 重试 {MAX_RETRIES} 次未完全合格: {if_info}")
            if df_retries > MAX_RETRIES - 1 and not df_checks[0][0]:
                warnings.append(f"DFimg 重试 {MAX_RETRIES} 次未完全合格: {df_info}")

        # 保持与原图一致的数据类型
        if if_aug.dtype != if_img.dtype:
            if_aug = if_aug.astype(if_img.dtype)
        if df_aug.dtype != df_img.dtype:
            df_aug = df_aug.astype(df_img.dtype)

        cv2.imwrite(out_if_path, if_aug)
        cv2.imwrite(out_df_path, df_aug)
        # 泽尼克系数不随图像几何变换改变，原样复制
        shutil.copyfile(zer_path, out_zer_path)

        generated.append((src_idx, new_idx))

        def _fmt(p, info, tag):
            s = (f"{tag}(angle={p['angle']:+.2f}°, "
                 f"tx={p['tx_ratio'] * 100:+.2f}%, "
                 f"ty={p['ty_ratio'] * 100:+.2f}%, scale={p['scale']:.3f}")
            if info:
                s += (f", E保留={info['energy_retention'] * 100:.1f}%, "
                      f"质心=({info['cx']:.0f},{info['cy']:.0f}), "
                      f"贴边={info['border_ratio'] * 100:.1f}%")
            return s + ")"

        msg = (f"[完成] {src_idx} -> {new_idx} | "
               + _fmt(if_params, if_info, "IF") + " | "
               + _fmt(df_params, df_info, "DF"))
        if sync_augment:
            msg += " | 同步"
        if ensure_spot and retries:
            msg += f" | 重采样{retries}次"
        print(msg)
        for w in warnings:
            print(f"[警告] 编号 {src_idx}: {w}")

    print(f"[结束] 共生成 {len(generated)} 组增强数据。")
    return generated


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="远场光斑数据随机增强（平移/旋转/伸缩）")
    parser.add_argument("data_dir", help="数据所在目录（包含 IFimg/DFimg/ZerCof 文件）")
    parser.add_argument("--output-dir", default=None,
                        help="增强结果输出目录，默认写回原目录")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    parser.add_argument("--rotation", type=float, default=ROTATION_RANGE,
                        help=f"旋转角度绝对值上限（度），默认 {ROTATION_RANGE}")
    parser.add_argument("--translation", type=float, default=TRANSLATION_RANGE,
                        help="平移比例上限（相对图像宽/高），"
                             f"如 0.05 表示 ±5%%，默认 {TRANSLATION_RANGE}")
    parser.add_argument("--scale-min", type=float, default=SCALE_RANGE[0],
                        help=f"伸缩比例下限，默认 {SCALE_RANGE[0]}")
    parser.add_argument("--scale-max", type=float, default=SCALE_RANGE[1],
                        help=f"伸缩比例上限，默认 {SCALE_RANGE[1]}")
    parser.add_argument("--sync", action="store_true",
                        help="同组 IFimg/DFimg 使用完全相同的增强参数（默认各自独立）")
    parser.add_argument("--no-spot-check", action="store_true",
                        help="关闭光斑保留校验（默认开启，不合格参数会自动重采样）")
    parser.add_argument("--min-energy-retention", type=float,
                        default=MIN_ENERGY_RETENTION,
                        help=f"光斑能量保留率下限，默认 {MIN_ENERGY_RETENTION}")
    parser.add_argument("--centroid-margin", type=float,
                        default=CENTROID_MARGIN_RATIO,
                        help="光斑质心距边界最小距离比例，"
                             f"默认 {CENTROID_MARGIN_RATIO}")
    parser.add_argument("--max-border-ratio", type=float,
                        default=MAX_BORDER_RATIO,
                        help=f"光斑贴边像素占比上限，默认 {MAX_BORDER_RATIO}")
    parser.add_argument("--overwrite", action="store_true",
                        help="目标文件已存在时允许覆盖")
    args = parser.parse_args()

    augment_dataset(
        args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        rotation_range=args.rotation,
        translation_range=args.translation,
        scale_range=(args.scale_min, args.scale_max),
        sync_augment=args.sync,
        ensure_spot=not args.no_spot_check,
        min_energy_retention=args.min_energy_retention,
        centroid_margin_ratio=args.centroid_margin,
        max_border_ratio=args.max_border_ratio,
        overwrite=args.overwrite,
    )
