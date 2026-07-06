"""
数据增强模块：Mosaic + Copy-Paste + Albumentations
解决工业场景下缺陷样本稀缺问题。
"""

import os
import sys
import cv2
import numpy as np
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import IMAGE_SIZE, DATA_DIR, AUG_DIR

try:
    import albumentations as A
    HAS_ALBU = True
except ImportError:
    HAS_ALBU = False
    print("[WARN] albumentations 未安装，跳过高级增强")


def mosaic_augment(img_paths: list, label_paths: list, output_size=IMAGE_SIZE):
    """
    Mosaic 增强：将 4 张图片拼接为 1 张
    解决：小目标缺陷容易被漏检，通过拼接增加上下文多样性
    """
    if len(img_paths) < 4:
        # 不足 4 张就复制
        img_paths = img_paths * (4 // len(img_paths) + 1)
    selected = random.sample(img_paths, 4)
    selected_lbl = random.sample(label_paths, 4)

    mosaic_img = np.zeros((output_size * 2, output_size * 2, 3), dtype=np.uint8)
    mosaic_labels = []

    # 中心点随机偏移（增加多样性）
    cx = int(random.uniform(output_size // 2, output_size * 3 // 2))
    cy = int(random.uniform(output_size // 2, output_size * 3 // 2))

    positions = [
        (0, 0, 0, cx, 0, cy),                          # 左上
        (0, cx, output_size * 2, 0, 0, cy),            # 右上
        (0, 0, cx, 0, cy, output_size * 2),            # 左下
        (0, cx, output_size * 2, 0, cy, output_size*2),# 右下
    ]

    for idx, (img_path, lbl_path) in enumerate(zip(selected, selected_lbl)):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = random.uniform(0.6, 1.4)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

        _, x1, x2, _, y1, y2 = positions[idx]
        # 随机在区域内放置
        paste_x = random.randint(x1, max(x1+10, x2-new_w))
        paste_y = random.randint(y1, max(y1+10, y2-new_h))
        pw, ph = min(new_w, output_size*2-paste_x), min(new_h, output_size*2-paste_y)

        mosaic_img[paste_y:paste_y+ph, paste_x:paste_x+pw] = img[:ph, :pw]

        # 更新标注坐标
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = parts[0]
                    xc, yc, bw, bh = map(float, parts[1:5])
                    # 转换到 mosaic 坐标系
                    new_xc = (xc * new_w + paste_x) / (output_size * 2)
                    new_yc = (yc * new_h + paste_y) / (output_size * 2)
                    new_bw = (bw * new_w) / (output_size * 2)
                    new_bh = (bh * new_h) / (output_size * 2)
                    mosaic_labels.append(f"{cls} {new_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}")

    # 裁剪到 output_size
    final = mosaic_img[cy-output_size//2:cy+output_size//2, cx-output_size//2:cx+output_size//2]
    if final.shape[:2] != (output_size, output_size):
        final = cv2.resize(final, (output_size, output_size))

    return final, mosaic_labels


def copypaste_augment(img_paths: list, label_paths: list, output_size=IMAGE_SIZE):
    """
    Copy-Paste 增强：将一张图中的缺陷区域复制粘贴到另一张图上
    解决：一张图通常只有 1~2 个缺陷，通过复制增加单图缺陷密度
    """
    if len(img_paths) < 2:
        return None, []

    src_idx, dst_idx = random.sample(range(len(img_paths)), 2)
    dst_img = cv2.imread(img_paths[dst_idx])
    src_img = cv2.imread(img_paths[src_idx])
    if dst_img is None or src_img is None:
        return None, []

    dst_img = cv2.resize(dst_img, (output_size, output_size))
    src_img = cv2.resize(src_img, (output_size, output_size))

    # 读取源图标注，找到缺陷区域
    if not os.path.exists(label_paths[src_idx]):
        return dst_img, []

    new_labels = []

    with open(label_paths[src_idx]) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = parts[0]
            xc, yc, bw, bh = map(float, parts[1:5])
            # 转为像素坐标
            x1 = int((xc - bw/2) * output_size)
            y1 = int((yc - bh/2) * output_size)
            x2 = int((xc + bw/2) * output_size)
            y2 = int((yc + bh/2) * output_size)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(output_size-1, x2), min(output_size-1, y2)

            # 在目标图上随机偏移粘贴
            dx = random.randint(-30, 30)
            dy = random.randint(-30, 30)
            nx1, ny1 = x1 + dx, y1 + dy
            nx2, ny2 = x2 + dx, y2 + dy

            if nx1 < 0 or ny1 < 0 or nx2 >= output_size or ny2 >= output_size:
                continue

            # 复制缺陷区域
            patch = src_img[y1:y2, x1:x2]
            dst_img[ny1:ny2, nx1:nx2] = cv2.addWeighted(
                dst_img[ny1:ny2, nx1:nx2], 0.3, patch, 0.7, 0
            )

            # 记录新标注
            nxc = (nx1 + nx2) / 2.0 / output_size
            nyc = (ny1 + ny2) / 2.0 / output_size
            nbw = (nx2 - nx1) / output_size
            nbh = (ny2 - ny1) / output_size
            new_labels.append(f"{cls} {nxc:.6f} {nyc:.6f} {nbw:.6f} {nbh:.6f}")

    # 保留目标图原有标注
    if os.path.exists(label_paths[dst_idx]):
        with open(label_paths[dst_idx]) as f:
            new_labels.extend([l.strip() for l in f if l.strip()])

    return dst_img, new_labels


def albumentations_pipeline():
    """标准 Albumentations 增强流水线"""
    if not HAS_ALBU:
        return None
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussNoise(std_range=(5.0, 30.0), p=0.4),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def run_augmentation(img_dir: str, lbl_dir: str, out_img_dir: str, out_lbl_dir: str):
    """
    对训练集运行完整增强流程
    输出：原始图像 + Mosaic 增强 + Copy-Paste 增强 + Albumentations 增强
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    lbl_files = [os.path.join(lbl_dir, os.path.splitext(os.path.basename(f))[0] + ".txt") for f in img_files]

    if not img_files:
        print("[WARN] 没有找到图像文件，请先生成合成数据")
        return

    # 先复制原始数据
    for i, (img_f, lbl_f) in enumerate(zip(img_files, lbl_files)):
        cv2.imwrite(f"{out_img_dir}/orig_{i:04d}.jpg", cv2.imread(img_f))
        if os.path.exists(lbl_f):
            with open(lbl_f) as fsrc, open(f"{out_lbl_dir}/orig_{i:04d}.txt", "w") as fdst:
                fdst.write(fsrc.read())

    # Mosaic 增强（生成原始数量 50% 的拼接图像）
    mosaic_count = len(img_files) // 2
    print(f"  生成 Mosaic 增强图像 x{mosaic_count}...")
    for i in range(mosaic_count):
        mosaic_img, mosaic_lbls = mosaic_augment(img_files, lbl_files)
        cv2.imwrite(f"{out_img_dir}/mosaic_{i:04d}.jpg", mosaic_img)
        with open(f"{out_lbl_dir}/mosaic_{i:04d}.txt", "w") as f:
            f.write("\n".join(mosaic_lbls))

    # Copy-Paste 增强
    cp_count = len(img_files) // 3
    print(f"  生成 Copy-Paste 增强图像 x{cp_count}...")
    for i in range(cp_count):
        cp_img, cp_lbls = copypaste_augment(img_files, lbl_files)
        if cp_img is not None:
            cv2.imwrite(f"{out_img_dir}/cp_{i:04d}.jpg", cp_img)
            with open(f"{out_lbl_dir}/cp_{i:04d}.txt", "w") as f:
                f.write("\n".join(cp_lbls))

    # Albumentations 增强
    if HAS_ALBU:
        aug_pipe = albumentations_pipeline()
        if aug_pipe:
            print(f"  生成 Albumentations 增强图像 x{len(img_files)}...")
            for i, img_f in enumerate(img_files):
                img = cv2.imread(img_f)
                if img is None:
                    continue
                # 读取 YOLO bbox
                bboxes, cls_labels = [], []
                lbl_f = os.path.join(lbl_dir, os.path.splitext(os.path.basename(img_f))[0] + ".txt")
                if os.path.exists(lbl_f):
                    with open(lbl_f) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_labels.append(int(parts[0]))
                                bboxes.append([float(x) for x in parts[1:5]])
                if bboxes:
                    try:
                        aug_result = aug_pipe(image=img, bboxes=bboxes, class_labels=cls_labels)
                        cv2.imwrite(f"{out_img_dir}/alb_{i:04d}.jpg", aug_result["image"])
                        with open(f"{out_lbl_dir}/alb_{i:04d}.txt", "w") as f:
                            for b, c in zip(aug_result["bboxes"], aug_result["class_labels"]):
                                f.write(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
                    except Exception:
                        pass

    total = len(os.listdir(out_img_dir))
    print(f"  增强完成！总计 {total} 张图像 → {out_img_dir}")


if __name__ == "__main__":
    train_img = os.path.join(DATA_DIR, "train", "images")
    train_lbl = os.path.join(DATA_DIR, "train", "labels")
    out_img = os.path.join(AUG_DIR, "train", "images")
    out_lbl = os.path.join(AUG_DIR, "train", "labels")
    run_augmentation(train_img, train_lbl, out_img, out_lbl)
