"""
合成工业缺陷图像生成器
在没有真实缺陷数据的情况下，生成带标注的合成数据用于 Demo 训练。
"""

import os
import sys
import cv2
import numpy as np
import random
from pathlib import Path

# 把项目根目录加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DEFECT_CLASSES, IMAGE_SIZE, SYNTHETIC_NUM_TRAIN, SYNTHETIC_NUM_VAL, DATA_DIR


def _random_color():
    """生成随机金属表面颜色（灰-银-铜色调）"""
    base = random.randint(100, 220)
    r = min(255, base + random.randint(-20, 20))
    g = min(255, base + random.randint(-15, 15))
    b = min(255, base + random.randint(-10, 25))
    return (b, g, r)


def _add_texture(img):
    """添加金属拉丝/磨砂纹理"""
    h, w = img.shape[:2]
    noise = np.random.randint(-8, 8, (h, w, 3), dtype=np.int16)
    # 水平拉丝效果
    for y in range(0, h, random.randint(1, 3)):
        shift = random.randint(-2, 2)
        if shift != 0:
            img[y, :] = np.roll(img[y, :], shift, axis=0)
    img = img.astype(np.int16) + noise
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_crack(img):
    """绘制裂纹（曲折线段）"""
    h, w = img.shape[:2]
    color = (random.randint(0, 40),) * 3  # 深色裂纹
    pts = []
    x, y = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
    for _ in range(random.randint(4, 10)):
        x += random.randint(-30, 30)
        y += random.randint(-20, 20)
        x = np.clip(x, 0, w-1)
        y = np.clip(y, 0, h-1)
        pts.append((int(x), int(y)))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i+1], color, random.randint(1, 3))
    return _bbox_from_points(pts)


def _draw_scratch(img):
    """绘制划痕（长直线）"""
    h, w = img.shape[:2]
    color = (random.randint(30, 80),) * 3
    angle = random.uniform(0, np.pi)
    length = random.randint(w//3, w//2)
    cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
    dx = int(length * np.cos(angle))
    dy = int(length * np.sin(angle))
    x1, y1 = cx - dx//2, cy - dy//2
    x2, y2 = cx + dx//2, cy + dy//2
    x1, y1 = np.clip(x1, 0, w-1), np.clip(y1, 0, h-1)
    x2, y2 = np.clip(x2, 0, w-1), np.clip(y2, 0, h-1)
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, random.randint(2, 4))
    x_min = min(x1, x2) - 5
    y_min = min(y1, y2) - 5
    x_max = max(x1, x2) + 5
    y_max = max(y1, y2) + 5
    return _clip_bbox(x_min, y_min, x_max, y_max, w, h)


def _draw_pit(img):
    """绘制凹坑（圆形暗斑）"""
    h, w = img.shape[:2]
    color = (random.randint(0, 50),) * 3
    cx, cy = random.randint(w//5, 4*w//5), random.randint(h//5, 4*h//5)
    radius = random.randint(10, 35)
    cx, cy = int(np.clip(cx, radius, w-radius)), int(np.clip(cy, radius, h-radius))
    cv2.circle(img, (cx, cy), radius, color, -1)
    # 边缘柔化
    cv2.circle(img, (cx, cy), radius + 2, (color[0]+30, color[1]+30, color[2]+30), 1)
    return _clip_bbox(cx-radius-3, cy-radius-3, cx+radius+3, cy+radius+3, w, h)


def _draw_inclusion(img):
    """绘制夹杂（不规则异色斑点）"""
    h, w = img.shape[:2]
    color = (random.randint(60, 180), random.randint(30, 90), random.randint(30, 90))
    cx, cy = random.randint(w//5, 4*w//5), random.randint(h//5, 4*h//5)
    pts = []
    radius = random.randint(10, 30)
    for angle in np.linspace(0, 2*np.pi, random.randint(6, 10)):
        r = radius + random.randint(-8, 8)
        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))
        pts.append((np.clip(x, 0, w-1), np.clip(y, 0, h-1)))
    cv2.fillPoly(img, [np.array(pts)], color)
    return _bbox_from_points(pts)


def _draw_oxidation(img):
    """绘制氧化皮（大面积不规则变色）"""
    h, w = img.shape[:2]
    color = (random.randint(40, 100), random.randint(80, 160), random.randint(120, 200))
    cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
    rx, ry = random.randint(40, 100), random.randint(30, 80)
    overlay = img.copy()
    cv2.ellipse(overlay, (int(cx), int(cy)), (rx, ry), random.randint(0, 180), 0, 360, color, -1)
    result = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    img[:] = result[:]   # 原地修改
    return _clip_bbox(cx-rx-5, cy-ry-5, cx+rx+5, cy+ry+5, w, h)


def _draw_stain(img):
    """绘制斑痕（小块变色区域）"""
    h, w = img.shape[:2]
    color = (random.randint(20, 80), random.randint(40, 100), random.randint(10, 60))
    cx, cy = random.randint(w//5, 4*w//5), random.randint(h//5, 4*h//5)
    radius = random.randint(15, 45)
    overlay = img.copy()
    cv2.circle(overlay, (int(cx), int(cy)), radius, color, -1)
    # 添加不规则边缘
    for _ in range(random.randint(3, 8)):
        angle = random.uniform(0, 2*np.pi)
        r2 = radius + random.randint(-10, 10)
        ex = int(cx + r2 * np.cos(angle))
        ey = int(cy + r2 * np.sin(angle))
        cv2.circle(overlay, (np.clip(ex, 0, w-1), np.clip(ey, 0, h-1)),
                   random.randint(5, 15), color, -1)
    result = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    img[:] = result[:]
    return _clip_bbox(cx-radius-10, cy-radius-10, cx+radius+10, cy+radius+10, w, h)


# ---- 工具函数 ----
def _bbox_from_points(pts):
    """从点集计算 bbox (x_center, y_center, width, height) 归一化"""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return _clip_bbox(min(xs), min(ys), max(xs), max(ys), IMAGE_SIZE, IMAGE_SIZE)


def _clip_bbox(x1, y1, x2, y2, w, h):
    """裁剪 bbox 并转为 YOLO 归一化格式"""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    x_c = (x1 + x2) / 2.0 / w
    y_c = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return (x_c, y_c, bw, bh)


# ---- 缺陷绘制调度 ----
_DRAW_FUNC = {
    0: _draw_crack,
    1: _draw_scratch,
    2: _draw_pit,
    3: _draw_inclusion,
    4: _draw_oxidation,
    5: _draw_stain,
}


def generate_one():
    """生成一张合成图像，返回 (image, yolo_annotations)"""
    color = _random_color()
    img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), color, dtype=np.uint8)
    img = _add_texture(img)

    # 随机放 1~4 个缺陷
    num_defects = random.randint(1, 4)
    annotations = []

    for _ in range(num_defects):
        cls_id = random.randint(0, 5)
        bbox = _DRAW_FUNC[cls_id](img)
        if bbox is not None:
            annotations.append(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")

    return img, annotations


def generate_dataset(output_dir: str, num_images: int):
    """生成数据集"""
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i in range(num_images):
        img, anns = generate_one()
        name = f"defect_{i:04d}"
        cv2.imwrite(f"{img_dir}/{name}.jpg", img)

        if anns:
            with open(f"{lbl_dir}/{name}.txt", "w") as f:
                f.write("\n".join(anns))

        if (i + 1) % 50 == 0:
            print(f"  已生成 {i+1}/{num_images} 张合成图像")

    print(f"  完成：{num_images} 张图像 → {img_dir}")


if __name__ == "__main__":
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    print("生成训练集合成缺陷图像...")
    generate_dataset(train_dir, SYNTHETIC_NUM_TRAIN)
    print("生成验证集合成缺陷图像...")
    generate_dataset(val_dir, SYNTHETIC_NUM_VAL)
    print("合成数据生成完毕！")
