"""
ONNX Runtime 推理脚本
加载 ONNX 模型，对单张图片进行缺陷检测，打印推理耗时和检测结果。
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DEFECT_CLASSES, ONNX_MODEL_PATH, CONF_THRESH, IOU_THRESH, MODEL_INPUT_SIZE


def letterbox(img, new_shape=640):
    """保持宽高比缩放 + 填充为正方形"""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_h, new_w = int(h * r), int(w * r)
    img = cv2.resize(img, (new_w, new_h))
    dw, dh = (new_shape - new_w) // 2, (new_shape - new_h) // 2
    padded = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    padded[dh:dh+new_h, dw:dw+new_w] = img
    return padded, r, dw, dh


def nms(boxes, scores, iou_thresh):
    """简易 NMS"""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        iou = w * h / areas[order[1:]]
        order = order[1:][iou < iou_thresh]
    return keep


def infer_onnx(image_path: str, onnx_path: str = None):
    """ONNX 推理单张图片，打印耗时与检测结果"""
    onnx_path = onnx_path or ONNX_MODEL_PATH

    if not os.path.exists(onnx_path):
        print(f"错误：ONNX 模型不存在 {onnx_path}")
        print("请先运行: python export/export_onnx.py")
        return None

    # ---- 1. 加载图像 ----
    if not os.path.exists(image_path):
        # 如果没有指定图像，从验证集随机选一张
        from config.settings import DATA_DIR
        val_dir = os.path.join(DATA_DIR, "val", "images")
        if os.path.exists(val_dir):
            imgs = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png'))]
            if imgs:
                image_path = os.path.join(val_dir, imgs[0])
        if not os.path.exists(image_path):
            print(f"错误：图像文件不存在 {image_path}")
            return None

    print(f"推理图像: {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"错误：无法读取图像 {image_path}")
        return None

    orig_h, orig_w = img_bgr.shape[:2]

    # ---- 2. 预处理 ----
    img_padded, ratio, dw, dh = letterbox(img_bgr, MODEL_INPUT_SIZE)
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))  # HWC → CHW
    img_input = np.expand_dims(img_input, axis=0)    # (1, 3, 160, 160)

    # ---- 3. ONNX 推理 ----
    try:
        import onnxruntime as ort
    except ImportError:
        print("错误：请安装 onnxruntime: pip install onnxruntime")
        return None

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # 正式推理（测速取多次平均）
    print("预热推理...")
    _ = session.run(None, {input_name: img_input.astype(np.float32)})

    print("测速推理（10 次取平均）...")
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: img_input.astype(np.float32)})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_ms = np.mean(times)

    # ---- 4. 解析输出 ----
    # YOLOv8 ONNX 输出 shape: (1, 84, 8400) = (1, 4+80, 8400)
    # 我们的模型只有 6 类，所以是 (1, 10, 8400) = (1, 4+6, 8400)
    output = outputs[0]  # (1, 4+nc, num_anchors)
    output = np.transpose(output[0], (1, 0))  # (num_anchors, 4+nc)

    boxes_xywh = output[:, :4]
    scores_all = output[:, 4:]

    # 获取每个 anchor 的最大置信度和类别
    class_ids = np.argmax(scores_all, axis=1)
    confs = np.max(scores_all, axis=1)

    # 过滤低置信度
    mask = confs > CONF_THRESH
    boxes_xywh = boxes_xywh[mask]
    confs = confs[mask]
    class_ids = class_ids[mask]

    # xywh → xyxy
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

    # 缩放到填充尺寸
    boxes_xyxy *= MODEL_INPUT_SIZE

    # NMS
    keep = nms(boxes_xyxy, confs, IOU_THRESH)
    boxes_xyxy = boxes_xyxy[keep]
    confs = confs[keep]
    class_ids = class_ids[keep]

    # ---- 5. 将坐标映射回原图 ----
    boxes_xyxy[:, 0] -= dw
    boxes_xyxy[:, 1] -= dh
    boxes_xyxy /= ratio

    # 裁剪到图像范围
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    # ---- 6. 打印结果 ----
    print(f"\n{'='*50}")
    print(f"  推理耗时: {avg_ms:.2f} ms (平均 10 次)")
    print(f"  检测到 {len(confs)} 个缺陷:")
    print(f"{'─'*50}")

    for i in range(len(confs)):
        cls_name = DEFECT_CLASSES[class_ids[i]] if class_ids[i] < len(DEFECT_CLASSES) else f"class_{class_ids[i]}"
        b = boxes_xyxy[i].astype(int)
        print(f"  [{cls_name:12s}]  置信度: {confs[i]:.3f}  |  "
              f"bbox: ({b[0]}, {b[1]}) → ({b[2]}, {b[3]})")

    if len(confs) == 0:
        print("  (无缺陷检出)")

    print(f"{'='*50}\n")

    # ---- 7. 绘制结果 ----
    img_vis = img_bgr.copy()
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]

    for i in range(len(confs)):
        b = boxes_xyxy[i].astype(int)
        color = colors[class_ids[i] % len(colors)]
        cv2.rectangle(img_vis, (b[0], b[1]), (b[2], b[3]), color, 2)
        label = f"{DEFECT_CLASSES[class_ids[i]]} {confs[i]:.2f}"
        cv2.putText(img_vis, label, (b[0], b[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 耗时叠加
    cv2.putText(img_vis, f"Inference: {avg_ms:.1f}ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out_path = "output/inference_result.jpg"
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(out_path, img_vis)
    print(f"结果图像保存至: {out_path}")

    return {"time_ms": avg_ms, "detections": len(confs), "image": out_path}


if __name__ == "__main__":
    # 支持命令行指定图像路径
    img_path = sys.argv[1] if len(sys.argv) > 1 else ""
    infer_onnx(img_path)
