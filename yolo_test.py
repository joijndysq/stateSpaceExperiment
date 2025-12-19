import os
# 不再强制设置 QT_QPA_PLATFORM='offscreen'（某些系统无该插件会导致 Qt 初始化失败）
# 检测是否有可用的显示（DISPLAY 环境变量），用于决定是否调用 cv2.imshow
has_display = bool(os.environ.get("DISPLAY"))
if not has_display:
    print("无可用 DISPLAY，运行在 headless 模式，已禁用窗口显示（不调用 cv2.imshow）。")
import time
import sys
from ultralytics import YOLO
import numpy as np
import cv2


# 使用固定模型与摄像头
model = YOLO('best.pt')
names = getattr(model, "names", {})
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    sys.exit(1)

# 初始化 Kalman Filter（4 状态: x,y,vx,vy ; 2 观测: x,y）
kalman = cv2.KalmanFilter(4, 2)
dt_init = 1/30.0
kalman.transitionMatrix = np.array([
    [1, 0, dt_init, 0],
    [0, 1, 0, dt_init],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)
kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=np.float32)
q = 1e-2
r = 5e-3
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * q
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
measurement = np.zeros((2,1), dtype=np.float32)

prev_time = time.time()  # 用于计算 dt


try:
    while True:
        # 先读取一帧
        ret, frame = cap.read()
        if not ret:
            break

        # 在每帧开始处计算 dt 并更新 transitionMatrix，随后 predict
        now = time.time()
        dt = now - prev_time if (now - prev_time) > 0 else dt_init
        prev_time = now
        # 更新状态转移矩阵
        kalman.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        start = time.time()
        # 先做预测（用于无检测时显示或备用）
        pred = kalman.predict()
        pred = pred.flatten()  # 确保一维索引安全
        pred_x, pred_y = int(pred[0]), int(pred[1])

        results = model(frame)[0]

        boxes = getattr(results, "boxes", None)
        if boxes is not None and len(boxes.xyxy) > 0:
            for i, xyxy in enumerate(boxes.xyxy):
                xy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy
                x1, y1, x2, y2 = map(int, xy.tolist())

                # 计算中心点并记录
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                conf = float(boxes.conf[i].cpu().numpy()) if hasattr(boxes.conf[i], "cpu") else float(boxes.conf[i])
                cls_idx = int(boxes.cls[i].cpu().numpy()) if hasattr(boxes.cls[i], "cpu") else int(boxes.cls[i])
                label = f"{names.get(cls_idx, str(cls_idx))} {conf:.2f}"

                # 绘制检测框与标签
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 6, y1), (0, 200, 0), -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # 绘制原始中心点（红色）
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                # 用测量更新 Kalman，并绘制平滑后点（蓝色）
                measurement[0,0] = float(cx)
                measurement[1,0] = float(cy)
                corrected = kalman.correct(measurement)
                corrected = corrected.flatten()
                corr_x, corr_y = int(corrected[0]), int(corrected[1])
                cv2.circle(frame, (corr_x, corr_y), 5, (255, 0, 0), -1)

        # 显示 FPS；仅当有显示环境时才调用 cv2.imshow / waitKey
        fps = 1.0 / (time.time() - start + 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if has_display:
            cv2.imshow("YOLOv8 - Live Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # headless：短暂睡眠以降低 CPU 占用，继续循环
            time.sleep(0.01)
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    # 仅在有显示时销毁窗口，避免 headless 下的 Qt 问题
    if has_display:
        cv2.destroyAllWindows()

