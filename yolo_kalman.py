import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import time
import sys
from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
model = YOLO('best.pt')
names = getattr(model, "names", {}) 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(f"无法打开摄像头")
    sys.exit(1)
# 多组 Q/R 参数（移除 Q=0.01,R=0.01 后共 5 组）
param_list = [
    (0.01, 0.1),
    (0.05, 0.1),
    (0.1, 0.1),
    (0.01, 0.05),
    (0.01, 0.005),  # 原来还有 (0.01,0.01) 已移除
]

#初始化多个Kalman Filter
dt_init = 1 / 30.0
kalman_filters = []
for q, r in param_list:
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([
        [1, 0, dt_init, 0],
        [0, 1, 0, dt_init],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * q
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
    # statePost/ statePre 默认为零
    kalman_filters.append({
        "kf": kf,
        "q": q,
        "r": r,
        "smooth_points": []  #滤波后的点
    })

measurement = np.zeros((2, 1), dtype=np.float32)

# 数据采集设置
N = 1000  #采集点数
raw_points = []      #list of (x,y)原始数据点

prev_time = time.time()
try:
    stop_capture = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #计算帧间隔 dt，并更新每个 kalman 的状态转移矩阵
        now = time.time()
        dt = now - prev_time if now - prev_time > 0 else dt_init
        prev_time = now
        for entry in kalman_filters:
            kf = entry["kf"]
            kf.transitionMatrix = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)

        start = time.time()
        results = model(frame)[0]
        preds = []
        for entry in kalman_filters:
            pred = entry["kf"].predict()
            preds.append((int(pred[0]), int(pred[1])))
        #绘制检测结果并更新滤波器
        boxes = getattr(results, "boxes", None)
        if boxes is not None and len(boxes.xyxy) > 0:
            for i, xyxy in enumerate(boxes.xyxy):
                xy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy
                x1, y1, x2, y2 = map(int, xy.tolist())
                #计算中心点
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                conf = float(boxes.conf[i].cpu().numpy()) if hasattr(boxes.conf[i], "cpu") else float(boxes.conf[i])
                cls_idx = int(boxes.cls[i].cpu().numpy()) if hasattr(boxes.cls[i], "cpu") else int(boxes.cls[i])
                name = names.get(cls_idx, str(cls_idx))
                label = f"{name} ({cx},{cy}) {conf:.2f}"

                #绘制原始中心点
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1) 
                #用测量更新每个 Kalman 并记录各自的校正结果
                measurement[0, 0] = float(cx)
                measurement[1, 0] = float(cy)
                for entry in kalman_filters:
                    corrected = entry["kf"].correct(measurement)
                    corr_x = int(corrected[0])
                    corr_y = int(corrected[1])
                    entry["smooth_points"].append((corr_x, corr_y))
                # 记录原始点（统一长度）
                raw_points.append((cx, cy))
                if len(raw_points) >= N:
                    stop_capture = True
                    break
                #在中心点上方绘制标签背景与文字
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = max(0, cx - t_size[0] // 2)
                text_y = max(0, cy - 10 - t_size[1])
                cv2.rectangle(frame, (text_x, text_y), (text_x + t_size[0] + 4, text_y + t_size[1] + 4), (0, 255, 0), -1)
                cv2.putText(frame, label, (text_x + 2, text_y + t_size[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if stop_capture:
                break
        # 绘制每组 Kalman 的当前平滑点（蓝色系不同亮度）
        # 至少为每组参数提供一种颜色（BGR）
        colors = [
            (255, 0, 0),    # red-ish
            (200, 100, 0),  # brownish
            (0, 255, 255),  # yellow-cyan
            (0, 200, 200),  # cyan
            (100, 100, 255),# light blue
            (0, 255, 0)     # green 
        ]
        for idx, entry in enumerate(kalman_filters):
            # 如果当前已有平滑点则画最近一个，否则画预测点
            if len(entry["smooth_points"]) > 0:
                px, py = entry["smooth_points"][-1]
            else:
                px, py = preds[idx]
            cv2.circle(frame, (px, py), 5, colors[idx % len(colors)], -1)
        time.sleep(0.001)
        if len(raw_points) % 100 == 0 and len(raw_points) > 0:
            print(f"采集进度: {len(raw_points)}/{N}")
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()

if len(raw_points) == 0:
    print("未采集到任何点，退出。")
    sys.exit(0)

raw_arr = np.array(raw_points[:N])

#为每组确保有 N 个滤波点
for entry in kalman_filters:
    sp = np.array(entry["smooth_points"][:N])
    if len(sp) < len(raw_arr):
        # 若某组少于 raw 的点，用预测填充末尾（用最后的预测/校正点重复）
        if len(sp) == 0:
            # 若完全没有点，填充原始点以避免空数组（极端情况）
            sp = raw_arr.copy()
        else:
            last = sp[-1]
            pad = np.tile(last, (len(raw_arr) - len(sp), 1))
            sp = np.vstack([sp, pad])
    entry["smooth_arr"] = sp

def compute_jitter(arr):
    if len(arr) < 2:
        return 0.0, 0.0
    diffs = arr[1:] - arr[:-1]
    dists = np.linalg.norm(diffs, axis=1)
    return float(dists.mean()), float(dists.std())

def compute_smoothness(arr):
    if len(arr) < 3:
        return 0.0, 0.0
    acc = arr[2:] - 2 * arr[1:-1] + arr[:-2]
    mags = np.linalg.norm(acc, axis=1)
    return float(mags.mean()), float(mags.std())

metrics_lines = []
#轨迹图
plt.figure(figsize=(8, 8))
plt.scatter(raw_arr[:, 0], raw_arr[:, 1], c='red', s=2, label='raw', alpha=0.6)
for idx, entry in enumerate(kalman_filters):
    sp = entry["smooth_arr"]
    label = f"Q={entry['q']},R={entry['r']}"
    plt.scatter(sp[:, 0], sp[:, 1], s=2, label=label, alpha=0.6)
plt.gca().invert_yaxis()
plt.title('Trajectory')
plt.legend()
plt.axis('equal')
traj_path = "kalman_multi_trajectory.png"
plt.tight_layout()
plt.savefig(traj_path, dpi=300)
plt.close()


# 为每组分别绘制时间序列对比并计算指标
for idx, entry in enumerate(kalman_filters):
    sp = entry["smooth_arr"]
    #时间序列图
    plt.figure(figsize=(12, 4))
    t = np.arange(len(raw_arr))
    plt.plot(t, raw_arr[:, 0], color='red', linewidth=0.8, label='raw_x')
    plt.plot(t, sp[:, 0], color='blue', linewidth=0.8, label='kalman_x')
    plt.plot(t, raw_arr[:, 1], color='orange', linewidth=0.8, label='raw_y', alpha=0.7)
    plt.plot(t, sp[:, 1], color='green', linewidth=0.8, label='kalman_y', alpha=0.7)
    plt.title(f"Time series  Q={entry['q']} R={entry['r']}")
    plt.legend(loc='upper right')
    time_path = f"timeseries_q{entry['q']}_r{entry['r']}.png".replace('.', 'p')
    plt.tight_layout()
    plt.savefig(time_path, dpi=300)
    plt.close()

    #计算：抖动、平滑度、MSE/RMSE
    raw_j_mean, raw_j_std = compute_jitter(raw_arr)
    sm_j_mean, sm_j_std = compute_jitter(sp)
    raw_s_mean, raw_s_std = compute_smoothness(raw_arr)
    sm_s_mean, sm_s_std = compute_smoothness(sp)

    diff = raw_arr - sp
    mse_per_axis = np.mean(diff**2, axis=0)
    mse_total = np.mean(np.sum(diff**2, axis=1))
    rmse_per_axis = np.sqrt(mse_per_axis)
    rmse_total = np.sqrt(mse_total)

    jitter_reduction = (raw_j_mean - sm_j_mean) / (raw_j_mean if raw_j_mean != 0 else 1.0)
    smoothness_reduction = (raw_s_mean - sm_s_mean) / (raw_s_mean if raw_s_mean != 0 else 1.0)

    metrics_lines.append(
        f"Params Q={entry['q']}, R={entry['r']}\n"
        f"  Jitter - raw: mean={raw_j_mean:.4f}, std={raw_j_std:.4f}; kalman: mean={sm_j_mean:.4f}, std={sm_j_std:.4f}; reduction={jitter_reduction*100:.2f}%\n"
        f"  Smoothness - raw: mean={raw_s_mean:.4f}, std={raw_s_std:.4f}; kalman: mean={sm_s_mean:.4f}, std={sm_s_std:.4f}; reduction={smoothness_reduction*100:.2f}%\n"
        f"  MSE per axis: x={mse_per_axis[0]:.4f}, y={mse_per_axis[1]:.4f}; MSE total={mse_total:.4f}\n"
        f"  RMSE per axis: x={rmse_per_axis[0]:.4f}, y={rmse_per_axis[1]:.4f}; RMSE total={rmse_total:.4f}\n"
    )
metrics_text = f"Samples: {len(raw_arr)}\n\n" + "\n\n".join(metrics_lines)
with open("kalman_multi.txt", "w") as f:
    f.write(metrics_text)
print("数据保存: kalman_multi.txt")

