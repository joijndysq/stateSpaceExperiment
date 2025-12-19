%% 一维车辆运动的卡尔曼滤波仿真（修正版）
% 状态变量：x = [p; v]
% 控制变量：u = 加速度
% 真实系统含加速度噪声
% 测量变量：y = p
% KF 估计状态 x_hat = [p; v]

clear; clc;

dt = 0.1;      % 采样周期
N = 100;       % 仿真步数

%% 状态转移矩阵
A = [1, dt;
     0, 1];

B = [0;
     dt];

C = [1, 0];

%% 噪声设置
sigma_a = 0.2;      % 加速度噪声标准差（过程噪声）
R = 1;              % 测量噪声方差

% 过程噪声离散化（加速度噪声 → 状态噪声）
Q = sigma_a^2 * [dt^4/4, dt^3/2;
                 dt^3/2, dt^2];

%% 真实初始状态
xreal = [0.2; 0];

%% KF 初始状态
xh = [0; 0];    % 初始估计
P = eye(2);     % 初始协方差

%% 控制输入（恒定加速度）
u = ones(1, N);

%% 噪声生成
w = mvnrnd([0 0], Q, N)';  % Nx 随机过程噪声
v = sqrt(R) * randn(1, N); % 测量噪声

%% 结果记录
y_real_list = zeros(1, N);
y_meas_list = zeros(1, N);
y_est_list  = zeros(1, N);
cov_list    = zeros(1, N);   % 估计误差方差 C*P*C'

%% ----------------------- 主循环 -----------------------
for k = 1:N

    % 真实系统状态演化
    xreal = A * xreal + B * u(k) + w(:, k);

    % 真实测量
    y_real = C * xreal;

    % 含噪声测量
    yv = y_real + v(k);

    % ====== KF 预测 ======
    x_pred = A * xh + B * u(k);
    P_pred = A * P * A' + Q;

    % ====== KF 更新 ======
    K = P_pred * C' / (C * P_pred * C' + R);
    xh = x_pred + K * (yv - C * x_pred);
    P = (eye(2) - K * C) * P_pred;

    % ====== 记录 ======
    y_real_list(k) = y_real;
    y_meas_list(k) = yv;
    y_est_list(k)  = C * xh;
    cov_list(k)    = C * P * C';
end

%% ===================== 作 图 =====================
t = (1:N) * dt;

figure(1);
subplot(2,2,1)
plot(t, y_real_list, t, y_meas_list, t, y_est_list, 'LineWidth',1.3)
legend('真实','测量','估计','Location','best')
title('车辆位移')
xlabel('时间/s'); ylabel('位置/m');

subplot(2,2,2)
plot(t, y_meas_list - y_real_list, t, y_est_list - y_real_list, 'LineWidth',1.3)
legend('测量误差','估计误差','Location','best')
title('位移误差')
xlabel('时间/s'); ylabel('误差/m');

subplot(2,2,[3 4])
plot(t, cov_list, 'LineWidth',1.3)
title('估计误差方差 C P C^T')
xlabel('时间/s'); ylabel('方差');
legend('方差','Location','best')

