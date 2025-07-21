% 输入参数
V = input('请输入舰船的航速：');      % 舰船的实际航速
V_max = input('请输入最高航速：');    % 舰船的最高航速，单位/节
DT = input('请输入排水量：');         % 舰船的排水量，单位/吨

SLs = 122 + 50*log(V/10) + 15*log(DT);   % 100Hz以上声级的经验公式,单位/dB

% 计算谱峰频率 f0
if V >= 10
    f0 = 300 - 200 * (V - 10) / 30;  % For V >= 10 knots
else
    f0 = 300;  % For V < 10 knots
end

SLf0 = SLs + 20 - 20*log(f0);            % 谱峰处声压谱级

% 设定频谱斜率
K1 = 1;                            % 低于谱峰频率上升斜率
K2 = -20;                           % 高于谱峰频率下降的斜率

% 计算频率点和对应的声压谱级
fk = 2.^(1:log2(f0));  % [1, f0) 区间的倍频程采样
fl = 2.^(log2(f0):log2(5000));  % [f0, fs/2] 区间的倍频程采样
f = [fk, f0, fl];  % 合并频率点

% 对应的幅度值
SLk = SLs + K1 * log(f0 ./ fk);  % 低频段的幅度值
SLl = SLf0 + K2 * log(fl ./ f0); % 高频段的幅度值
m = [SLk, SLf0, SLl];  % 对应的幅度值

% 确保频率点和幅度数组长度一致
if length(f) ~= length(m)
    error('频率点和幅度值的数量不一致，请检查频率和幅度计算方法');
end

% 归一化频率和幅度
f_max = max(f);  % 获取最大频率
f_prime = f / f_max;  % 归一化频率
m_prime = 10.^(m / 20); % 归一化幅度（幅度转换为比例）

% 第一个频率必须是 0，最后一个频率必须是 1
f_prime = [0, f_prime];  % 添加频率0
m_prime = [m_prime(1), m_prime];  % 对应幅度，确保幅度与频率对齐

% 设计 FIR 滤波器
n = 128;  % 滤波器的阶数
b = fir2(n, f_prime, m_prime);  % 使用fir2函数设计滤波器

% 生成高斯白噪声
x = randn(1, 100000);  % 生成高斯白噪声

% 使用滤波器进行滤波
gx = filter(b, 1, x);  % 应用 FIR 滤波器
gx_normalized_minus1_1 = gx / max(abs(gx));
N = length(gx_normalized_minus1_1);  % 获取信号长度
fs = 10000;  % 采样频率
t = (0:N-1) / fs;  % 时间向量

% 绘制时域图
figure;
subplot(2, 1, 1);  % 使用子图分两个画面
plot(t,gx_normalized_minus1_1);
title('Filtered Ship Noise Simulation (Time Domain)');
xlabel('Time');
ylabel('Amplitude');

% FFT计算
f_fft = linspace(0, fs / 2, N / 2);  % 频率向量（单位：Hz）
Gx_fft = abs(fft(gx));  % 计算FFT并取绝对值
Gx_fft = Gx_fft(1:N/2);  % 只取前半部分
Gx = 20*log10(Gx_fft)-100

% 绘制频域图
subplot(2, 1, 2);
plot(f_fft, Gx);  % 使用对数坐标轴绘制频域图
title('Filtered Ship Noise Simulation (Frequency Domain)');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
xlim([1, 5000]);  % 设置x轴的频率范围

% 显示图形
grid on;
