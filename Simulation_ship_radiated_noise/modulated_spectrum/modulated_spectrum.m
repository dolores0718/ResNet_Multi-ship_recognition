% 参数设置
n = 4;                       % 螺旋桨叶片数量
fp = 10;                     % 螺旋桨轴频 (Hz)
fb = fp * n;                 % 螺旋桨叶频
m = 4;                       % 螺旋桨常数
% T0 = 1 / fb;                 % 脉冲间隔时间
T0 = 0.2
nb = 5;                     % 每个轴频周期内的脉冲数量
Emax = 1;                    % 每个周期的首个脉冲幅度
Emean = 0.5
fs = 10000;                   % 采样率 (Hz)
T = 5;                       % 信号持续时间 (秒)
% pulse_width = T0/ (m * n);    % 脉冲宽度sigma
pulse_width = 0.05
n_pulses = round(nb * T);    % 脉冲的数量


% 时间轴
t = 0:1/fs:T-1/fs;    % 时间轴

% 初始化调制谱信号
mt = zeros(1, length(t));

% 生成调制谱信号
for cycle = 1:ceil(n_pulses / nb)
    % 计算周期的起始时间
    cycle_start_time = (cycle - 1) * nb * T0;

    % 为每个周期生成脉冲
    for i = 1:nb
        % 当前脉冲时间点
        pulse_time = cycle_start_time + (i - 1) * T0;

        % 检查是否超出信号时间范围
        if pulse_time >= T
            break;
        end

        % 设置脉冲幅度
        if i == 1
            Ei = Emax;  % 每个周期首个脉冲取最大幅度
        else
            Ei = Emean / 2 + rand() * (3 * Emean / 2 - Emean / 2);  % 其他脉冲随机幅度
        end

        % 生成高斯脉冲并叠加
        pulse_signal = (Ei / sqrt(2 * pi)) * ...
                       exp(-((t - pulse_time).^2) / (2 * pulse_width^2));
        mt = mt + pulse_signal;
    end
end

normalized_signal = mt / max(abs(mt));

% 绘制时域波形
figure;
subplot(2, 1, 1);
plot(t, normalized_signal);
title('Modulating spectral signal time domain waveform');
xlabel('Time(s)');
ylabel('Normalized amplitude');
grid on;

subplot(2, 1, 2);
MT = fftshift(fft(mt./(n_pulses))); %用fft得出离散傅里叶变换
N = length(mt)
f=linspace(-fs/2, fs/2-1, N);  %频域横坐标，注意奈奎斯特采样定理，最大原信号最大频率不超过采样频率的一半
f1 = 10 * f
plot(f1,abs(MT));    %画双侧频谱幅度图
title('Spectogram');
xlabel("Frequency/Hz");
ylabel("Power(dB)");
xlim([0 200]);
grid on;