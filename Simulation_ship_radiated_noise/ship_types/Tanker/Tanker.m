clc;
clear all;
clf;

Fs = 32000;
n = 6; % Number of propeller blades
s = 1; % Propeller speed

%%%% Below 100Hz, the line spectrum frequencies can be represented as follows %%%%
Ts = 0:1/Fs:2.5; % Time series
ni = 8; % Eight line spectra
Ak = 0.5 * rand(1,10); % Line spectrum amplitude
Ik = 0.25 * pi * rand(1,10); % Random phases

% Calculate the line spectrum frequencies
f1 = zeros(1, ni);
C = [];
for m1 = 1:ni
    f1(m1) = m1 * n * s;
end
f = f1;
disp(f)

%%% K represents the number of line spectra, Ak, f, Ik correspond to the amplitude, frequency, and phase of the k-th line spectrum
%% Calculate the power spectrum %%%
fi = 80;
L = 4; % The highest frequency of line spectra cannot exceed 100Hz, with the highest being ni*fi/L
fait = 0.25 * pi;
temp = 1:1:ni;
G = 0;
for i = 1:1:ni               
   G1 = Ak(i) * sin(temp(i) * (2 * pi * fi / L * Ts + fait) + Ik(i));
   G = G + G1;
end

G = G + 7.5 * sin(2 * pi * 350 * Ts) + 6.5 * sin(2 * pi * 700 * Ts);

figure;
plot(Ts, G, 'k');
xlabel('Time t/s');
ylabel('Amplitude V');
title('Line Spectrum Time Domain Waveform');

%% 

% Power spectrum analysis
window = boxcar(length(G)); % Rectangular window 
nfft = 3000; 
[Pxxm, f] = periodogram(G, window, nfft, Fs); % Direct method 
PxxM = 10 * log10(Pxxm);

figure(); 
plot(f, PxxM, 'k');
xlabel('f/Hz');
ylabel('Power Spectrum/dB'); % Plot the graph before 3000
title('Line Spectrum');
xlim([0, 1000]);

%%%% White noise construction using a FIR filter with a specific response %%%%

%%%% Use Box-Muller method to generate normally distributed random numbers (mean = 0, std = 1) %%%%

m = 10;
n1 = 5;
t0 = 5000; % Generate normally distributed random numbers

for i = 1:t0   
    a = rand;    
    b = rand;     
    X1(i) = sqrt((-2) * log10(a)) * cos(2 * pi * b);   
    X2(i) = sqrt((-2) * log10(a)) * sin(2 * pi * b);     
    Y1 = X1 * n1 + m;    
    Y2 = X2 * n1 + m; 
end
disp(Y1); % Calculate the mean and standard deviation 
M1 = mean(Y1);
N1 = std(Y1);
disp(M1);
disp(N1); 
disp(Y1); % Calculate the mean and standard deviation 
M2 = mean(Y2);
N2 = std(Y2); 
disp(M2); 
disp(N2); 
h1 = lillietest(Y1);     % If h1 = 1, it means the null hypothesis is rejected, otherwise, it is accepted (i.e., the original distribution is normal)
disp(h1); 
h2 = lillietest(Y2);     % If h2 = 1, it means the null hypothesis is rejected, otherwise, it is accepted (i.e., the original distribution is normal) 
disp(h2);

To = 0.02; 
y = Y2;
N = length(y);
A(1:N) = exp(-To); 
C(1:N) = 1; 
R(1:N) = 1; 
Q(1:N) = 1 - exp(-2 * To); 
P(1) = 1; 

x(1) = y(1); % x(0) = 0; H(1) = 1; 
for n = 2:N 
    P1(n) = A(n) * A(n) * P(n - 1) + Q(n); 
    H(n) = C(n) * P1(n) / (C(n) * C(n) * P1(n) + R(n)); 
    x(n) = A(n) * x(n - 1) + H(n) * (y(n) - C(n) * A(n) * x(n - 1)); 
    P(n) = (1 - C(n) * H(n)) * P1(n); 
end 
i = [1:N]; 
figure(3)
subplot(311)
% x = x(n) / abs(x(n));
plot(i, x, 'k') 
xlabel('Sample Points'); ylabel('Amplitude'); title('White Noise Time Domain Waveform');
set(get(gca, 'XLabel'), 'FontSize', 12); % Graph text size set to 12
set(get(gca, 'YLabel'), 'FontSize', 12);
set(get(gca, 'TITLE'), 'FontSize', 10);
% grid on; 

%%%%% Power Spectrum of White Noise %%%%%%%%
subplot(3,1,2);
P = abs(fft(x)); % Spectrum
fm = (Fs / 2) * N / Fs;
f = (0: fm) * Fs / N;

plot(f, P(1:length(f)), 'k');
% grid on
title('Power Spectrum of White Noise'); xlabel('Frequency'); ylabel('Amplitude');
set(get(gca, 'XLabel'), 'FontSize', 12); % Graph text size set to 12
set(get(gca, 'YLabel'), 'FontSize', 12);
set(get(gca, 'TITLE'), 'FontSize', 10);

CX = xcorr(x, 'unbiased'); % Calculate autocorrelation function of sequence x(n)
CXk = fft(CX, N);
Px = abs(CXk);
index = 0: round(N / 2);
k = index * Fs / N;
Px0 = 10 * log10(Px(index + 1));
subplot(313)
plot(k, Px0, 'k'); 
% grid on;
xlabel('Frequency'); ylabel('Power Spectrum/dB');
title('Power Spectrum of White Noise');
set(get(gca, 'XLabel'), 'FontSize', 12); % Graph text size set to 12
set(get(gca, 'YLabel'), 'FontSize', 12);
set(get(gca, 'TITLE'), 'FontSize', 10);

%% 

%%%% Ideal Ship Radiated Noise Model %%%%
V = 13; % Ship speed
DT = 25000; % Ship displacement, in tons

SLs = 112 + 50 * log(V / 10) + 15 * log(DT); % Empirical formula for sound level above 100Hz, in dB
% Calculate the spectral peak frequency f0
if V >= 10
    f0 = 300 - 200 * (V - 10) / 30;  % For V >= 10 knots
else
    f0 = 300;  % For V < 10 knots
end

SLf0 = SLs + 20 - 20 * log(f0); % Sound pressure level at the spectral peak

K1 = 1; % Slope below the spectral peak frequency
K2 = -20; % Slope above the spectral peak frequency

%%%% Model the continuous spectrum %%%%
ff = 0:1:10000;
SLf = (SLf0 - K1 * log(f0 ./ ff)) .* (ff < f0) + SLf0 .* (ff == f0) + (SLf0 + K2 * log(ff ./ f0)) .* (f0 < ff);

figure(5)
plot(ff, SLf, 'k');
xlabel('Frequency/Hz'); xlim([0, 1100])
ylabel('Sound Pressure Level/dB'); % ylim([160, 200])
title('Expected Continuous Spectrum Curve')
% grid on;
%% 

% Periodic components
Nf = 5000;
n = 0:Nf-1;
t = n / Fs;
f = (0:Nf-1) * Fs / Nf;
Nfft = 2048;
si = 3 * sin(2 * pi * 60 * t) + 2 * sin(2 * pi * 90 * t) + 1 * sin(2 * pi * 120 * t);

% Signal = wgn(2048, 1, 60); % White noise with SNR -20.93 (60), -11.08 (50), 0.219 (35)

[N0, Wc] = buttord([0.05, 0.051], [0.025, 0.1], 6, 9); % wp = 300 / 6000 = 0.05, ws1 = 150 / 6000 = 0.025, ws2 = 600 / 6000 = 0.01
[b, a] = butter(N0, Wc);
xn1 = filter(b, a, x');
figure(6);
subplot(211)
plot(xn1, 'k');
xlabel('Sample Points')
ylabel('Amplitude')
title('Continuous Spectrum Time Domain Waveform')
xn = xn1 + (si(1:length(xn1)))';

subplot(212)
N1 = 2048;
Pxx = 10 * log10(abs(fft(xn, Nfft).^2));
f11 = (0: length(Pxx)-1) * Fs / length(Pxx);
plot(f11, Pxx, 'k');
xlabel('f/Hz');
ylabel('Power Spectrum/dB'); title('Continuous Spectrum')
xlim([0 3000]); % Plot before 3000

% [t1, r1] = size(Pxx);
% [t2, r2] = size(PxxM);
% B1 = zeros((t2 - t1), 1); B = [Pxx; B1];


% 计算 Power Spectrum 和 Time Domain Waveform
PP = Pxx(1:length(PxxM)) + PxxM;
ss = (G((1:length(xn))))' + xn;
PP = Pxx(1:length(PxxM)) + PxxM+100;
figure;
subplot(2,1,2);
plot(f11(1:length(PP)), PP)
xlabel('f/Hz');
ylabel('Power Spectrum/dB');
title('Tanker Radiated Noise');
xlim([0,5000]);

subplot(2,1,1);
ss = (G((1:length(xn))))' + xn;
% tf = t * Ts;
pp = length(ss);
tf = Ts * pp / Fs;
plot(tf(1:length(ss)), ss / norm(ss))
title('Tanker Radiated Noise Time Domain Waveform');
xlabel('Time(s)');
ylabel('Normalized Amplitude');

%% MFCC
signal = ss; % 输入信号
Fs = 32000; % 采样频率

% 参数设置
win_length = 0.025; % 窗长 (秒)
hop_length = 0.01; % 窗间隔 (秒)
n_mfcc = 15; % 提取的MFCC数量
n_fft = 512; % FFT 点数
num_mel_filters = 26; % Mel 滤波器数量

% 1. 预加重
pre_emphasis = 0.97; % 预加重系数
signal = filter([1 -pre_emphasis], 1, signal);

% 2. 分帧
frame_length = round(win_length * Fs); % 每帧采样点数
frame_hop = round(hop_length * Fs); % 帧移采样点数
frames = buffer(signal, frame_length, frame_length - frame_hop, 'nodelay');

% 3. 加窗
hamming_window = hamming(frame_length); % Hamming 窗
frames = frames .* hamming_window;

% 4. 快速傅里叶变换与功率谱
mag_frames = abs(fft(frames, n_fft)); % FFT 变换
pow_frames = (1 / n_fft) * (mag_frames .^ 2); % 功率谱

% 5. Mel 滤波器组
mel_filters = melFilterBank(Fs, n_fft, num_mel_filters); % Mel 滤波器
mel_energy = pow_frames(1:(n_fft / 2 + 1), :)' * mel_filters'; % Mel 滤波后能量

% 6. 对数能量计算
log_mel_energy = log(mel_energy + eps); % 避免 log(0)

% 7. 离散余弦变换 (DCT)
coeffs = dct(log_mel_energy, [], 2); % 计算 DCT
mfcc = coeffs(:, 1:n_mfcc); % 提取前 n_mfcc 个系数

% 8. 绘制 3D MFCC 图
num_frames = size(mfcc, 1); % 总帧数
num_coeffs = size(mfcc, 2); % MFCC 系数数量
[frame_idx, coeff_idx] = meshgrid(1:num_frames, 1:num_coeffs);

% 创建图形
figure;

% 使用 surf 绘制填充颜色的表面，并保存句柄
h = surf(frame_idx, coeff_idx, mfcc', 'FaceColor', 'interp', 'EdgeColor', 'k'); % 填充颜色并添加网格线


% 设置标签和标题
xlabel('Frame Index');
ylabel('MFCC Coefficients');
zlabel('Amplitude');
title('3D MFCC Features of Tanker');
colorbar;
colormap("parula");

% 设置视角
view(60, 30);
set(gcf, 'LineWidth', 2);
% 启用插值着色
shading interp;

%% 设置 STFT 参数
window_length = 1024; % 窗口长度
noverlap = window_length / 2; % 窗口重叠量（50%重叠）
nfft = 1024; % FFT 点数

% 计算信号的 STFT
[~, F, T, P] = spectrogram(ss, window_length, noverlap, nfft, Fs);

% 将功率谱转换为 dB 标度
P_dB = 10 * log10(abs(P));

% 绘制 STFT 时频图
figure;

% 绘制时频图
imagesc(T, F, P_dB); % 时频图
axis xy; % 反转 y 轴，使频率从低到高
xlabel('Time(s)');
ylabel('Frequency (Hz)');
title('STFT of Tanker');
ylim([0,3000]);
colorbar; % 显示色标
caxis([-100 0]); % 设置颜色条范围（dB）
