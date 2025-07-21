clc;
clear all;
clf;

Fs = 5000;
n = 4; % 螺旋桨叶片数
s = 4; % 螺旋桨的转速

%%%% 100Hz以下线谱频率可表示为如下 %%%%
Ts = 0:1/Fs:2.5; % 时间序列
ni = 8; % 八条线谱
Ak = 0.5 * rand(1, 10); % 线谱幅度
Ik = 0.25 * pi * rand(1, 10); % 随机相位

% 计算线谱频率
f1 = zeros(1, ni);
C = [];
for m1 = 1:ni
    f1(m1) = m1 * n * s;
end
f = f1;

%%% K为线谱数，Ak, f, Ik分别为第k条线谱对应的幅度、频率、相位
%% 求取功率谱 %%
fi = 80;
L = 4; % 线谱最高频率不能超过100Hz, 最高为ni*fi/L
fait = 0.25 * pi;
temp = 1:1:ni;
G = 0;
for i = 1:ni
    G1 = Ak(i) * sin(temp(i) * (2 * pi * fi / L * Ts + fait) + Ik(i));
    G = G + G1;
end

G = G + 7.5 * sin(2 * pi * 350 * Ts) + 6.5 * sin(2 * pi * 700 * Ts);

% 使用subplot将时域波形和功率谱放在同一窗口中
figure;

% 绘制时域波形
subplot(2, 1, 1); % 2行1列，选择第1个位置
plot(Ts, G);
ylim([-20,20])
xlabel('Time/s');
ylabel('Amplitude/V');
title('Line Spectrum Time-Domain Waveform');

% 功率谱分析
window = boxcar(length(G)); % 矩形窗
nfft = 3000; 
[Pxxm, f] = periodogram(G, window, nfft, Fs); % 直接法
PxxM = 10 * log10(Pxxm);

% 绘制功率谱
subplot(2, 1, 2); % 2行1列，选择第2个位置
plot(f, PxxM);
xlabel('Frequency/Hz');
ylabel('Power Spectrum/dB'); % 画出3000前的图
title('Line Specturm');
xlim([0, 2500]); % 限制频率范围为0-1000 Hz
