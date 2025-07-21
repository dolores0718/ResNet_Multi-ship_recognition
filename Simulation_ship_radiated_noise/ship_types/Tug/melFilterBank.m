function mel_filters = melFilterBank(Fs, n_fft, num_bands)
    % 生成 Mel 滤波器组
    % Fs: 采样率
    % n_fft: FFT 点数
    % num_bands: 滤波器组的数量
    
    % 计算 Mel 尺度的频率范围
    f_min = 0; % 最低频率
    f_max = Fs / 2; % Nyquist 频率 (最高频率)
    
    % 将频率转换为 Mel 频率
    mel_min = freqToMel(f_min);
    mel_max = freqToMel(f_max);
    
    % 等间隔划分 Mel 频率
    mel_points = linspace(mel_min, mel_max, num_bands + 2);
    
    % 将 Mel 频率点转换回 Hz 频率
    hz_points = melToFreq(mel_points);
    
    % 将 Hz 频率点映射到 FFT 的 bin 索引
    bin_points = floor((n_fft + 1) * hz_points / Fs);
    
    % 初始化滤波器组
    mel_filters = zeros(num_bands, n_fft / 2 + 1);
    
    % 创建三角形滤波器
    for m = 2:(num_bands + 1)
        % 左、中、右边界
        f_left = bin_points(m - 1);
        f_center = bin_points(m);
        f_right = bin_points(m + 1);
        
        % 斜率向上部分
        for k = f_left:f_center
            mel_filters(m - 1, k + 1) = (k - f_left) / (f_center - f_left);
        end
        
        % 斜率向下部分
        for k = f_center:f_right
            mel_filters(m - 1, k + 1) = (f_right - k) / (f_right - f_center);
        end
    end
end

% 辅助函数：Hz 转换为 Mel 频率
function mel = freqToMel(freq)
    mel = 2595 * log10(1 + freq / 700);
end

% 辅助函数：Mel 频率转换为 Hz
function freq = melToFreq(mel)
    freq = 700 * (10 .^ (mel / 2595) - 1);
end
