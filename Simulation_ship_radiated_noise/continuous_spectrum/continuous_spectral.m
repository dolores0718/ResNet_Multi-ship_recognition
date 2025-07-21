%% 

%%%%%理想舰船辐射噪声模型%%%%%
V=input('请输入舰船的航速：');      % 舰船的实际航速
V_max=input('请输入最高航速：');    % 舰船的最高航速，单位/节
DT=input('请输入排水量：');         % 舰船的排水量,单位/吨

SLs=122+50*log(V/10)+15*log(DT);   % 100Hz以上声级的经验公式,单位/dB
% f0=1000-900*V/V_max;             % 谱峰频率,单位/Hz
if V >= 10
    f0 = 300 - 200 * (V - 10) / 30;  % For V >= 10 knots
else
    f0 = 300;  % For V < 10 knots
end
SLf0=SLs+20-20*log(f0);            % 谱峰处声压谱级


K1=1;                            % 低于谱峰频率上升斜率
K2=-20;                            % 高于谱峰频率下降的斜率

%%%%对连续谱进行建模%%%%%
ff=0:1:10000;
SLf=(SLf0-K1*log(f0./ff)).*(ff<f0)+SLf0.*(ff==f0)+(SLf0+K2*log(ff./f0)).*(f0<ff);

figure(5)
plot(ff,SLf,'k');
xlabel('Frequency/Hz');xlim([0,2500])
 ylabel('Spectral Level/dB');%ylim([160,200])
 title('Continuous Spectral')
% grid on;

