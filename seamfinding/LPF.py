import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# 예시 데이터 (노이즈가 포함된 1D 신호)
fs = 1000  # 샘플링 주파수 (1000Hz)
t = np.linspace(0, 1, fs)  # 1초 동안의 시간 벡터
x = np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.5, fs)  # 50Hz 신호 + 노이즈

# Butterworth 로우 패스 필터 설계
cutoff_freq = 100  # 차단 주파수 (100Hz)
order = 4  # 필터 차수
b, a = signal.butter(order, cutoff_freq / (0.5 * fs), btype='low')

# 필터 적용
filtered_data = signal.filtfilt(b, a, x)

# 원본 데이터와 필터링된 데이터 비교
plt.figure(figsize=(10, 6))
plt.plot(t, x, label='Original Data')
plt.plot(t, filtered_data, label='Filtered Data', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Low-Pass Filtering with Butterworth Filter')
plt.show()
