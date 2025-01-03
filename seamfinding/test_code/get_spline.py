# 3차원 공간에서 스플라인 피팅
# 점들 사이를 적당히 이어 곡선처럼 만드는 방법

from scipy.interpolate import splprep, splev
import numpy as np
import matplotlib.pyplot as plt

from get_data_points import read_line


file_path = './data/5_butt_wide_2.txt'  # 파일 경로 설정
points = read_line(file_path)

x, y, z = points[:, 0], points[:, 1], points[:, 2]

# t 매개변수 정의 (각 점의 순서)
t = np.linspace(0, 1, len(x))

# 3D 스플라인 피팅
tck, u = splprep([x, y, z], s=2)  # 스플라인 파라미터 추정
x_fit, y_fit, z_fit = splev(u, tck)  # 스플라인 평가

# 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 원본 데이터
ax.scatter(x, y, z, label='Original Data', color='blue')

# 3D 곡선
ax.plot(x_fit, y_fit, z_fit, label='Fitted Curve', color='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()
