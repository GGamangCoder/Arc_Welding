from test_final import read_line
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


file_path = './data/LTS_gather.txt'  # 파일 경로 설정
points = read_line(file_path)

X = points[:, 0]
Y = points[:, 1]
Z = points[:, 2]

T_matrix = np.c_[X, Y, np.ones_like(X)]
d_vec = -Z

params = np.linalg.lstsq(T_matrix, d_vec, rcond=None)[0]

A, B, D = params
C = 1

print(f"norm(A, B, C): ({A}, {B}, {C}) , d: {D} ")


# 6. 3D 시각화 준비
# 3D 그래프 설정
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 7. 점들을 3D 공간에 시각화
scatter = ax.scatter(X, Y, Z, color='b', label='Data Points')

# 8. 평면을 그리기 위해 X, Y 값의 범위 생성
x_range = np.linspace(min(X), max(X), 10)
y_range = np.linspace(min(Y), max(Y), 10)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# 9. 평면 방정식에 따라 Z 값을 계산
Z_grid = -(A * X_grid + B * Y_grid + D) / C

# 10. 평면 그리기
ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.5, color='r', label='Fitted Plane')

# 11. 레이블 추가
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 12. 제목 추가
ax.set_title('3D Points and Fitted Plane using LSM')

# 13. 그래프 보이기
# 축 범위 설정
plane_legend = Line2D([0], [0], color='r', lw=4, label='Fitted Plane')
ax.legend(handles=[scatter, plane_legend])

plt.show()