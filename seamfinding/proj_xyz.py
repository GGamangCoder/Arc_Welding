import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D 데이터 생성
points = np.loadtxt("./data/8_single_bevel.txt")

# 3D 플로팅 준비
fig = plt.figure(figsize=(12, 10))

# XY 평면에 사영 (Z 값이 0)
ax1 = fig.add_subplot(221)
ax1.scatter(points[:, 0], points[:, 1], c='r', marker='o')
ax1.set_title('Projection onto XY plane')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_aspect('equal')

# YZ 평면에 사영 (X 값이 0)
ax2 = fig.add_subplot(222)
ax2.scatter(points[:, 1], points[:, 2], c='g', marker='o')
ax2.set_title('Projection onto YZ plane')
ax2.set_xlabel('Y')
ax2.set_ylabel('Z')
ax2.set_aspect('equal')

# ZX 평면에 사영 (Y 값이 0)
ax3 = fig.add_subplot(223)
ax3.scatter(points[:, 0], points[:, 2], c='b', marker='o')
ax3.set_title('Projection onto ZX plane')
ax3.set_xlabel('X')
ax3.set_ylabel('Z')
ax3.set_aspect('equal')

# 3D 데이터 자체 시각화 (전체 데이터 확인용)
ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(points[:, 0], points[:, 1], points[:, 2], c='orange', marker='o')
ax4.set_title('Original 3D Data')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')

plt.tight_layout()
plt.show()
