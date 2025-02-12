import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D 데이터 생성
weld_type = "2_fillet_gap_2"
points = np.loadtxt(f"./data/{weld_type}.txt")

# 3D 플로팅 준비
fig = plt.figure()
fig.suptitle(f'Type: {weld_type}', fontsize=16)

ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
