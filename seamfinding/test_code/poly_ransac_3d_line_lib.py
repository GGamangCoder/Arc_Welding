import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from get_data_points import read_line


file_path = './data/1_fillet_gap.txt'  # 파일 경로 설정
points = read_line(file_path)
x, y = points[:, 0], points[:, 1]

# ---------------------------------------------------------------------
degree = 2
poly_model = make_pipeline(PolynomialFeatures(degree), RANSACRegressor())

xy = points[:, :2]
z = points[:, 2]

poly_model.fit(xy, z)

# 예측 곡선 생성
x_range = np.linspace(np.min(x), np.max(x), 100)
y_range = np.linspace(np.min(y), np.max(y), 100)
X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
Z_pred = poly_model.predict(np.c_[X_mesh.ravel(), Y_mesh.ravel()]).reshape(X_mesh.shape)

# 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, color='blue', s=5)

# RANSAC을 통한 다항식 곡선 시각화
ax.plot_surface(X_mesh, Y_mesh, Z_pred, color='orange', alpha=0.6)


ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

