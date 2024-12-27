##########################################
# 3차원에서 모든 점을 포함하는 평면 찾기
# ㄴ  현재 사용하지 않음.
#########################################

from get_data_points import read_line
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

# 3차원 Polynomial RANSAC 함수
def ransac_polynomial_3d(x, y, z, degree, n_iterations=1000, threshold=1.0):
    best_inliers = []
    best_model = None

    for _ in range(n_iterations):
        sample_indices = np.random.choice(len(x), degree + 1, replace=False)
        x_sample, y_sample, z_sample = x[sample_indices], y[sample_indices], z[sample_indices]
        
        # 다항식 특성 생성
        poly_features = PolynomialFeatures(degree=degree)
        X_sample = poly_features.fit_transform(np.vstack((x_sample, y_sample)).T)

        # 회귀 모델 피팅
        model = LinearRegression()
        model.fit(X_sample, z_sample)
        
        # 전체 데이터에 대한 예측
        X = poly_features.transform(np.vstack((x, y)).T)
        z_pred = model.predict(X)
        
        residuals = np.abs(z - z_pred)
        inliers = np.where(residuals < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = model

    return best_model, best_inliers

# 데이터 생성 및 적용
# np.random.seed(0)
# x = np.random.uniform(-3, 3, 100)
# y = np.random.uniform(-3, 3, 100)
# z = 0.5 * x**2 + 0.3 * y**2 + x + y + 2 + np.random.normal(0, 1, x.shape)
# z[::10] += 10 * np.random.normal(size=10)  # 이상치 추가


file_path = './data/LTS_gather.txt'  # 파일 경로 설정
points = read_line(file_path)
x, y, z = points[:, 0], points[:, 1], points[:, 2]
deg = 5


# 3차 다항식 피팅
best_model, inliers = ransac_polynomial_3d(x, y, z, degree=deg)

# 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, color='blue', label='Data')
ax.scatter(x[inliers], y[inliers], z[inliers], color='red', label='Inliers')

# 평면 그리기
x_grid, y_grid = np.meshgrid(np.linspace(np.min(x), np.max(x), 10), np.linspace(np.min(y), np.max(y), 10))
X_grid = np.vstack((x_grid.ravel(), y_grid.ravel())).T
Z_grid = best_model.predict(PolynomialFeatures(degree=deg).fit_transform(X_grid)).reshape(x_grid.shape)

ax.plot_surface(x_grid, y_grid, Z_grid, alpha=0.5, color='orange')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title(f"Polynomial RANSAC Fitting - {deg}'")

ax.legend()
plt.show()
