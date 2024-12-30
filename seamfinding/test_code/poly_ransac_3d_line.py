import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from get_data_points import read_line


# 3D Polynomial RANSAC 함수
def ransac_polynomial_curve_3d(x, y, z, degree, n_iterations=1000, threshold=0.1):
    best_inliers = []
    best_model = None

    for _ in range(n_iterations):
        # 무작위로 두 점 선택
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


# 3D 데이터 생성
# np.random.seed(0)
# num_points = 100
# x = np.random.uniform(-3, 3, num_points)
# y = np.random.uniform(-3, 3, num_points)
# z = 0.5 * x**2 + 0.3 * y**2 + 2 + np.random.normal(0, 0.2, num_points)  # 대략적인 곡선


file_path = './data/1_fillet_gap.txt'  # 파일 경로 설정
points = read_line(file_path)
x, y, z = points[:, 0], points[:, 1], points[:, 2]
deg = 2

# Polynomial RANSAC 적용
best_model, inliers = ransac_polynomial_curve_3d(x, y, z, degree=deg)

# 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, color='blue', label='Data')
ax.scatter(x[inliers], y[inliers], z[inliers], color='red', label='Inliers')

# 곡선 그리기
x_fit = np.linspace(min(x), max(x), 100)
y_fit = np.linspace(min(y), max(y), 100)
X_grid = np.array(np.meshgrid(x_fit, y_fit)).T.reshape(-1, 2)
Z_fit = best_model.predict(PolynomialFeatures(degree=deg).fit_transform(X_grid)).reshape(len(x_fit), len(y_fit))

ax.plot_surface(x_fit, y_fit, Z_fit, color='orange', alpha=0.5)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title(f"Polynomial RANSAC Fitting/line - {deg}'")

ax.legend()
plt.show()
