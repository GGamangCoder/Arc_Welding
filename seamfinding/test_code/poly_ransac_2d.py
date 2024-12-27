#########################################
# 2d 평면 상에 다각형 ransac 을 적용한 함수
# ㄴ  현재 사용하지 않음.
#########################################


import numpy as np
import matplotlib.pyplot as plt
from get_data_points import read_line


# 다항식 피팅 함수
def fit_polynomial(x, y, degree):
    return np.polyfit(x, y, degree)

# 잔차 계산
def calculate_residuals(y_true, y_pred):
    return np.abs(y_true - y_pred)

# Polynomial RANSAC
def ransac_polynomial(x, y, degree, n_iterations=1000, threshold=1.0):
    best_inliers = []
    best_poly = None

    for _ in range(n_iterations):
        sample_indices = np.random.choice(len(x), degree + 1, replace=False)
        x_sample, y_sample = x[sample_indices], y[sample_indices]

        poly = np.polyfit(x_sample, y_sample, degree)
        y_pred = np.polyval(poly, x)

        residuals = calculate_residuals(y, y_pred)
        inliers = np.where(residuals < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_poly = poly

    return best_poly, best_inliers

# 데이터 생성 및 적용
# np.random.seed(0)
# x = np.linspace(-3, 3, 100)
# y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, x.shape)
# y[::10] += 10 * np.random.normal(size=10)  # 이상치

file_path = './data/proj_data.txt'  # 파일 경로 설정
points = read_line(file_path)
x, y, z = points[:, 0], points[:, 1], points[:, 2]
deg = 2

best_poly, inliers = ransac_polynomial(x, y, degree=deg)
y_ransac = np.polyval(best_poly, x)

plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, y_ransac, color='red', label='Polynomial RANSAC Fit')
plt.legend()
plt.show()
