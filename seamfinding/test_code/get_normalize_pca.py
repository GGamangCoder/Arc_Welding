import numpy as np
from sklearn.decomposition import PCA
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 데이터 로드
points = np.loadtxt("./data/6_butt_narrow.txt")  # 파일 경로에 맞게 설정

data = points[10:-10]
x, y, z = data[:, 0], data[:, 1], data[:, 2]
points = np.column_stack((x, y, z))

# 2. 데이터 정규화 (Z-score 정규화)
mean = points.mean(axis=0)
std = points.std(axis=0)
normalized_points = (points - mean) / std

# 3. PCA로 평면 추정
pca = PCA(n_components=2)
points_2d = pca.fit_transform(normalized_points)  # 3D -> 2D 투영
restored_points = pca.inverse_transform(points_2d)  # 2D -> 3D 복원

# PCA 구성 요소로부터 법선 벡터 계산
vec1, vec2 = pca.components_  # 두 개의 주성분 벡터
plane_normal = np.cross(vec1, vec2)  # 외적을 이용한 법선 벡터 계산
plane_normal /= np.linalg.norm(plane_normal)  # 단위 벡터로 정규화

# 데이터의 중심 좌표
plane_center = np.mean(normalized_points, axis=0)

# 4. 평면 방정식: ax + by + cz + d = 0
a, b, c = plane_normal
d = -np.dot(plane_normal, plane_center)

# 5. 복원된 데이터 역정규화
restored_points_original_scale = restored_points * std + mean

def fit_polynomial_and_find_significant_points(projected_points, degree=3, threshold=0.05):
    # x와 y 분리
    x, y = projected_points[:, 0], projected_points[:, 1]
    
    # 다항식 추정
    poly_fit = Polynomial.fit(x, y, degree)
    y_fit = poly_fit(x)  # 추정된 y 값

    # 1차 미분을 이용한 기울기 계산
    first_derivative = poly_fit.deriv(1)
    # 2차 미분을 이용한 기울기 변화 계산
    second_derivative = poly_fit.deriv(2)

    slopes = first_derivative(x)
    slope_changes = np.abs(second_derivative(x))  # 기울기 변화량

    # 기울기가 0인 점 찾기 (기울기가 0이면 최저점 후보)
    minima_candidate_indices = np.where(np.abs(slopes) < threshold)[0]

    # 2차 미분이 양수인 점 찾기 (최저점 기준)
    minima_indices = [idx for idx in minima_candidate_indices if second_derivative(x[idx]) > 0]

    return poly_fit, minima_indices


# points_2d 데이터 (사영된 2D 데이터) 사용
poly_fit, minimum_points_indices = fit_polynomial_and_find_significant_points(points_2d, degree=4, threshold=0.02)
print(poly_fit, minimum_points_indices)

# 시각화
x, y = points_2d[:, 0], points_2d[:, 1]
# x, y = vec1, vec2
x_sorted = np.sort(x)
y_fit_sorted = poly_fit(x_sorted)





''' ---------------------------- 시각화 '''
plt.figure()
plt.scatter(x, y, label="Projected Points", alpha=0.5)
plt.plot(x_sorted, y_fit_sorted, label=f"Fitted Curve (Degree={poly_fit.degree()})", color="g")
plt.scatter(x[minimum_points_indices], y[minimum_points_indices], color="r", label="Minimum Points", s=50)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Curve Fitting and Minimum Points Detection (2D Data)")
plt.show()

