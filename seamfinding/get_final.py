'''
1. PCS로 평면을 추정하여 점들을 사영
2. polynomial RANSAC 돌려서 다차 곡선 추정
3. 최저점을 파악
    ㄴ 2가지 방법(도함수 비교 / 라이브러리 이용)
4. 해당 점 근처에서 직선 RANSAC
5. 결과 시각화
'''

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

from get_line_test import *
from line_to_line import closestDistanceBetweenLines


# Step 1: PCA for plane estimation
def estimate_plane_pca(points):
    pca = PCA(n_components=2)       # n_components: 주축 개수
    pca.fit(points)
    # Project data onto the estimated plane
    projected_points = pca.transform(points)
    # Recover the plane's origin and axes for visualization
    origin = pca.mean_                      # 평균, Per-feature empirical mean, estimated from the training set.
    normal_vector = pca.components_[0]      # 주축 벡터, Principal axes in feature space, representing the directions of maximum variance in the data.

    return projected_points, pca, origin, pca.components_[0], pca.components_[1]

# Step 2: Polynomial fitting with RANSAC
def fit_polynomial_ransac(projected_points, degree=4):
    x, y = projected_points[:, 0], projected_points[:, 1]
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(x.reshape(-1, 1))      # 변형된 배열

    # Fit using RANSAC
    ransac = RANSACRegressor()
    ransac.fit(X_poly, y)
    y_fit = ransac.predict(X_poly)

    return ransac, x, y, y_fit

# Step 3-1: Detect global minima(2차 도함수가 0)
# 4차 이상이라고 생각하고 1차 미분값이 0이면서 - -> + 바뀌는 구간 중 최솟값
def find_global_minima(poly_coefficients, x_range):
    poly = Polynomial(poly_coefficients)
    
    # 1차 도함수 계산
    first_derivative = poly.deriv(1)

    # 1차 도함수의 근 찾기
    critical_points = first_derivative.roots()

    # x 범위 내의 값 필터링
    valid_points = [x for x in critical_points if min(x_range) <= x <= max(x_range)]
    # print(f"valid_points: {valid_points}")

    # 최솟값 판별: 1차 미분 값 변화 확인
    minima_candidates = []
    for x in valid_points:
        slope_left = first_derivative(x - 1e-6)     # x보다 조금 작은 값
        slope_right = first_derivative(x + 1e-6)    # x보다 조금 큰 값
        # print(f"x, left, right: {x} {slope_left} {slope_right}")

        if slope_left < 0 and slope_right > 0:  # -에서 +로 변화
            minima_candidates.append(x)
    # print(f"minima_candidates: {minima_candidates}")

    # 최솟값 찾기
    if minima_candidates:
        y_values = [poly(x) for x in minima_candidates]
        min_index = np.argmin(y_values)
        min_x, min_y = minima_candidates[min_index], y_values[min_index]
        return min_x, min_y
    else:
        return None, None  # 유효한 최솟값이 없는 경우

# Step 3-2: 보간 라이브러리 이용
from scipy.optimize import minimize_scalar

def find_minimum_with_scipy(poly_coefficients, x_range):
    """
    Scipy를 사용한 최저점 찾기.

    Parameters:
        poly_coefficients: 다항식 계수 리스트.
        x_range: (min_x, max_x)의 범위.

    Returns:
        (min_x, min_y): 최저점 좌표.
    """
    poly = np.poly1d(poly_coefficients)

    result = minimize_scalar(poly, bounds=x_range, method='bounded')
    if result.success:
        min_x = result.x        # 최저가 되는 x 값
        min_y = result.fun      # 그때의 y 값
        return min_x, min_y
    else:
        raise ValueError("최저점 찾기에 실패했습니다.")

# Step 4:찾은 결과와 가장 가까운 데이터(인덱스) 구하기
def find_closest_index(min_x, x_range):
    # min_x와 x_range 값들의 차이를 계산하고, 가장 작은 차이를 가진 인덱스를 찾기
    closest_index = np.argmin(np.abs(x_range - min_x))
    return closest_index

# Step 4-1: RANSAC for line fitting near the minima
# 옆에 함수 가져오기 -- test_final fit_line_ransac

# Step 5: Visualization utility
def plot_3d(points, projected_points, poly_ransac, poly_x, poly_y, minima, plane_origin, plane_normal_1, plane_normal_2, line_1, line_2, inv_points=None):
    fig = plt.figure(figsize=(12, 6))
    
    # Original 3D points
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], label='Original Points', s=1)
    if inv_points is not None:
        ax.scatter(inv_points[0], inv_points[1], inv_points[2], label='minima points', s=20)
    ax.quiver(plane_origin[0], plane_origin[1], plane_origin[2],
              plane_normal_1[0], plane_normal_1[1], plane_normal_1[2],
              length=5, color='red', label='Main Normal')
    ax.quiver(plane_origin[0], plane_origin[1], plane_origin[2],
              plane_normal_2[0], plane_normal_2[1], plane_normal_2[2],
              length=5, color='darkred', label='Sub Normal')
    #########################################################
    # 직선 그리기 - 그래프 길이는 조절 가능 & 교점 찾기
    # p1, direction_1 = line_1
    # line_1_points = np.array([p1 + t * direction_1 for t in np.linspace(-20, 20, 10)])

    # ax.plot(line_1_points[:, 0], line_1_points[:, 1], line_1_points[:, 2], color='cyan', label='fitting line_1')

    # p2, direction_2 = line_2
    # line_2_points = np.array([p2 + t * direction_2 for t in np.linspace(-20, 20, 10)])

    # ax.plot(line_2_points[:, 0], line_2_points[:, 1], line_2_points[:, 2], color='navy', label='fitting line_2')

    # intersection_1, intersection_2, dist_1_to_2 = closestDistanceBetweenLines(p1, p1+direction_1, p2, p2+direction_2)
    # ax.scatter(intersection_1[0], intersection_1[1], intersection_1[2], color='red', s=20)
    # ax.scatter(intersection_2[0], intersection_2[1], intersection_2[2], color='darkred', s=50)
    #########################################################

    ax.set_title("3D Points and PCA Plane")
    ax.legend()
    
    # 2D projected points and polynomial
    ax2 = fig.add_subplot(122)
    ax2.scatter(projected_points[:, 0], projected_points[:, 1], label='Projected Points', s=1)
    ax2.plot(poly_x, poly_y, color='orange', label='Fitted Polynomial')
    ax2.scatter(minima[0], minima[1], color='red', label='Minima', s=20)
    ax2.set_title("2D Projected Points and Fitted Curve")
    ax2.legend()
    
    plt.show()

# Main pipeline
def process_3d_data(points):
    projected_points, pca, origin, normal_1, normal_2 = estimate_plane_pca(points)
    # print(f"main normal: {normal_1}")
    # print(f"sub normal: {normal_2}")

    n_degree = 5
    # ransac 객체, poly_features 항들/계수, x/y: proj_points(pca 새로운 축), y_fit: 적용
    ransac, x, y, y_fit = fit_polynomial_ransac(projected_points, degree=n_degree)

    poly_coefficients = ransac.estimator_.coef_         # 높은 차수 부터 낮은 차수 순으로
    poly_intercept = ransac.estimator_.intercept_       # 가장 마지막 항, 즉 y 절편
    # print(f'계수: {poly_coefficients} / 상수: {poly_intercept}')

    poly_coefficients[0] = poly_intercept
    print(f"{n_degree}차 방정식 계수: {poly_coefficients}")
    minima = find_global_minima(poly_coefficients, x)

    # poly_coefficients = poly_coefficients[::-1]     # 계수 역순 정렬, 일반적인 n차 곡선 방정식
    # poly_coefficients[-1] = poly_intercept
    # print(f"{n_degree}차 방정식 계수: {poly_coefficients}")
    # minima = find_minimum_with_scipy(poly_coefficients, (x[0], x[-1]))

    # inv_points = pca.inverse_transform(minima)          # pca -> 원 좌표계 복구
    # print(f"역변환 좌표: {inv_points}")

    min_idx = find_closest_index(minima[0], x)
    inv_points = points[min_idx]
    print(f"역변환 좌표: {inv_points}")

    # 직선 회귀
    data_num = 30
    start_points = points[min_idx-data_num:min_idx, :]
    end_points = points[min_idx:min_idx+data_num, :]
    
    start_line, start_line_inliers = fit_line_ransac(start_points)
    end_line, end_line_inliers = fit_line_ransac(end_points)

    # print("Best start_line model (p1, dir_1)/inliers:", start_line, start_line_inliers)
    # print("Best end_line model (p2, dir_2)/inliers:", end_line, end_line_inliers)

    plot_3d(points, projected_points, ransac, x, y_fit, minima, origin, normal_1, normal_2, start_line, end_line, inv_points)

# Example usage
if __name__ == "__main__":
    points = np.loadtxt("./data/8_single_bevel.txt")
    # x, y, z = points[:, 0], points[:, 1], points[:, 2]

    process_3d_data(points)
