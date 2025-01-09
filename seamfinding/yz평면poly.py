import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import RANSACRegressor

from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial.polynomial import Polynomial

import timeit


def fit_polynomial_ransac(axis1, axis2, degree=4):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(axis1.reshape(-1, 1))

    # Fit using RANSAC
    ransac = RANSACRegressor()
    ransac.fit(X_poly, axis2)
    z_fit = ransac.predict(X_poly)

    return ransac, z_fit

def plot_3d(points, Y, Z, z_fit, minima, inv_points):
    fig = plt.figure(figsize=(12, 6))
    
    # Original 3D points
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], label='Original Points', s=1)
    ax.scatter(inv_points[0], inv_points[1], inv_points[2], label='minima points', color='red', s=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Original Points")
    ax.legend()
    
    # 2D projected points and polynomial
    ax2 = fig.add_subplot(122)
    ax2.scatter(Y, Z, label='Projected Points', s=1)
    ax2.plot(Y, z_fit, color='orange', label='Fitted Polynomial')
    ax2.scatter(minima[0], minima[1], color='red', label='Minima', s=20)
    ax2.set_title("2D Projected Points and Fitted Curve")
    ax2.legend()
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Z')
    plt.show()

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
        return None, None

def find_closest_index(min_x, x_range):
    # min_x와 x_range 값들의 차이를 계산하고, 가장 작은 차이를 가진 인덱스를 찾기
    closest_index = np.argmin(np.abs(x_range - min_x))
    return closest_index

def process_3d_data():
    points = np.loadtxt("./data/8_single_bevel.txt")

    Y, Z = points[:, 1], points[:, 2]

    n_degree = 4
    ransac, z_fit = fit_polynomial_ransac(Y, Z, degree=n_degree)

    poly_coefficients = ransac.estimator_.coef_         # 높은 차수 부터 낮은 차수 순으로
    poly_intercept = ransac.estimator_.intercept_       # 가장 마지막 항, 즉 y 절편
    poly_coefficients[0] = poly_intercept
    print(f"{n_degree}차 방정식 계수: {poly_coefficients}")

    minima = find_global_minima(poly_coefficients, Y)       # y, z 결과 == (min_y, min_z)
    print(minima)
    min_idx = find_closest_index(minima[0], Y)

    inv_points = points[min_idx]
    print('inv_points: ', inv_points)

    plot_3d(points, Y, Z, z_fit, minima, inv_points)


if __name__ == "__main__":
    # 3D 데이터 생성

    # 실행 시간 계산 평균
    # excutime_time = timeit.timeit('process_3d_data()', globals=globals(), number=5)
    # excutime_time = timeit.timeit(process_3d_data, number=5)
    # print(f"Execution time: {excutime_time:.4f} seconds")

    process_3d_data()
