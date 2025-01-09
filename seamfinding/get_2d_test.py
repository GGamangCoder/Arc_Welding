'''
1. PCS로 평면을 추정하여 점들을 사영
2. polynomial RANSAC 돌려서 다차 곡선 추정
3. 최저점을 파악
4. 해당 점 근처에서 직선 RANSAC
5. 결과 시각화
'''

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

from get_line_test import fit_line_ransac



# Step 1: PCA for plane estimation
def estimate_plane_pca(points):
    pca = PCA(n_components=2)       # n_components: 주축 개수
    pca.fit(points)
    # Project data onto the estimated plane
    projected_points = pca.transform(points)
    # Recover the plane's origin and axes for visualization
    origin = pca.mean_
    normal_vector = pca.components_[1]

    return projected_points, pca, origin, normal_vector

# Step 2: Polynomial fitting with RANSAC
def fit_polynomial_ransac(projected_points, degree=4):
    x, y = projected_points[:, 0], projected_points[:, 1]
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(x.reshape(-1, 1))

    # Fit using RANSAC
    ransac = RANSACRegressor()
    ransac.fit(X_poly, y)
    y_fit = ransac.predict(X_poly)

    return ransac, poly_features, x, y, y_fit

# Step 3: Detect global minima
def find_global_minima(poly_coefficients, x_range, plane_point, plane_normal):
    poly = Polynomial(poly_coefficients)
    x_values = np.linspace(min(x_range), max(x_range), 1000)
    y_values = poly(x_values)
    min_index = np.argmin(y_values)

    return x_values[min_index], y_values[min_index]

    # min_x, min_y = x_values[min_index], y_values[min_index]

    # # 2D 좌표를 3D로 복원
    # plane_point_2d = np.array([min_x, min_y, 0])  # 평면 좌표계의 z = 0으로 초기화
    # projection_matrix = np.identity(3) - np.outer(plane_normal, plane_normal)  # 평면 투영 행렬
    # global_min_3d = plane_point + projection_matrix @ (plane_point_2d - plane_point)

    # return global_min_3d

# Step 3-1: minima와 가장 가까운 원래 3차원 점들 중 인덱스
def find_closest_point_index(goal_point, points):
    points = np.array(points)
    goal_point = np.array(goal_point)

    distances = np.linalg.norm(points - goal_point, axis=1)

    closest_index = np.argmin(distances)
    
    return closest_index

# Step 4: RANSAC for line fitting near the minima
def fit_line_near_minima(points, minima_point, window=0.05):     # window: 최저점 근처 영역 range
    # Filter points near the minima
    filtered_points = points[
        (points[:, 0] > minima_point[0] - window) &
        (points[:, 0] < minima_point[0] + window)
    ]
    if len(filtered_points) < 2:
        raise ValueError("Insufficient points near minima for line fitting.")
    
    x, y = filtered_points[:, 0], filtered_points[:, 1]
    ransac = RANSACRegressor()
    ransac.fit(x.reshape(-1, 1), y)

    return ransac

# Visualization utility
def plot_3d(points, projected_points, poly_ransac, poly_x, poly_y, minima, plane_origin, plane_normal):
    fig = plt.figure(figsize=(12, 6))
    
    # Original 3D points
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], label='Original Points', s=1)
    ax.quiver(plane_origin[0], plane_origin[1], plane_origin[2],
              plane_normal[0], plane_normal[1], plane_normal[2],
              length=5, color='red', label='Plane Normal')
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
    projected_points, pca, origin, normal = estimate_plane_pca(points)
    ransac, poly_features, x, y, y_fit = fit_polynomial_ransac(projected_points, degree=4)
    poly_coefficients = ransac.estimator_.coef_
    minima = find_global_minima(poly_coefficients, x, origin, normal)
    try:
        line_model = fit_line_near_minima(projected_points, minima)
        print(f"Line equation near minima: y = {line_model.estimator_.coef_[0]}x + {line_model.estimator_.intercept_}")

        # min_idx = find_closest_point_index(minima, points)
        # window = 30
        # start_points = points[min_idx-window:min_idx-1, :]
        # end_points = points[min_idx+1:min_idx+window, :]
        # start_line, start_line_inliers = fit_line_ransac(start_points)
        # end_line, end_line_inliers = fit_line_ransac(end_points)

    except ValueError as e:
        print(str(e))
    
    plot_3d(points, projected_points, ransac, x, y_fit, minima, origin, normal)

# Example usage
if __name__ == "__main__":
    points = np.loadtxt("./data/3_circle_hole.txt")  # 파일 경로에 맞게 설정
    # x, y, z = points[:, 0], points[:, 1], points[:, 2]

    process_3d_data(points)
