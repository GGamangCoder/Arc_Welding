'''
0. 3차원 점들을 원점으로 이동
1. 방향성 없는 점들의 좌표계 설정 - 이동방향 기준 X 축 혹은 Y축(로드리게즈 회전 변환)
2. 평면 추정하여 n차 곡선 회귀 추정 - RANSAC
3. 1차 도함수를 이용해 최저점 찾기
4. 3번에서 찾은 점을 기준으로 좌우 직선 RANSAC
5. 최종 seam finding 
6. 해당 결과 시각화
'''

import numpy as np

import copy
import random

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

from line_to_line import closestDistanceBetweenLines




# Step1: 좌표계 설정 및 회전 변환
def rotation_formula(data, dir="X"):
    """
    3D 점들의 좌표계를 설정하고, 주어진 방향으로 회전 변환을 수행한다.
    이동 방향을 기준으로 로드리게즈 회전 변환 공식을 이용한다.

    params:
        data (ndarray): 3D 데이터 배열. (x, y, z) 좌표를 의미.
        dir (str): 데이터들을 정렬할 방향(기본값: "X")

    return:
        transformed_data (ndarray): 회전 변환해서 얻은 데이터
    """
    P_start = data[0]
    P_end = data[-1]
    vector = P_end - P_start

    unit_vector = vector / np.linalg.norm(vector)    # 단위 벡터 변환

    # 로봇 이동 방향(데이터 수집)을 회전축으로 사용
    if dir == "X":
        target_vector = np.array([1, 0, 0])
    elif dir == "Y":
        target_vector = np.array([0, 1, 0])

    # 회전 축 (벡터의 외적)
    rotation_axis = np.cross(unit_vector, target_vector)

    # 회전 각도 (두 벡터 간의 내적을 이용하여 계산)
    cos_theta = np.dot(unit_vector, target_vector)
    theta = np.arccos(cos_theta)

    # 회전 행렬
    K = np.array([
        [0                 , -rotation_axis[2] , rotation_axis[1]],
        [rotation_axis[2]  , 0                 , -rotation_axis[0]],
        [-rotation_axis[1] , rotation_axis[0]  , 0]
    ])

    # 로드리게즈 회전 변환 공식
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    # 데이터 변환
    transformed_data = np.dot(data, R.T)

    return transformed_data


# Step2: 다항식 회귀
def fit_polynomial_ransac(points, degree=4):
    """
    n차 다항식 회귀 추정

    params:
        points (ndarray): 회귀 추정할 점들의 2D 배열
        degree (int, optional): 다항식 차수(기본값: 4)

    return:
        ransac (RANSACRegressor): RANSAC 회귀 추정 객체
        x (ndarray): 원본 데이터의 x 좌표
        y (ndarray): 원본 데이터의 y 좌표
        y_fit (ndarray): 예측된 y 좌표
    """
    x, y = points[:, 0], points[:, 1]
    poly_features = PolynomialFeatures(degree=degree)    # 다항식 특징 생성(예: x^3, x^2 * y, ...)
    X_poly = poly_features.fit_transform(x.reshape(-1, 1))

    # RANSAC 회귀 모델 생성 및 학습
    ransac = RANSACRegressor()
    ransac.fit(X_poly, y)
    y_fit = ransac.predict(X_poly)

    return ransac, x, y, y_fit


# Step3: 특이점 찾기
def find_global_minima(poly_coefficients, x_range):
    """
    global minima 찾기
    1차 도함수 값이 0이면서 `(-)`에서 `(+)`로 바뀌는 점들 중 최솟값으로 한다.

    params:
        poly_coefficients (ndarray): RANSAC 회귀 모델의 추정된 회귀 계수
        x_range (list): 현재 데이터의 x값 구간

    return:
        min_x, min_y (float): 최저점의 x, y 좌표
    """
    poly = Polynomial(poly_coefficients)    # 다항식 생성
    first_derivative = poly.deriv(1)        # 1차 도함수 계산, 근 얻기
    critical_points = first_derivative.roots()    # 1차 도함수의 근 찾기

    # x 범위 내의 값 필터링
    valid_points = [x for x in critical_points if min(x_range) <= x <= max(x_range)]

    minima_candidates = []
    for x in valid_points:
        slope_left = first_derivative(x - 1e-6)     # x보다 조금 작은 값
        slope_right = first_derivative(x + 1e-6)    # x보다 조금 큰 값

        if slope_left < 0 and slope_right > 0:      # -에서 +로 변화
            minima_candidates.append(x)

    if minima_candidates:
        y_values = [poly(x) for x in minima_candidates]
        min_index = np.argmin(y_values)
        min_x, min_y = minima_candidates[min_index], y_values[min_index]
        return min_x, min_y
    else:
        # raise ValueError("유효한 최솟값을 찾을 수 없습니다. 주어진 x 범위 내에 최솟값이 없습니다.")
        return None, None


# Step 3-1: 찾은 결과와 가장 가까운 데이터(인덱스) 구하기
def find_closest_index(min_x, x_range):
    closest_index = np.argmin(np.abs(x_range - min_x))
    return closest_index

# Step 4: RANSAC for line fitting near the minima
def fit_line_ransac(points, n_iterations=100, threshold=0.1):
    """
    RANSAC으로 직선 회귀 추정

    params:
        points (ndarray): 추정할 점들
        n_iterations (int): 반복 횟수
        threshold (float): 임계값

    return:
        best_line (ndarray): (p1, dir), 한 점과 방향
        best_inliers_cnt (int): 인라이어 갯수
    """
    best_inliers_cnt = 0
    best_line = None

    for _ in range(n_iterations):
        sample_indices = random.sample(range(len(points)), 2)
        p1, p2  = points[sample_indices]

        # 직선의 방향 벡터
        direction = p2 - p1
        direction = direction / np.linalg.norm(direction)

        inliers_cnt = 0
        for point in points:
            cross_prod = np.linalg.norm(np.cross(point - p1, direction))
            dist_point2line = cross_prod / np.linalg.norm(direction)
            if dist_point2line < threshold:
                inliers_cnt += 1

        if inliers_cnt > best_inliers_cnt:
            best_inliers_cnt = inliers_cnt
            best_line = (p1, direction)
        
    return best_line, best_inliers_cnt

# Step 5: 최종 결과(Seam) 반환하는 부분


# Step 6: Visualization utility
def plot_3d(origin_points, rot_points, projected_points, poly_x, poly_y, line_1, line_2, minima, idx=None):

    fig = plt.figure(figsize=(12, 6))
    
    # Original 3D points
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(origin_points[:, 0], origin_points[:, 1], origin_points[:, 2], label='Origin points', color='g', s=5)
    ax.scatter(rot_points[:, 0], rot_points[:, 1], rot_points[:, 2], label='Formula Points', color='b', s=5)
    if idx == None:
        print("유효한 최솟값을 찾을 수 없습니다.")
    else:
        min_point = origin_points[idx]
        ax.scatter(min_point[0], min_point[1], min_point[2], label='minima points', color='r', s=20)
        print(f"min_idx: {idx}")
        print(f"3차원 위 좌표: {min_point}")

    #########################################################
    # 직선 그리기 - 그래프 길이는 조절 가능 & 교점 찾기
    p1, direction_1 = line_1
    line_1_points = np.array([p1 + t * direction_1 for t in np.linspace(-25, 25, 10)])          # 숫자는 그냥 스케일이라 시각화에만 관여

    ax.plot(line_1_points[:, 0], line_1_points[:, 1], line_1_points[:, 2], color='cyan', label='fitting line_1')

    p2, direction_2 = line_2
    line_2_points = np.array([p2 + t * direction_2 for t in np.linspace(-25, 25, 10)])

    ax.plot(line_2_points[:, 0], line_2_points[:, 1], line_2_points[:, 2], color='navy', label='fitting line_2')

    intersection_1, intersection_2, dist_1_to_2 = closestDistanceBetweenLines(p1, p1+direction_1, p2, p2+direction_2)
    ax.scatter(intersection_1[0], intersection_1[1], intersection_1[2], color='darkred', s=20)
    # ax.scatter(intersection_2[0], intersection_2[1], intersection_2[2], color='darkred', s=50)
    #########################################################

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title("3D Points and Rotation")
    ax.legend()
    
    # 2D projected points and polynomial
    ax2 = fig.add_subplot(122)
    ax2.scatter(projected_points[:, 0], projected_points[:, 1], label='Projected Points', s=1)
    ax2.plot(poly_x, poly_y, color='orange', label='Fitted Polynomial')
    ax2.scatter(minima[0], minima[1], label='Minima', color='r', s=20)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')

    ax2.set_title("2D Projected Points and Fitted Curve")
    ax2.legend()

    plt.show()

# Main pipeline
def process_3d_data(points, n_degree, dir, count, iterations, threshold):
    # 원래 포인트 복사
    origin_points = copy.deepcopy(points)

    # 원점 기준 변경
    first_point = copy.deepcopy(points[0])
    points -= first_point               # 이거 나중에 원복 필요.

    # 데이터 회전 변환(좌표계 설정)
    rotation_points = rotation_formula(points, dir)
    
    # X 축으로 정렬하면 xz 평면을, Y축일 경우에는 yz 평면을 본다.
    if dir == "X":
        proj_plane_points = rotation_points[:, [0, 2]]
    else:
        proj_plane_points = rotation_points[:, 1:3]

    # 여기서 X는 주축(x or y), Y는 z를 의미한다.
    ransac, X, Y, Y_fit = fit_polynomial_ransac(proj_plane_points, degree=n_degree)

    # RANSAC 회귀 추정 객체 - 다항식의 계수와 y 절편
    poly_coefficients = ransac.estimator_.coef_
    poly_intercept = ransac.estimator_.intercept_
    poly_coefficients[0] = poly_intercept       # y 절편까지 반영
    print(f"{n_degree}차 방정식 계수: {poly_coefficients}")

    # 회귀 곡선에서 최저점
    minima = find_global_minima(poly_coefficients, X)

    # 회귀 추정된 점과 가장 가까운 원 데이터의 index
    min_idx = find_closest_index(minima[0], X)

    # 직선 회귀 점들 추출: 기준점으로부터 양 쪽으로 count 만큼
    start_points = origin_points[min_idx - count : min_idx, :]
    end_points = origin_points[min_idx : min_idx + count, :]

    start_line, start_line_inliers = fit_line_ransac(start_points, iterations, threshold)
    end_line, end_line_inliers = fit_line_ransac(end_points, iterations, threshold)

    print(f"start line 인라이어(개): {start_line_inliers}")
    print(f"end line 인라이어(개): {end_line_inliers}")

    rotation_points += first_point
    
    plot_3d(origin_points, rotation_points, proj_plane_points, X, Y_fit, start_line, end_line, minima, min_idx)


if __name__ == "__main__":
    # 데이터 불러오기
    points = np.loadtxt("./data/1_fillet_gap.txt")

    # 값 입력하기(다항식 차수, 회전 방향(진행 방향), 직선 회귀 데이터: 갯수/반복 횟수/임계값)
    # 다항식 차수 결정
    degree = 3

    # 센서 게더링 방향(== 툴 이동 방향)
    axis_dir = "X"

    # 직선 회귀 데이터
    data_count = 30         # 데이터 개수
    iterations = 1000        # 반복 횟수, 기본값 100
    threshold = 0.05         # 임계값, 기본값 0.1

    # main 함수 호출
    process_3d_data(points, degree, axis_dir, data_count, iterations, threshold)