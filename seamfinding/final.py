'''
0. 3차원 점들을 원점으로 이동
1. 방향성 없는 점들의 좌표계 설정 - 이동방향 기준 X 축 혹은 Y축(로드리게즈 회전 변환)
2. 한 평면(XZ 혹은 YZ)으로 사영하여 다항 회귀 추정 - RANSAC
3. global minima를 찾아 근접 인덱스 구하기
4. 찾은 인덱스 전/후 미분값을 통해 직선 구간 구하기 -> 모재 결정
5. 모재에 따라 seam finding(최종 seam 결정)
6. 해당 결과 시각화
'''

import numpy as np

import copy
import random

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

from line_to_line import closestDistanceBetweenLines        # Hi6 PlyUtlCalculateXLine 함수 통해서 결정

# 글로벌 변수 선언 - cpp define
GROOVE_TYPE = 0
BUTT_TYPE = 1


'''---------------------------- 공통 함수 - 직접 구현 ----------------------------'''
# 평면에서 두 점 사이의 거리
def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2 ) ** 0.5

    return distance

# 벡터의 크기 구하는 함수
def norm(vec):
    norm_vector = (vec[0]**2 + vec[1]**2 + vec[2]**2) ** 0.5
    
    return norm_vector


# Step1: 좌표계 설정 및 회전 변환
def rotation_formula(data, dir):
    """
    3D 점들의 좌표계를 설정하고, 주어진 방향으로 회전 변환을 수행한다.
    이동 방향을 기준으로 로드리게즈 회전 변환 공식을 이용한다.

    params:
        data (ndarray): 3D 데이터 배열. (x, y, z) 좌표를 의미.
        dir (str): 데이터들을 정렬할 방향(기본값: "X")

    return:
        transformed_data (ndarray): 회전 변환해서 얻은 데이터
    """
    P_start = data[10]
    P_end = data[-11]
    vector = P_end - P_start

    unit_vector = vector / norm(vector)    # 단위 벡터 변환

    # 로봇 이동 방향(데이터 수집)을 회전축으로 사용
    if dir == "X":
        target_vector = np.array([1, 0, 0])
    elif dir == "Y":
        target_vector = np.array([0, 1, 0])
    else:
        raise ValueError("잘못된 축이 입력되었습니다.")

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


# Step 2: 다항식 회귀
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


# Step 3: 특이점 찾기
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
        # min_index = np.argmin(y_values)
        min_index = get_smallest_index(y_values)
        min_x, min_y = minima_candidates[min_index], y_values[min_index]
        return [min_x, min_y]
    else:
        # raise ValueError("유효한 최솟값을 찾을 수 없습니다. 주어진 x 범위 내에 최솟값이 없습니다.")
        return [None, None]

# Step 3-1: 주어진 배열에서 가장 작은 값의 인덱스 반환 함수
def get_smallest_index(arr):
    min_value = arr[0]
    min_index = 0

    for i in range(1, len(arr)):  # 첫 번째 값을 제외한 나머지 값들에 대해
        if arr[i] < min_value:
            min_value = arr[i]
            min_index = i

    return min_index

# Step 3-2: 찾은 결과와 가장 가까운 데이터(인덱스) 구하기
def find_closest_index(x_range, min_x):
    # numpy 내장 라이브러리 활용
    # closest_index = np.argmin(np.abs(x_range - min_x))

    closest_index = -1       # index는 무조건 0 ~ len(x_range) 사이를 가질 수 밖에 없음.
    min_diff = float('inf')

    i = 0
    for x in x_range:
        diff = abs(x - min_x)
        if diff < min_diff:
            min_diff = diff
            closest_index = i
        i += 1

    return closest_index


# Step 4: 최저점 기준으로 평평한 직선 부분 구하기
def get_flat_line(points, idx, threshold=0.5):
    """
    주어진 점에서의 1차 미분값을 계산하고, 주어진 threshold보다 작은 미분값을 갖는 구간을 찾는다.

    params:
        points (ndarray): x, y 좌표들을 포함하는 2D 배열
        idx (int): 기준점의 인덱스
        threshold (float): 미분값의 임계값

    return:
        left_idx (int): 미분값이 threshold 이상인 가장 왼쪽 인덱스
        right_idx (int): 미분값이 threshold 이상인 가장 오른쪽 인덱스
    """
    x, y = points[:, 0], points[:, 1]

    # 라이브러리 이용
    # dy_dx = np.gradient(y, x)

    dy_dx = [0] * len(y)  # dy_dx를 저장할 리스트 초기화

    # 중앙 차분 계산 (0 < i < len(y)-1 범위에서)
    for i in range(1, len(y) - 1):
        dy_dx[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])

    # 첫 번째 점과 마지막 점은 앞 차분과 뒤 차분으로 처리
    dy_dx[0] = (y[1] - y[0]) / (x[1] - x[0])  # 앞 차분
    dy_dx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])  # 뒤 차분

    left_idx = idx
    while (left_idx > 0) and (abs(dy_dx[left_idx]) < threshold):
        left_idx -= 1

    right_idx = idx
    while (right_idx < len(points)) and (abs(dy_dx[right_idx]) < threshold):
        right_idx += 1

    print('left/right idx: ', left_idx, right_idx)
    print(f'left/right X val: {points[left_idx][0]:.6f}, {points[right_idx][0]:.6f}')

    return left_idx, right_idx

# Step 5: 형상에 따라 최종 seam point를 찾아 해당 인덱스 혹은 좌표를 반환하는 함수
# Step 5-1: 현재 형상이 groove(계곡)인 경우
def process_groove_type(points, idx, cnt, iter, threshold):
    # 직선 회귀 점들 추출: 기준점으로부터 양 쪽으로 count 만큼
    start_points = points[idx - cnt : idx, :]
    end_points = points[idx : idx + cnt, :]

    start_line, start_line_inliers = fit_line_ransac(start_points, iter, threshold)
    end_line, end_line_inliers = fit_line_ransac(end_points, iter, threshold)

    # print(f"start line 인라이어(개): {start_line_inliers} / {len(start_points)}")
    # print(f"end line 인라이어(개): {end_line_inliers} / {len(end_points)}")

    p1, direction_1 = start_line
    line_1_points = np.array([p1 + t * direction_1 for t in np.linspace(-25, 25, 10)])          # 숫자는 그냥 스케일이라 시각화에만 관여

    p2, direction_2 = end_line
    line_2_points = np.array([p2 + t * direction_2 for t in np.linspace(-25, 25, 10)])

    intersection_1, intersection_2, dist_1_to_2 = closestDistanceBetweenLines(p1, p1+direction_1, p2, p2+direction_2)

    return start_line, end_line, intersection_1

# Step 5-2: 현재 형상이 butt(평평한 모재 접합)인 경우 -- 중간값
# 바닥 직선에서 직선 회귀 통해 구간의 가운데로 하니 오히려 오차가 더 발생함.
def process_butt_type(left, right):
    seam_idx = (left + right) // 2

    return seam_idx

# RANSAC for line fitting near the minima
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
        direction = direction / norm(direction)

        inliers_cnt = 0
        for point in points:
            cross_prod = norm(np.cross(point - p1, direction))
            dist_point2line = cross_prod / norm(direction)
            if dist_point2line < threshold:
                inliers_cnt += 1

        if inliers_cnt > best_inliers_cnt:
            best_inliers_cnt = inliers_cnt
            best_line = (p1, direction)
        
    return best_line, best_inliers_cnt


# Step Final: Visualization utility
def plot_3d(weld_type, origin_points, dir, rot_points, projected_points, poly_x, poly_y, line_1, line_2, min_idx, seam):

    fig = plt.figure(figsize=(12, 6))
    fig.canvas.manager.set_window_title(f"{weld_type}")

    # Original 3D points
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(origin_points[:, 0], origin_points[:, 1], origin_points[:, 2], label='Origin Points', color='g', s=2)
    ax.scatter(rot_points[:, 0], rot_points[:, 1], rot_points[:, 2], label='Formula Points', color='b', s=2)

    ax.scatter(origin_points[min_idx][0], origin_points[min_idx][1], origin_points[min_idx][2], label='Minima Point', color='r', s=20)

    if welding_type == BUTT_TYPE:
        seam_idx = seam
        seam_point = origin_points[seam_idx]
        ax.scatter(seam_point[0], seam_point[1], seam_point[2], label='Seam Point', color='#39FF14', s=20)
        print(f"seam_idx / total: {seam_idx} / {len(origin_points)}")
        print(f"seam 좌표: {seam_point}")
    else:
        p1, direction_1 = line_1
        line_1_points = np.array([p1 + t * direction_1 for t in np.linspace(-25, 25, 10)])          # 숫자는 그냥 스케일이라 시각화에만 관여
        ax.plot(line_1_points[:, 0], line_1_points[:, 1], line_1_points[:, 2], color='#FFFF00', label='Start line')

        p2, direction_2 = line_2
        line_2_points = np.array([p2 + t * direction_2 for t in np.linspace(-25, 25, 10)])
        ax.plot(line_2_points[:, 0], line_2_points[:, 1], line_2_points[:, 2], color='#FFA500', label='End line')

        print(f'seam 좌표: [ {seam[0]:.6f} {seam[1]:.6f} {seam[2]:.6f} ]')
        ax.scatter(seam[0], seam[1], seam[2], label='Seam Point', color='#39FF14', s=20)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title("3D Points and Rotation")
    ax.legend()
    
    # 2D projected points and polynomial
    ax2 = fig.add_subplot(122)
    ax2.scatter(projected_points[:, 0], projected_points[:, 1], label='Projected Points', s=1)
    ax2.plot(poly_x, poly_y, color='orange', label='Fitted Polynomial')
    ax2.scatter(projected_points[min_idx][0], projected_points[min_idx][1], label='Minima', color='r', s=20)
    if welding_type == BUTT_TYPE:
        seam_idx = seam
        seam_point = projected_points[seam_idx]
        ax2.scatter(seam_point[0], seam_point[1], label='Seam point', color='#39FF14', s=20)

    ax2.set_xlabel(f'{dir}')
    ax2.set_ylabel('Z')

    ax2.set_title("2D Projected Points and Fitted Curve")
    ax2.legend()

    fig.suptitle(f"type: {weld_type}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 위에 제목 공간을 확보
    plt.show()

# Main pipeline
def process_3d_data(weld_type, points, n_degree, dir, count, iterations, threshold):
    # 원래 포인트 복사
    # origin_points = copy.deepcopy(points)
    origin_points = points

    # 원점 기준 변경
    first_point = copy.deepcopy(points[0])
    points -= first_point               # 이거 나중에 원복 필요.

    # 데이터 회전 변환(좌표계 설정)
    rotation_points = rotation_formula(points, dir)
    
    # X 축으로 정렬하면 xz 평면을, Y축일 경우에는 yz 평면을 본다.
    if dir == "X":
        proj_plane_points = rotation_points[:, [0, 2]]
    elif dir == "Y":
        proj_plane_points = rotation_points[:, 1:3]
    else:
        raise ValueError("잘못된 축이 입력되었습니다.")

    # 여기서 X는 주축(x or y), Y는 z를 의미한다.
    ransac, X, Y, Y_fit = fit_polynomial_ransac(proj_plane_points, degree=n_degree)

    # RANSAC 회귀 추정 객체 - 다항식의 계수와 y 절편
    poly_coefficients = ransac.estimator_.coef_
    poly_intercept = ransac.estimator_.intercept_
    poly_coefficients[0] = poly_intercept       # y 절편까지 반영
    # print(f"{n_degree}차 방정식 계수: {poly_coefficients}")

    # 회귀 곡선에서 최저점
    minima = find_global_minima(poly_coefficients, X)

    # 회귀 추정된 점과 가장 가까운 원 데이터의 index
    if minima[0] == None:
        print('minima is not defined')
        min_idx = len(points) // 2
    else:
        print('minima is defined')
        min_idx = find_closest_index(X, minima[0])

    # 여기에서 모재 타입 결정하는 함수 추가.
    plane_threshold = 0.5       # 평평한 경우에는 미분값이 거의 0에 수렴, default 기울기를 0.5로 했으나 실험 결과로 결정 필요
    left_idx, right_idx = get_flat_line(proj_plane_points, min_idx, threshold=plane_threshold)

    # 바닥 평평한 면의 길이
    bottom_distance = distance(proj_plane_points[left_idx], proj_plane_points[right_idx])
    print(f'바닥 길이: {bottom_distance:.6f}')

    dist_criteria = 0.5       # 단위는 cm, 현재 오차 고려 0.5cm
    global welding_type
    if bottom_distance < dist_criteria:
        welding_type = GROOVE_TYPE
        start_line, end_line, intersect = process_groove_type(origin_points, min_idx, count, iterations, threshold)
        seam = intersect                                    # 여기서는 seam 좌표
    else:
        welding_type = BUTT_TYPE
        seam = process_butt_type(left_idx, right_idx)       # 여기서는 seam 인덱스
        start_line, end_line = None, None       # 미사용

    # rotation_points += first_point
    
    # 형상에 따라 seam이냐 seam_idx냐 구분
    plot_3d(weld_type, origin_points, dir, rotation_points, proj_plane_points, X, Y_fit, start_line, end_line, min_idx, seam)


if __name__ == "__main__":
    # 데이터 불러오기
    # weld_type = "0_fillet"
    # weld_type = "1_fillet_gap"
    # weld_type = "2_fillet_gap_2"
    # weld_type = "3_circle_hole"
    weld_type = "4_butt_wide"
    # weld_type = "5_butt_wide_2"
    # weld_type = "6_butt_narrow"
    # weld_type = "7_butt_narrow_2"
    # weld_type = "8_single_bevel"
    points = np.loadtxt(f"./data/{weld_type}.txt")

    # 값 입력하기(다항식 차수, 회전 방향(진행 방향), 직선 회귀 데이터: 갯수/반복 횟수/임계값)
    # 다항식 차수 결정
    degree = 6

    # 센서 게더링 방향(== 툴 이동 방향)
    axis_dir = "X"

    # 직선 회귀 데이터
    data_count = 50         # 데이터 개수
    iterations = 100        # 반복 횟수, 기본값 100
    threshold = 0.05         # 임계값, 기본값 0.1

    # main 함수 호출
    process_3d_data(weld_type, points, degree, axis_dir, data_count, iterations, threshold)
