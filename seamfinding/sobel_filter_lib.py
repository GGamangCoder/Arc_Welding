
import numpy as np
import copy

import matplotlib.pyplot as plt

import cv2

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

    unit_vector = vector / np.linalg.norm(vector)    # 단위 벡터 변환

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


def plot_3d(weld_type, origin_points, dir, rot_points, x, y, filter_thres):
    image = np.vstack((x, y)).T

    # 노이즈 제거를 위한 가우시안 필터 적용
    image_smooth = cv2.GaussianBlur(image.astype(np.float32), (5, 5), 1.5)

    # Sobel 필터 적용
    sobel_x = cv2.Sobel(image_smooth, cv2.CV_64F, 1, 0, ksize=3)  # 수직
    sobel_y = cv2.Sobel(image_smooth, cv2.CV_64F, 0, 1, ksize=3)  # 수직

    # 엣지 세기 계산
    edges = np.sqrt(sobel_x**2 + sobel_y**2)

    #-------------------------------------
    # 시각화 하기
    fig = plt.figure(figsize=(12, 6))
    # 전체 제목 추가
    fig.suptitle(f'Type: {weld_type} (thres={filter_thres})', fontsize=16)

    # 왼쪽 그래프: 원본 데이터 (x, y)
    ax = fig.add_subplot(121)
    ax.scatter(x, y, color='blue', label='Original Data')
    ax.set_title('Original Data')
    ax.set_xlabel(f'{dir}')
    ax.set_ylabel('Z')

    # 오른쪽 그래프: Sobel 필터 적용 결과
    ax2 = fig.add_subplot(122)
    ax2.imshow(edges, cmap='gray', aspect='auto')        # aspect='auto' x, y 축 균등
    ax2.set_title('Sobel Filter')
    # ax2.axis('off')  # x, y 축을 숨깁니다.

    idx = []
    # 필터 적용 결과 좌표 표시 (원본 데이터 위에 표시)
    for i in range(len(x)):
        if edges[i][0] > np.max(edges) * filter_thres:  # 필터 결과가 특정 값 이상인 경우
            ax.text(x[i], y[i], f'({x[i]:.2f}, {y[i]:.2f})', fontsize=8, color='red')
            idx.append(i)

    print(idx)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 위에 제목 공간을 확보
    plt.show()

# Main pipeline
def process_3d_data(weld_type, points, dir, filter_thres=0.3):
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
        proj_x = rotation_points[:, 0]
        proj_y = rotation_points[:, 2]
    elif dir == "Y":
        proj_x = rotation_points[:, 1]
        proj_y = rotation_points[:, 2]
    else:
        raise ValueError("잘못된 축이 입력되었습니다.")

    plot_3d(weld_type, origin_points, dir, rotation_points, proj_x, proj_y, filter_thres)


if __name__ == "__main__":
    # 데이터 불러오기
    # weld_type = "1_fillet_gap"
    weld_type = "5_butt_wide_2"
    # weld_type = "6_butt_narrow"
    # weld_type = "8_single_bevel"
    points = np.loadtxt(f"./data/{weld_type}.txt")

    # 값 입력하기(다항식 차수, 회전 방향(진행 방향), 직선 회귀 데이터: 갯수/반복 횟수/임계값)

    # 센서 게더링 방향(== 툴 이동 방향)
    axis_dir = "X"

    # sobel filter 임계값
    filter_thres = 0.6

    # main 함수 호출
    process_3d_data(weld_type, points, axis_dir, filter_thres)
