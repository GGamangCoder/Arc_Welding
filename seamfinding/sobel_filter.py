
import numpy as np
import copy

import matplotlib.pyplot as plt

import cv2
from scipy.ndimage import convolve

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


''' ---------------------------------------------------- '''
def gaussian_kernel(size=5, sigma=1.5):
    """ size x size 크기의 가우시안 커널 생성 """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)  # 합이 1이 되도록 정규화

def gaussian_kernel_1d(size, sigma):
    ''' size: kernel 크기, sigma: 표준 편차 '''
    x = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()      # 합이 1이 되도록 정규화

    return kernel

def sobel_filter(image):    
    # Sobel 커널 정의
    Gx = np.array([[-1, 0, 1], 
                   [-2, 0, 2], 
                   [-1, 0, 1]], dtype=np.float32)

    Gy = np.array([[-1, -2, -1], 
                   [0,  0,  0], 
                   [1,  2,  1]], dtype=np.float32)
    

    # 컨볼루션 적용
    grad_x = convolve(image, Gx, mode='constant', cval=0.0)
    grad_y = convolve(image, Gy, mode='constant', cval=0.0)

    # 최종 그래디언트 크기 계산
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return grad_x, grad_y, gradient_magnitude


def sobel_filter_1d(val, scale=1, delta=0):
    # 1d 타입 커널 형성
    kernel = np.array([-1, 0, 1])

    # 데이터 타입을 np.float64로 변환
    val = val.astype(np.float64)

    sobel_ = np.zeros_like(val, dtype=np.float64)       # 초기화 과정

    # 경계 처리를 위한 반사 패딩 (reflect padding)
    padded_ = np.pad(val, pad_width=1, mode='reflect')  # 양 옆에 1개의 요소를 반사 패딩으로 추가 == 왼쪽 끝을 안으로, 1번 항목이 0,2 번으로, 2번부터 뒤로 하나씩 밀리기, 맨 뒤에서도 마찬가지, 가운데서 역전되나?

    for i in range(1, len(val) - 1):
        sobel_[i] = np.sum(kernel * padded_[i-1 : i+2])

    # 정규화
    sobel_ = scale * sobel_ + delta

    # 배열 모양 맞추기
    sobel_ = sobel_.reshape(-1, 1)

    return sobel_


def plot_3d(weld_type, origin_points, dir, rot_points, x, y, filter_thres):
    # Sobel 필터 적용
    # cv2.Sobel(src, ddepth, dx, dy, dst=None, ksize=None, scale=None, delta=None, borderType=None)

    # x, y 모두 좌표로 1d 타입. dx=1을 줘도 의미가 없음.
    y_2d = y.reshape((-1, 1))
    image_smooth = cv2.GaussianBlur(y_2d, (9, 1), 2.0)
    # sobel_x = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3)  # 수직
    sobel_y = cv2.Sobel(image_smooth, cv2.CV_64F, 0, 1, ksize=3)  # 수직
    sobel_y_abs = np.sqrt(sobel_y**2)
    print(f'lib: {sobel_y_abs[:10]}')

    # image = np.vstack((x, y)).T
    # g_x, g_y, mag = sobel_filter(image)
    # mag_f = [f'{value:.6f}' for value in mag[1:10].flatten()]
    # print(f'Img: {image[:10]}')
    # print(f'Mag: {mag[:10]}')

    # 가우시안 필터 적용
    k_size = 5      # 가우시안 블러 커널 크기
    sigma = 1.0     # 표준 편차
    gaussian_k = gaussian_kernel_1d(k_size, sigma)
    smoothed_data = convolve(y, gaussian_k)
    
    print(y[:10])
    print(smoothed_data[:10])

    sobel_ = sobel_filter_1d(smoothed_data)

    # 엣지 세기 계산 -- y의 결과가 음수로 나오면 이상하게 나와서 필요하긴 하겠음!!
    try:
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
    except:
        edges = np.sqrt(sobel_y**2)
        print(f'1df: {edges[:10]}')

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

    # 필터 적용 결과 좌표 표시 (원본 데이터 위에 표시) & 해당 index 저장
    idx = []
    for i in range(len(x)):
        if edges[i] > np.max(edges) * filter_thres:  # 필터 결과가 특정 값 이상인 경우
            ax.text(x[i], y[i], f'({x[i]:.2f}, {y[i]:.2f})', fontsize=8, color='red')
            idx.append(i)
    print(f'엣지 후보군: {idx}')
    
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
    weld_type = "4_butt_wide"
    # weld_type = "6_butt_narrow"
    # weld_type = "8_single_bevel"
    points = np.loadtxt(f"./data/{weld_type}.txt")

    # 값 입력하기(다항식 차수, 회전 방향(진행 방향), 직선 회귀 데이터: 갯수/반복 횟수/임계값)\

    # 센서 게더링 방향(== 툴 이동 방향)
    axis_dir = "X"

    # sobel filter 임계값
    filter_thres = 0.4

    # main 함수 호출
    process_3d_data(weld_type, points, axis_dir, filter_thres)
