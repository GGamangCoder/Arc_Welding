import numpy as np
import matplotlib.pyplot as plt


# 로드리게즈 회전 변환 구현, dir은 진행 방향 - x 혹은 y
def rotation_formula(data, dir):
    # 첫 번째 점과 마지막 점을 사용해 벡터 계산
    P_start = data[0]
    P_end = data[-1]
    vector = P_end - P_start

    # 단위 벡터 변환
    unit_vector = vector / np.linalg.norm(vector)

    # (1, 0, 0) 방향 벡터
    if dir == 0:
        target_vector = np.array([1, 0, 0])
    elif dir == 1:
        target_vector = np.array([0, 1, 0])

    # 회전 축 (벡터의 외적)
    rotation_axis = np.cross(unit_vector, target_vector)

    # 회전 각도 (두 벡터 간의 내적을 이용하여 계산)
    cos_theta = np.dot(unit_vector, target_vector)
    theta = np.arccos(cos_theta)

    # 회전 행렬 (Rodrigues' rotation formula)
    K = np.array([
        [0                 , -rotation_axis[2] , rotation_axis[1]],
        [rotation_axis[2]  , 0                 , -rotation_axis[0]],
        [-rotation_axis[1] , rotation_axis[0]  , 0]
    ])

    # 로드리게즈 회전 변환 공식
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    # 데이터를 변환
    transformed_data = np.dot(data, R.T)

    return transformed_data

'''---------------------------------------------------------------------------------------------------------------'''

def main_func(points):
    # 데이터 정렬을 위해 첫 점을 원점으로 이동
    points -= points[0]

    # 진행 방향
    dir = "X"
    # 주축에 따라서 축 설정
    # X 축이면 인덱스 0, Y 축이면 인덱스 1
    axis_idx = 0 if dir == "X" else 1
    print(dir, "축: ", axis_idx)

    transformed_data = rotation_formula(points, axis_idx)


    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')

    # -------------------------------------------- 1번 그래프
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], label='Origin points', color='g', s=3)
    ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], label='Trans points', color='b', s=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title("3d Points & Rotation")
    ax.legend()        # 범례 시각화

    # -------------------------------------------- 2번 그래프
    ax2 = fig.add_subplot(122)
    ax2_label = f'{dir}Z points'
    ax2.scatter(transformed_data[:, axis_idx], transformed_data[:, 2], label=ax2_label, s=5)     # 주축에 따라서 축 설정

    # ax2.scatter(projected_points[:, 0], projected_points[:, 1], label='Projected Points', s=1)
    # ax2.plot(poly_x, poly_y, color='orange', label='Fitted Polynomial')
    # ax2.scatter(minima[0], minima[1], color='red', label='Minima', s=20)

    ax2.set_xlabel(f'{dir}')
    ax2.set_ylabel('Z')

    ax2.set_title(f"2D {dir}Z Proj Plane")
    ax2.legend()

    plt.show()


if __name__ == "__main__":
    points = np.loadtxt("./data/3_circle_hole.txt")
    
    main_func(points)