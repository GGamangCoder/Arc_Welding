import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def get_pca(points):
    """
    PCA를 사용하여 주축 벡터와 중심점을 계산
    :param points: (N, 3) 형태의 numpy 배열 (x, y, z 데이터)
    :return: 주축 벡터 (principal_axis), 중심점 (mean_point)
    """
    pca = PCA(n_components=3)
    pca.fit(points)
    
    principal_axis = pca.components_[0]  # 첫 번째 주성분 벡터
    mean_point = np.mean(points, axis=0)  # 데이터의 중심점

    return principal_axis, mean_point

def rotate_to_align(points, principal_axis):
    """
    주축을 z축에 정렬하기 위한 회전 행렬 계산 및 적용
    :param points: (N, 3) 형태의 numpy 배열 (x, y, z 데이터)
    :param principal_axis: 주축 벡터
    :return: 회전된 데이터
    """
    # 주축을 z축으로 정렬하기 위한 회전 행렬 계산
    z_axis = np.array([0, 0, 1])  # 정렬할 축 (z축)
    rotation_axis = np.cross(principal_axis, z_axis)  # 회전 축 계산
    angle = np.arccos(np.dot(principal_axis, z_axis))  # 두 벡터 사이의 각도
    
    # 회전 행렬 생성 (Rodrigues' rotation formula)
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    # 회전 적용
    rotated_points = np.dot(points - np.mean(points, axis=0), R.T)  # 중심점 기준 회전
    return rotated_points, R


points = np.loadtxt("./data/3_circle_hole.txt")  # 파일 경로에 맞게 설정

# 주축과 중심점 계산
principal_axis, mean_point = get_pca(points)
print(f"주축 벡터: {principal_axis}")
print(f"중심점: {mean_point}")

# 주축을 z축으로 정렬
rotated_points, rotation_matrix = rotate_to_align(points, principal_axis)
print(f"회전 행렬:\n{rotation_matrix}")

# 시각화
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], label='Original Data', c='blue', alpha=0.6)
ax.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2], label='Aligned Data', c='red', alpha=0.6)
# ax.quiver(mean_point[0], mean_point[1], mean_point[2],
#           principal_axis[0], principal_axis[1], principal_axis[2],
#           length=50, color='green', label='Principal Axis')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

ax.set_title("3D Data Alignment")
ax.legend()
plt.show()