from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def read_line(file_path):
    pos = []
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            if len(values) == 3:
                pos.append(values)
    
    points = np.array(pos)

    return points
    

def plot_3d_with_plane(pos, theta):
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    # 추정된 평면의 z값 구하기 -> z = a*x + b*y + c
    x_range, y_range = np.meshgrid(np.linspace(min(x), max(y)),
                                np.linspace(min(x), max(y)))

    z_pred = theta[0] * x_range + theta[1] * y_range + theta[2]


    fig = plt.figure()
    plt.title('sensor getter data')

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, marker='o', s=5)

    ax.plot_surface(x_range, y_range, z_pred, color='r', alpha=0.5, label="Estimated Plane")

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # plt.legend()
    plt.show()


def distance_point_to_line(point, line_point1, line_point2):
    # 주어진 점과 직선(두 점) 사이의 거리 계산
    line_vec = line_point2 - line_point1
    point_vec = point - line_point1
    line_length_squared = np.dot(line_vec, line_vec)
    
    if line_length_squared == 0:
        return np.linalg.norm(point_vec)
    
    projection = np.dot(point_vec, line_vec) / line_length_squared
    projection_vec = projection * line_vec
    nearest_point = line_point1 + projection_vec
    return np.linalg.norm(nearest_point - point)

def ransac(points, n=50, threshold=0.1, iterations=100):
    best_inlier_count = 0
    best_line = None

    for _ in range(iterations):
        # 두 개의 랜덤 포인트 선택
        sample_indices = np.random.choice(points.shape[0], size=2, replace=False)
        line_point1 = points[sample_indices[0]]
        line_point2 = points[sample_indices[1]]

        # 모든 점에 대해 거리 계산
        distances = [distance_point_to_line(point, line_point1, line_point2) for point in points]

        # 인라이어 결정
        inliers = np.array([points[i] for i in range(len(distances)) if distances[i] < threshold])
        inlier_count = len(inliers)

        # 최적의 인라이어 수가 최대인 경우 업데이트
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_line = (line_point1, line_point2)

    return best_line



def plot_3d(pos, line_start, line_end):
    # print("좌표 갯수: ", pos.shape)
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    fig = plt.figure()
    plt.title('sensor getter data')

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, marker='o', s=5)

    # min_index = np.argmin(z)
    # min_pos = pos[min_index]

    # print("변곡점: ", min_pos)

    # 최솟값을 가지는 점을 강조 표시
    # ax.scatter(min_pos[0], min_pos[1], min_pos[2], color='red', s=50, label="Minimum Point")

    # 직선 그리기
    ax.plot(*zip(line_start, line_end), color='r', label='RANSAC Line')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    # ax.legend()

    plt.show()



lts_file = "LTS_gather.txt"

pos = read_line(lts_file)

line_start = ransac(pos[:50])
line_end = ransac(pos[-50:])

plot_3d(pos, line_start[0], line_start[1])


# theta = fit_plane_least_squares(pos)

# plot_3d_with_plane(pos, theta)


# plane_params_ransac = estimate_plane_ransac(pos)

# plot_3d_with_plane(pos, plane_params_ransac)