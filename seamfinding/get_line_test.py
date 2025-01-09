from read_data_points import read_line

import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
import datetime

from line_to_line import closestDistanceBetweenLines
# from line_to_line_2 import closest_distance_between_lines

from sklearn.decomposition import PCA
from numpy.polynomial.polynomial import Polynomial



def fit_plane(points):      # 3점으로부터 평면 방정식 획득
    p1, p2, p3 = points
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    d = -np.dot(normal, p1)
    return normal, d


def distance(point, normal, d):
    return abs(np.dot(normal, point) + d) / np.linalg.norm(normal)


def plane_ransac(points, n_iterations=1000, threshold=0.01):
    best_model = None
    best_inliers = []

    for _ in range(n_iterations):
        sample_points = random.sample(list(points), 3)
        normal, d = fit_plane(np.array(sample_points))

        inliers = []
        for point in points:
            if distance(point, normal, d) < threshold:
                inliers.append(point)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (normal, d)

    return best_model, best_inliers


def projection_to_plane(points, model):
    # 법선 벡터와 거리값
    normal, d = model
    projected_points = []
    for point in points:
        distance = np.dot(normal, point) + d
        projected_point = point - (distance / np.linalg.norm(normal)) * normal
        projected_points.append(projected_point)

    return np.array(projected_points)


def fit_line_ransac(points, n_iterations=100, threshold=0.1):
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


# 직선 A에 대해 한 점: pA, 방향 고유벡터: dA
def get_intersection_3(p1, d1, p2, d2):
    '''
    참고 문헌: https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments 
    '''
    d1_dot_d1 = np.dot(d1, d1)
    d2_dot_d2 = np.dot(d2, d2)
    d1_dot_d2 = np.dot(d1, d2)

    # 두 직선의 시작점 사이의 벡터
    w = p1 - p2
    w_dot_d1 = np.dot(w, d1)
    w_dot_d2 = np.dot(w, d2)

    # 방정식을 위한 행렬과 벡터 정의
    A = np.array([[d1_dot_d1, -d1_dot_d2], [-d1_dot_d2, d2_dot_d2]])
    B = np.array([-w_dot_d1, -w_dot_d2])

    # t와 s를 계산
    try:
        t, s = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return None  # 직선이 평행한 경우

    # 최단 거리 점 계산
    closest_point_on_line1 = p1 + t * d1
    closest_point_on_line2 = p2 + s * d2

    # 두 점 사이의 최단 거리 계산
    min_distance = np.linalg.norm(closest_point_on_line1 - closest_point_on_line2)

    return closest_point_on_line1, closest_point_on_line2, min_distance


def update(frame, ax):
    ax.view_init(elev=10, azim=frame)

# 현재 시간
def get_time():
    now = datetime.datetime.now()
    curTime = now.strftime("%Y_%m_%d_%H:%M:%S")
    return curTime

# 원하는 데이터 수집
def gathering_data(data, filename='make_data/dist_gather.txt'):
    curTime = get_time()
    point1 = ", ".join([f'{p:.3f}' for p in data[0]])
    point2 = ", ".join([f'{p:.3f}' for p in data[1]])
    dist = f'{data[2]:.3f}'
    data_format = f'{curTime}, 좌표1: [{point1}], 좌표2: [{point1}], 거리차: {dist}'
    with open(filename, 'a') as file:
        file.write(data_format + '\n')


def polynomial_fit(points_2d, degree=3):
    x = np.vander(points_2d[:, 0], degree+1)
    y = points_2d[:, 1]
    coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return coeffs




def plot_3d(points, inliers, model, line_1, line_2):
    normal, d = model
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # 원래 점들 (아웃라이어 포함)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='green', alpha=0.5, s=20, label='Outliers')
    
    # 인라이어 점들
    inliers = np.array(inliers)
    ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], color='blue', alpha=0.8, s=20, label='Inliers')

    #########################################################
    # 평면 그리기
    #########################################################
    # xx, yy = np.meshgrid(np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 10),
    #                      np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 10))

    # zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    # ax.plot_surface(xx, yy, zz, color='yellow', alpha=0.5, rstride=100, cstride=100)

    #########################################################
    # 직선 그리기
    #########################################################
    p1, direction_1 = line_1
    line_1_points = np.array([p1 + t * direction_1 for t in np.linspace(-100, 100, 10)])

    ax.plot(line_1_points[:, 0], line_1_points[:, 1], line_1_points[:, 2], color='cyan', label='fitting line_1')

    # ax.scatter(p1[0], p1[1], p1[2], color='red', s=100)

    p2, direction_2 = line_2
    line_2_points = np.array([p2 + t * direction_2 for t in np.linspace(-100, 100, 10)])

    ax.plot(line_2_points[:, 0], line_2_points[:, 1], line_2_points[:, 2], color='navy', label='fitting line_2')

    # ax.scatter(p2[0], p2[1], p2[2], color='red', s=100)

    # intersection_1, intersection_2 = get_intersection_2(p1, direction_1, p2, direction_2)
    # intersection_1, intersection_2, dist_1_to_2 = get_intersection_3(p1, direction_1, p2, direction_2)
    # print('직선 사이 거리: ', dist_1_to_2)

    # try:
    #     ax.scatter(intersection_1[0], intersection_1[1], intersection_1[2], color='red', s=100)
    #     ax.scatter(intersection_2[0], intersection_2[1], intersection_2[2], color='red', s=100)
    # except:
    #     print("해를 구할 수 없음")

    '''  --------------------------------------------------------------------

    #########################################################
    # 두 직선 사이 가장 가까운 거리의 두 점 찾기
    # intersection_1, intersection_2, dist_1_to_2 = closest_distance_between_lines(p1, direction_1, p2, direction_2)
    intersection_1, intersection_2, dist_1_to_2 = closestDistanceBetweenLines(p1, p1+direction_1, p2, p2+direction_2)
    print('--- point1:', intersection_1)
    print('--- point2:', intersection_2)
    print('--- between dist: ', dist_1_to_2)

    gathering_data([intersection_1.tolist(), intersection_2.tolist(), dist_1_to_2])         # 오차 보기 위해서 계속해서 저장

    ax.scatter(intersection_1[0], intersection_1[1], intersection_1[2], color='red', s=50)
    ax.scatter(intersection_2[0], intersection_2[1], intersection_2[2], color='darkred', s=50)
    #########################################################
    --------------------------------------------------------------------  '''


    # 축 범위 설정
    ax.set_xlim(np.min(points[:, 0] - 5), np.max(points[:, 0] + 5))
    ax.set_ylim(np.min(points[:, 1] - 4), np.max(points[:, 1] + 4))
    ax.set_zlim(np.min(points[:, 2] - 4), np.max(points[:, 2] + 4))

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    # ax.set_title('RANSAC Fitting - LPS')
    ax.legend()

    # z축 기준 회전 관찰 - 주석하여 생략 가능
    # ani = animation.FuncAnimation(fig, update, fargs=(ax, ), frames=360, interval=50)

    # curTime = get_time()
    # ani.save(f'make_data/result_{curTime}.gif', fps=15)

    plt.show()


def execute():
    # file_path = './data/LTS_gather.txt'  # 파일 경로 설정
    file_path = './data/5_butt_wide_2.txt'
    points = read_line(file_path)
    num = len(points) // 2
    print(num)

    start_points = points[50:(num -10), :]
    end_points = points[(num+10):-50, :]

    # RANSAC 실행
    d2_model, d2_inliers = plane_ransac(points)

    # 추정 평면에 대한 결과
    # print("Best 2d_model (normal, d):", d2_model)
    # print("Number of 2d_inliers:", len(d2_inliers))

    # 추정 평면에 사영(projection)
    # proj_start_points = projection_to_plane(start_points, d2_model)
    # proj_end_points = projection_to_plane(end_points, d2_model)

    '''  --------------------------------------------------------------------
    # projection 데이터 수집

    proj_points = projection_to_plane(points, d2_model)

    print(type(proj_points))

    filename = './data/proj_data.txt'
    np.savetxt(filename, proj_points, fmt='%.2f', delimiter=' ')

    return
    --------------------------------------------------------------------  '''

    # 시작 점과 끝점에 대해
    start_line, start_line_inliers = fit_line_ransac(start_points)
    end_line, end_line_inliers = fit_line_ransac(end_points)

    print("Best start_line model (p1, dir_1)/inliers:", start_line, start_line_inliers)
    print("Best end_line model (p2, dir_2)/inliers:", end_line, end_line_inliers)

    plot_3d(points, d2_inliers, d2_model, start_line, end_line)



if __name__ == "__main__":
    execute()
