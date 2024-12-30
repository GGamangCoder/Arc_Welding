from get_data_points import read_line

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def transition(points):
    first_ = points[0]
    points = points - first_

    return points

def rotation(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    new_points = np.vstack((x, y, z))

    angle = np.arctan2(np.mean(y), np.mean(x))  # x축과의 각도 계산

    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])

    rot_points = R_roll @ new_points

    return rot_points


def plot_3d(points):
    # 회전된 데이터로 새로운 x, y, z 값
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # 회전된 데이터를 시각화
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, alpha=0.5, s=20, label='Raw Data', color='b')

    # 축 범위 설정
    ax.set_xlim(np.min(points[:, 0] - 2), np.max(points[:, 0] + 2))
    ax.set_ylim(np.min(points[:, 1] - 2), np.max(points[:, 1] + 2))
    ax.set_zlim(np.min(points[:, 2] - 2), np.max(points[:, 2] + 2))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()



def execute():
    file_path = './data/1_fillet_gap.txt'
    # file_path = './data/8_single_bevel.txt'
    points = read_line(file_path)

    points = transition(points)

    plot_3d(points)



if __name__ == "__main__":
    execute()
