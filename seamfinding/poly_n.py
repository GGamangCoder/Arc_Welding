from get_data_points import read_line

import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline


# 다차 곡선 회귀 추정 - x, y, z: 3d 좌표, n_degree: 차수(default==2)
def fit_polynomial_regression(x, y, z, n_degree=2):
    '''
    Parameters:
    - x, y, z: 3차원 데이터
    - n_degree: 다항식 차수(default==3)

    Returns:
    - model: 회귀 모델
    - coeffs: 회귀 계수
    - intercept: 절편
    '''
    poly = PolynomialFeatures(degree=n_degree)
    Y = poly.fit_transform(np.column_stack((y, z)))

    # data = np.column_stack((x, y))

    model = RANSACRegressor(
        # estimator=LinearRegression(),
        residual_threshold=0.3,
        max_trials=100
    )
    model.fit(Y, x)

    x_pred = model.predict(Y)

    return model, x_pred




# plot 시각화하기
def plot_3d(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    point_x, point_y, point_z = points[:, 0], points[:, 1], points[:, 2]

    ax.scatter(point_x, point_y, point_z, color='green', alpha=0.5, s=20, label='data points')

    model, x_pred = fit_polynomial_regression(point_x, point_y, point_z, 2)

    ax.plot(x_pred, point_y, point_z, color='red', label='Fitted Curve', lw=2)

    # 축 범위 설정
    margin = 5
    ax.set_xlim(np.min(point_x - margin), np.max(point_x + margin))
    ax.set_ylim(np.min(point_y - margin), np.max(point_y + margin))
    ax.set_zlim(np.min(point_z - margin), np.max(point_z + margin))

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    # ax.set_title('RANSAC Fitting - LPS')
    ax.legend()

    plt.show()


def execute():
    file_path = './data/5_butt_wide_2.txt'
    # file_path = './data/8_single_bevel.txt'
    points = read_line(file_path)

    plot_3d(points)



if __name__ == "__main__":
    execute()
