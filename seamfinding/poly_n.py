from get_data_points import read_line

import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# 다차 곡선 회귀 추정 - x, y, z: 3d 좌표, n_degree: 차수(default==3)
def fit_polynomial_regression(x, y, z, n_degree=3):
    '''
    Parameters:
    - x, y, z: 3차원 데이터
    - n_degree: 다항식 차수(default==3)

    Returns:
    - model: 회귀 모델
    - coeffs: 회귀 계수
    - intercept: 절편
    '''
    # x와 y에 대해 다항식 특징 생성
    poly = PolynomialFeatures(degree=n_degree)
    X_poly = poly.fit_transform(np.column_stack((x, y)))  # x와 y를 결합

    # z에 대한 회귀 모델 학습
    model = LinearRegression()
    model.fit(X_poly, z)

    z_pred = model.predict(X_poly)

    return model, z_pred

    # poly = PolynomialFeatures(degree=n_degree)
    # X_poly = poly.fit_transform(x.reshape(-1, 1))

    # # y에 대한 회귀 모델 학습
    # model_y = LinearRegression()
    # model_y.fit(X_poly, y)

    # # z에 대한 회귀 모델 학습
    # model_z = LinearRegression()
    # model_z.fit(X_poly, z)

    # # 예측값 계산
    # y_pred = model_y.predict(X_poly)
    # z_pred = model_z.predict(X_poly)

    # return model_y, y_pred, model_z, z_pred


# plot 시각화하기
def plot_3d(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    point_x, point_y, point_z = points[:, 0], points[:, 1], points[:, 2]

    # 원래 점들 (아웃라이어 포함)
    ax.scatter(point_x, point_y, point_z, color='green', alpha=0.5, s=20, label='data points')

    # 다항식 회귀 모델 학습 및 예측
    # sorted_indices = np.argsort(point_z)
    # point_x = point_x[sorted_indices]
    # point_y = point_y[sorted_indices]
    # point_z = point_z[sorted_indices]

    model, z_pred = fit_polynomial_regression(point_x, point_y, point_z, 2)

    ax.plot(point_x, point_y, z_pred, color='red', label='Fitted Curve', lw=2)

    # ax.plot(point_x[:], point_y[:], z_pred[:], color='blue', label='Fitted Curve', lw=2)

    ''' -------------------------------------------------------------------- '''

    # 예측된 z 값을 따라 데이터를 정렬 (곡선을 그리기 위한 순서)
    # sorted_indices = np.argsort(point_x)  # x 값에 따라 데이터 정렬

    # x_sorted, y_sorted, z_sorted = point_x[:], pred_y[:], pred_z[:]

    # 예측된 z 값을 따라 2D 곡선 그리기
    # ax.plot(x_sorted, y_sorted, z_sorted, color='blue', label='Fitted Curve', lw=2)
    # ax.plot(pred[:, 0], pred[:, 1], point_z[:], color='blue', label='Fitted Curve', lw=2)

    
    ''' -------------------------------------------------------------------- '''

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
    file_path = './data/2_fillet_gap_2.txt'
    # file_path = './data/8_single_bevel.txt'
    points = read_line(file_path)

    plot_3d(points)



if __name__ == "__main__":
    execute()
