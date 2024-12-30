##########################################
# 주성분 분석(PCA) - 단순 평면 표현 시각화
# 얻어지는 평면에서 특이점 찾기
#########################################

from get_data_points import read_line

import numpy as np
from sklearn.preprocessing import PolynomialFeatures        # n차 다항식
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt

########

from sklearn.decomposition import PCA


file_path = './data/1_fillet_gap.txt'  # 파일 경로 설정
points = read_line(file_path)

# X = points[:, 0]
# Y = points[:, 1]
# Z = points[:, 2]


#########################################
# RANSAC을 사용하여 평면 추정
#########################################

X = points[5:-5, :2]  # x, y 좌표
y = points[5:-5, 2]   # z 좌표

# RANSAC 모델 생성 및 학습
ransac = RANSACRegressor()
ransac.fit(X, y)

# 추정된 평면의 계수
slope_x, slope_y = ransac.estimator_.coef_
intercept = ransac.estimator_.intercept_

# 평면 방정식 출력
print(f"Estimated plane equation: z = {slope_x:.4f} x + {slope_y:.4f} y + {intercept:.4f}")


#########################################
# PCA 적용하기
#########################################

datas = points

pca = PCA(n_components=2)

X_pca = pca.fit_transform(datas)
# PCA 관련 
# vec1, vec2 = pca.components_


x_pca = X_pca[:, 0]
y_pca = X_pca[:, 1]

degree = 2

poly = PolynomialFeatures(degree)
x_poly = poly.fit_transform(x_pca.reshape(-1, 1))

ransac = RANSACRegressor()
ransac.fit(x_poly, y_pca)
y_ransac = ransac.predict(x_poly)

coeffs = ransac.estimator_.coef_[::-1]
a = []      # 계수 담는 배열
for i in coeffs:
    a.append(i)

a, b = coeffs[0], coeffs[1]

# 2차 다항식일 경우,
if degree == 2:
    # a, b = a[0], a[1]
    if a > 0:
        # 최저점 계산 (x = -b / 2a)
        min_point_x = -b / (2 * a)
        min_point_y = ransac.predict(poly.transform([[min_point_x]]))  # y값 계산
        # min_point_3d = pca.inverse_transform(np.array([[min_point_x, min_point_y]]))  # 원래 3D 좌표로 변환
        # print(f"2차 다항식 최저점 (3D 좌표): {min_point_3d}")
    else:
        min_point_x, min_point_y, min_point_3d = None, None, None

elif degree == 3:
    # a, b = a[0], a[1]
    # c = a[2]
    c = coeffs[2]
    d = ransac.estimator_.intercept_

    # 두 번째 도함수는 6ax + 2b = 0
    # 변곡점 위치 계산: x = -b / (3a)
    inflection_point_x = -b / (3 * a)
    inflection_point_y = ransac.predict(poly.transform([[inflection_point_x]]))  # 변곡점의 y값 계산

    # 최저점 계산 (두 번째 도함수가 양수일 때)
    min_point_x = inflection_point_x  # 변곡점 근처가 최저점일 수 있음
    min_point_y = inflection_point_y
    # min_point_3d = pca.inverse_transform(np.array([[min_point_x, min_point_y]]))  # 원래 3D 좌표로 변환

    # print(f"3차 다항식 변곡점 및 최저점 (3D 좌표): {min_point_3d}")

# a, b, c = coeffs[0], coeffs[1], coeffs[2]
# # print(coeffs, ransac.estimator_.intercept_)

# inflection_points = -b / (3*a)

# print(f'변곡점: {inflection_points}')


# 시각화
plt.scatter(x_pca, y_pca, color='blue', label='PCA Transformed Data')
# plt.plot(x_pca, y_ransac, color='green', label='RANSAC Fitted Curve')
# # plt.axvline(inflection_points, color='green', linestyle='--', label=f'Inflection Point at x={inflection_points:.2f}')
if min_point_x is not None:
    print("min_point_x, min_point_y:", min_point_x, min_point_y)
    plt.scatter(min_point_x, min_point_y, color='red', label='Minimum Point')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('RANSAC Polynomial Fitting on PCA Data')
plt.legend()
plt.show()