# 여러가지 모델로 RANSAC 
# 사실상 spline 보간이랑 비슷한 방법

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

from get_data_points import read_line


file_path = './data/5_butt_wide_2.txt'  # 파일 경로 설정
points = read_line(file_path)

x, y, z = points[:, 0], points[:, 1], points[:, 2]

# 다항식 피팅
# estimators = [
#     ("OLS", LinearRegression()),
#     ("Theil-Sen", TheilSenRegressor(random_state=42)),
#     ("RANSAC", RANSACRegressor(random_state=42)),
#     ("HuberRegressor", HuberRegressor()),
# ]
poly_ransac = make_pipeline(PolynomialFeatures(degree=5), RANSACRegressor())
poly_ransac.fit(np.column_stack((y, z)), x)

# 곡선 생성
yz_fit = np.column_stack((y, z))
X_pred = poly_ransac.predict(yz_fit)

# 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 원본 데이터
ax.scatter(x, y, z, color='blue', label='Original Data')

# 곡선 데이터
ax.plot(X_pred, y, z, color='red', label='Fitted Curve')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()
