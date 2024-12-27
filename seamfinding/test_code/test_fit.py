import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 1. 데이터 파일 읽기 (LTS_gather.txt와 유사한 형식)
data = np.loadtxt('../data/LTS_gather.txt')  # 데이터 파일 경로를 맞춰주세요

# x, y, z 좌표 추출
X = data[:, :2]  # 첫 두 컬럼: x, y
Z = data[:, 2]   # 세 번째 컬럼: z

# 2. 다항식 차수 설정 (N차)
degree = 4  # 예시로 3차 다항식 모델

# 3. 다항식 특징 생성 (x, y -> x, y, x^2, xy, y^2, ...)
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)

# 4. RANSAC 모델 초기화
ransac = RANSACRegressor()
ransac.fit(X_poly, Z)

# 5. RANSAC 모델을 이용한 예측값
Z_pred = ransac.predict(X_poly)

# 6. RANSAC 모델의 평균 제곱 오차 (MSE)
mse = mean_squared_error(Z, Z_pred)
print(f"Mean Squared Error: {mse}")

# 7. 결과 시각화 (3D 그래프)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 원본 데이터 (x, y, z)
ax.scatter(X[:, 0], X[:, 1], Z, color='blue', label='Original Data')

# RANSAC 모델로 예측된 z 값
ax.scatter(X[:, 0], X[:, 1], Z_pred, color='red', label='RANSAC Predicted')

# 레이블 추가
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
