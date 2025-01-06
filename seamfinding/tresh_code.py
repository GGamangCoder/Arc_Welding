# 회전 변환
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


# Step 3-2: Gradient Descent (경사하강법)
def find_minima_GD(poly_coefficients, x_start, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
    """
    경사하강법을 사용하여 최저점 찾기.

    Parameters:
        poly_coefficients: 다항식 계수 리스트.
        x_start: 초기값.
        learning_rate: 학습률.
        max_iter: 최대 반복 횟수.
        tolerance: 수렴 허용 오차.
    Returns:
        (x_min, y_min): 최저점 좌표.
    """
    poly = np.poly1d(poly_coefficients)
    gradient = np.polyder(poly)
    x_current = x_start

    for _ in range(max_iter):
        x_next = x_current - learning_rate * gradient(x_current)
        if abs(x_next - x_current) < tolerance:
            break
        x_current = x_next

    y_min = poly(x_current)
    return x_current, y_min






# Step 5: RANSAC for line fitting near the minima
def fit_line_near_minima(points, minima_point, window=0.05):     # window: 최저점 근처 영역 range
    # Filter points near the minima
    filtered_points = points[
        (points[:, 0] > minima_point[0] - window) &
        (points[:, 0] < minima_point[0] + window)
    ]
    if len(filtered_points) < 2:
        raise ValueError("Insufficient points near minima for line fitting.")
    
    x, y = filtered_points[:, 0], filtered_points[:, 1]
    ransac = RANSACRegressor()
    ransac.fit(x.reshape(-1, 1), y)

    return ransac

# 직선 회귀
try:
    line_model = fit_line_near_minima(projected_points, minima)
    print(f"Line equation near minima: y = {line_model.estimator_.coef_[0]}x + {line_model.estimator_.intercept_}")

except ValueError as e:
    print(str(e))