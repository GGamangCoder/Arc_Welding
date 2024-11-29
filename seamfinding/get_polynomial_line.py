import numpy as np




def fit_polynomial(data_points, degree):
    X = []
    Y = []
    
    for x, y, z in data_points:
        row = [x**i * y**j for i in range(degree+1) for j in range(degree+1 - i)]
        X.append(row)
        Y.append(z)

    X = np.array(X)
    Y = np.array(Y)

    # 최소제곱법으로 다항식 계수 추정
    coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
    return coeffs

# 1차 및 2차 미분을 수치적으로 구하는 함수
def first_derivative(coeffs, x, y):
    # 1차 도함수
    dfdx = sum(i * coeffs[i+j] * (x**(i-1)) * (y**j) for i in range(1, len(coeffs)) for j in range(len(coeffs) - i))
    dfdy = sum(j * coeffs[i+j] * (x**i) * (y**(j-1)) for i in range(len(coeffs)) for j in range(1, len(coeffs)))
    return dfdx, dfdy

def second_derivative(coeffs, x, y, degree):
    d2fdx2 = 0
    d2fdy2 = 0
    d2fdxdy = 0
    # 2차 도함수S
    for i in range(degree +1):
        for j in range(degree+1 - i):
            if x == 0 or y == 0:
                continue
            d2fdx2 += i * (i-1) * coeffs[i+j] * (x**(i-2)) * (y**j)
            d2fdy2 += j * (j-1) * coeffs[i+j] * (x**i) * (y**(j-2))
            d2fdxdy += i * j * coeffs[i+j] * (x**(i-1)) * (y**(j-1))
    return d2fdx2, d2fdy2, d2fdxdy


# 변곡점은 두 번째 도함수가 0이 되는 점이므로, 이를 찾아야 합니다.
def find_inflection_points(coeffs, x_range, y_range, degree):
    inflection_points = []
    
    min_x, min_y = 0, 0
    for x in x_range:
        for y in y_range:
            d2fdx2, d2fdy2, d2fdxdy = second_derivative(coeffs, x, y, degree)
            min_d2fdx2, min_d2fdy2 = 999, 999
            # 두 번째 도함수가 0인 지점 찾기
            if abs(d2fdx2) < 1e-8 and abs(d2fdy2) < 1e-8:  # 작은 값이면 변곡점으로 간주
                if min_d2fdx2 > abs(d2fdx2) and min_d2fdy2 > abs(d2fdy2):
                    min_d2fdx2 = abs(d2fdx2)
                    min_d2fdy2 = abs(d2fdy2)
                    min_x, min_y = x, y
    inflection_points.append((min_x, min_y))
    
    return inflection_points


def read_line(file_path):
    pos = []
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            if len(values) == 3:
                pos.append(values)
    
    points = np.array(pos)
    return points


file_path = './data/LTS_gather.txt'  # 파일 경로 설정
points = read_line(file_path)
x_range, y_range, z = points[1:, 0], points[1:, 1], points[1:, 2]


degree = 3
coeffs = fit_polynomial(points, degree)


inflection_points = find_inflection_points(coeffs, x_range, y_range, degree)
print('변곡점:', inflection_points)