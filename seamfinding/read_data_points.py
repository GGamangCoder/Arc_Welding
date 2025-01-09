# 파일 불러오는 함수

import numpy as np

def read_line(file_path):
    pos = []
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            if len(values) == 3:
                pos.append(values)
    
    points = np.array(pos)

    # 첫 번째 점을 0으로 하여 shift
    # point_0 = points[0]
    # points -= point_0

    return points

# print(len(read_line('./data/1_fillet_gap.txt')))