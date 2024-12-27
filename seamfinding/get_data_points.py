# 파일 불러오는 함수


import numpy as np

def read_line(file_path):
    pos = []
    small = 9999;
    small_idx = 0;
    idx = 0;
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            if values[2] < small:
                small = values[2]
                small_idx = idx
            if len(values) == 3:
                pos.append(values)
            idx += 1
    
    print(small_idx, small)
    points = np.array(pos)
    return points