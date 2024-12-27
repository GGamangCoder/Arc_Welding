# 직선과 직선 사이의 거리

import numpy as np

    
def closest_distance_between_lines(a, d1, b, d2):
    """
    평행하지 않은 두 직선 사이의 최단 거리와 최단 거리를 이루는 점 P1, P2를 계산하는 함수
    
    Args:
    a : np.array - 직선 L1 위의 한 점 (x1, y1, z1)
    d1 : np.array - 직선 L1의 방향 벡터 (a1, b1, c1)
    b : np.array - 직선 L2 위의 한 점 (x2, y2, z2)
    d2 : np.array - 직선 L2의 방향 벡터 (a2, b2, c2)
    
    Returns:
    distance : float - 두 직선 사이의 최단 거리
    P1 : np.array - 직선 L1 상의 최단 거리를 이루는 점
    P2 : np.array - 직선 L2 상의 최단 거리를 이루는 점
    """
    
    ab = b - a
    
    d1_dot_d1 = np.dot(d1, d1)
    d2_dot_d2 = np.dot(d2, d2)
    d1_dot_d2 = np.dot(d1, d2)
    ab_dot_d1 = np.dot(ab, d1)
    ab_dot_d2 = np.dot(ab, d2)
    
    # 두 직선 사이의 최단 거리 상의 매개변수 t, s 계산
    denom = d1_dot_d1 * d2_dot_d2 - d1_dot_d2 ** 2

    if abs(denom) < 0.001:  # 방향 벡터가 거의 평행한 경우 처리
        t, s = 0, ab_dot_d2 / d2_dot_d2  # 두 점을 직선의 초기점으로 고정
    else:
        t = (ab_dot_d1 * d2_dot_d2 - ab_dot_d2 * d1_dot_d2) / denom
        s = (ab_dot_d2 * d1_dot_d1 - ab_dot_d1 * d1_dot_d2) / denom
        
    P1 = a + t * d1
    P2 = b + s * d2
   
    return P1, P2, np.linalg.norm(P2 - P1)

