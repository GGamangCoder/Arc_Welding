# 선분(segments)과 선분 사이의 거리
# 선분/직선 유무 결정

import numpy as np


def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):
    # 직선/선분 여부 체크, 직선일 경우 true
    if clampAll:
        clampA0 = True
        clampA1 = True
        clampB0 = True
        clampB1 = True

    A = a1 - a0
    B = b1 - b0
    normA = np.linalg.norm(A)
    normB = np.linalg.norm(B)

    d1 = A / normA
    d2 = B / normB

    cross = np.cross(d1, d2);
    denom = np.linalg.norm(cross)**2

    # 평행할 때(denom = 0), 두 선 사이의 거리 계산
    if not denom:       # 평행할 때(denom = 0)
        d0 = np.dot(d1, (b0 - a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(d1, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 and d1 <= 0:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0, b0, np.linalg.norm(a0 - b0)
                    return a0, b1, np.linalg.norm(a0 - b1)

            # Is segment B after A?
            elif d0 >= normA and d1 >= normA:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1,b0,np.linalg.norm(a1 - b0)
                    return a1,b1,np.linalg.norm(a1 - b1)


        # Segments overlap, return distance between parallel segments
        return None, None, np.linalg.norm(((d0 * d1) + a0) - b0)


    # 평행하지 않을 경우,
    t = (b0 - a0);
    detA = np.linalg.det([t, d2, cross])
    detB = np.linalg.det([t, d1, cross])

    t0 = detA/denom;
    t1 = detB/denom;

    pA = a0 + (d1 * t0) # Projected closest point on segment A
    pB = b0 + (d2 * t1) # Projected closest point on segment B


    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > normA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > normB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > normA):
            dot = np.dot(d2, (pA - b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > normB:
                dot = normB
            pB = b0 + (d2 * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > normB):
            dot = np.dot(d1, (pB - a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > normA:
                dot = normA
            pA = a0 + (d1 * dot)


    return pA, pB, np.linalg.norm(pA - pB)
