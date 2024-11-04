# 선분(segments)과 선분 사이의 거리

import numpy as np


def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):
    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0=True
        clampA1=True
        clampB0=True
        clampB1=True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    normA = np.linalg.norm(A)
    normB = np.linalg.norm(B)

    d1 = A / normA
    d2 = B / normB

    cross = np.cross(d1, d2);
    denom = np.linalg.norm(cross)**2


    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:       # 평행할 때(denom = 0)
        d0 = np.dot(d1,(b0-a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(d1,(b1-a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0,b0,np.linalg.norm(a0-b0)
                    return a0,b1,np.linalg.norm(a0-b1)


            # Is segment B after A?
            elif d0 >= normA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1,b0,np.linalg.norm(a1-b0)
                    return a1,b1,np.linalg.norm(a1-b1)


        # Segments overlap, return distance between parallel segments
        return None,None,np.linalg.norm(((d0*d1)+a0)-b0)


    # Lines criss-cross: Calculate the projected closest points
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
            dot = np.dot(d2,(pA-b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > normB:
                dot = normB
            pB = b0 + (d2 * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > normB):
            dot = np.dot(d1,(pB-a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > normA:
                dot = normA
            pA = a0 + (d1 * dot)


    return pA,pB,np.linalg.norm(pA-pB)

'''
예제 코드 - clamp 설정 여부에 대하여

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# need function : closestDistanceBetweenLines

a1=np.array([0.000015,-0.000046,387.265137])
a0=np.array([0.000004,-0.000015,52.138947])
b0=np.array([39.939014,11.414734,-4.960205])
b1=np.array([-43.097198,11.414734,5.373566])

t0=closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=True)[0]
t1=closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=True)[1]

# closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=True) # in the area
# closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=False)

print('t0', type(t0), t0)
print('t1', type(t1), t1)


x1 = np.linspace(a1[0],a0[0],100)
y1 = np.linspace(a1[1],a0[1],100)
z1 = np.linspace(a1[2],a0[2],100)

x2 = np.linspace(b1[0],b0[0],100)
y2 = np.linspace(b1[1],b0[1],100)
z2 = np.linspace(b1[2],b0[2],100)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x1,y1,z1)
ax.plot(x2,y2,z2)

ax.scatter(t0[0], t0[1], t0[2])
ax.scatter(t1[0], t1[1], t1[2])

x3 = np.linspace(t1[0],t0[0],100)
y3 = np.linspace(t1[1],t0[1],100)
z3 = np.linspace(t1[2],t0[2],100)

ax.plot(x3,y3,z3)

plt.show()

'''
