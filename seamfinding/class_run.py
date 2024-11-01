from class_ransac import RANSAC_3D_Class
import numpy as np
import matplotlib.pyplot as plt


def read_line(file_path):
    pos = []
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            if len(values) == 3:
                pos.append(values)
    
    points = np.array(pos)

    return points


lts_file = "LTS_gather.txt"
pos = read_line(lts_file)


iter_max = 100
offset = 1

plane = RANSAC_3D_Class(iter_max, offset)

x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

plane.setData(x, y, z)

plane.processRANSAC()

get_plane = plane.getRANSAC()
# print(type(get_plane))
# print(get_plane)


X = np.arange(-15, 15, 0.5)
Y = np.arange(-30, 0, 0.5)
X, Y = np.meshgrid(X, Y)
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, marker='x', s=2)

ax.plot_surface(X, Y, get_plane, rstride=1, cstride=1, alpha=0.5, color='lightcyan')

# 축 범위 설정 - x, y, z 다 가능
ax.axes.set_zlim3d(bottom = -25, top = 5)


# Label
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

