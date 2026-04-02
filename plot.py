import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(0, np.pi, 100)
def X(r):
    return r*np.sin(r)*np.cos(r)
def Y(r):
    return r*np.sin(r)*np.sin(r)
def Z(r):
    return r*np.cos(r)
fig, ax = plt.figure('3d plot'), plt.axes(projection='3d')
ax.plot3D(X(r), Y(r), Z(r))
ax.scatter3D(X(0), Y(0), Z(0), color='red')
# a = 1
# ax.scatter3D(X(a), Y(a), Z(a), color='black')
for i in range(10):
    ax.scatter3D([X(0), X(0.3*i)], [Y(0), Y(0.3*i)], [Z(0), Z(0.3*i)], color='black')
    ax.plot3D([X(0), X(0.3*i)], [Y(0), Y(0.3*i)], [Z(0), Z(0.3*i)], color='black')
# ax.plot3D([X(0), X(a)], [Y(0), Y(a)], [Z(0), Z(a)], color='black')
ax.plot3D([-np.pi,np.pi], [0, 0], [0, 0], color='black')
ax.plot3D([0, 0], [-np.pi,np.pi], [0, 0], color='grey')
ax.plot3D([0, 0], [0, 0], [-np.pi,np.pi], color='grey')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('3dplot.png', dpi=300, bbox_inches='tight')
plt.show()