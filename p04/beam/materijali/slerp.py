import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

def calc_angle(a, b):
    inner = np.inner(v0, v1)
    norms = np.linalg.norm(v0) * np.linalg.norm(v1)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    deg = np.rad2deg(rad)
    print('angle: ', deg)
    return rad

ax = plt.axes()
v0 = np.array([1, 0])
v1 = np.array([0, 1])

ax.arrow(0.0, 0.0, v0[0], v0[1], head_width=0.1, head_length=0.2, fc='lightblue', ec='black')
ax.arrow(0.0, 0.0, v1[0], v1[1], head_width=0.1, head_length=0.2, fc='lightblue', ec='black')
    
ts = np.linspace(0, 1, 8)
for t in ts[1:-1]:
    v_lerp = (1-t)*v0 + t*v1

    theta = calc_angle(v0, v1)
    v_slerp = np.sin((1-t)*theta)/np.sin(theta)*v0+np.sin(t*theta)/np.sin(theta)*v1  
    ax.arrow(0.0, 0.0, v_lerp[0], v_lerp[1], head_width=0.1, head_length=0.2, fc='lightgreen', ec='black')
    ax.arrow(0.0, 0.0, v_slerp[0], v_slerp[1], head_width=0.1, head_length=0.2, fc='salmon', ec='black')

plt.grid()

plt.xlim(-0.5,1.5)
plt.ylim(-0.5,1.5)
plt.show()
