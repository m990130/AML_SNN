import pickle
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

with open('name.p', 'rb') as f:
    wc = pickle.load(f)

comps = wc.get_components(12)
loss = wc.loss

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = loss
x = comps[:, 4]
y = comps[:, 9]
ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()
