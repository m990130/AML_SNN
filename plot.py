import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D

with open('weights<class \'models.CNN.Raw_CNN\'>_0.0.p', 'rb') as f:
    wc = pickle.load(f)

comps = wc.get_components(12)
loss = wc.loss

N = len(loss)
x = comps[:, 1]
y = comps[:, 9]
z = -np.array(loss)
t = -np.array(loss)

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be (numlines) x (points per line) x 2 (for x and y)
points = np.array([x, y, z]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
fig = plt.figure()
ax = fig.gca(projection='3d')

# Create a continuous norm to map from data points to colors
norm = (t-t.min())/(t.max()-t.min())
cmap=plt.get_cmap('plasma')
colors=[cmap(norm[ii]) for ii in range(N-1)]
lii_list = []

for ii in range(N-1):
    segii=segments[ii]
    lii,=ax.plot(segii[:,0],segii[:,1],segii[:,2],color=colors[ii],linewidth=2)
    #lii.set_dash_joinstyle('round')
    #lii.set_solid_joinstyle('round')
    lii.set_solid_capstyle('round')
    lii_list.append(lii)

cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=t.min(), vmax=t.max())

#cax, _ = mpl.colorbar.make_axes(ax)
cbaxes = fig.add_axes([0.025, 0.1, 0.03, 0.8])
cb1 = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                norm=norm,
                                )
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min()-0.01, y.max()+0.01)
#col = LineCollection(lii_list)
plt.show()