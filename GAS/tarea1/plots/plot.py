import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-30, 30, 10000)
y = np.linspace(-30, 30, 10000)
X, Y = np.meshgrid(x, y)

Z = 100*(Y-(X)**2)**2+(X-1)**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 3D axes

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis')

# Optional: add color bar
fig.colorbar(surf)

plt.show()
