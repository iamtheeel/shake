###
# Footfall
# Joshua Mehlman
# MIC Lab
# Spring, 2025
###
# Make a 3d Plot
###

import numpy as np
from matplotlib import pyplot, cm

#Copy pasted the data from the spred sheet to the chatbot
# Define the matrix with columns: [Learning Rate, Weight Decay, Val Acc (RMS Error)]
data_matrix = np.array([
    [0.001,     0.00001,    0.0928417068413714],
    [0.001,     0.000001,   0.092841983217382],
    [0.001,     0.0000001,  0.0928418017372496],
    [0.0001,    0.00001,    0.0751446272280765],
    [0.0001,    0.000001,   0.0817933315791413],
    [0.0001,    0.0000001,  0.0636231826290749],
    [0.00001,   0.00001,    0.0537739462293444],
    [0.00001,   0.000001,   0.0534209607236096],
    [0.00001,   0.0000001,  0.0539255499633141],
    [0.00001,   0.001,      0.0529330475499118],
    [0.00001,   0.0001,     0.0528842205049463],
    [0.00001,   0.00001,    0.0537739462293444],
    [0.000001,  0.001,      0.0871188232543952],
    [0.000001,  0.0001,     0.0881331568394255],
    [0.000001,  0.00001,    0.0880636348045407],
    [0.0000001, 0.001,      0.0928555151127907],
    [0.0000001, 0.0001,     0.092854744836081],
    [0.0000001, 0.00001,    0.0928563189516]
])

X = data_matrix[:,0] # Learning Rate
Y = data_matrix[:,1] # Weight Decay
Z = data_matrix[:,2] # RMS
print(X)

print("Unique X values:", np.unique(X))
print("Contains 0.001:", 0.001 in X)

'''
# Create a structured grid for the mesh plot
grid_x, grid_y = np.meshgrid(
    np.logspace(np.log10(min(X)), np.log10(max(X)), 30),
    np.logspace(np.log10(min(Y)), np.log10(max(Y)), 30)
)

# Interpolate Z values for the grid
grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')

# Create 3D figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Wireframe (mesh) plot
ax.plot_wireframe(grid_x, grid_y, grid_z, color='black', linewidth=0.7)
'''

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X, Y, Z, zdir='z', c= 'red')
ax.scatter(X, Y, Z, c='red', marker='o', alpha=0.7, s=50)

# **Set axis limits**
ax.set_xlim([1e-6, 1e-3])  # Limit X (Learning Rate)
ax.set_ylim([1e-6, 1e-3])  # Limit Y (Weight Decay)
ax.set_zlim([0.05, 0.1])   # Limit Z (Validation Accuracy)
#ax.set_xlim([min(X) * 0.9, max(X) * 1.1])
#ax.set_ylim([min(Y) * 0.9, max(Y) * 1.1])
#ax.set_zlim([min(Z) * 0.9, max(Z) * 1.1])
#mask = X == 0.001
#ax.scatter(X[mask], Y[mask], Z[mask], c='blue', marker='o', s=200, label="X=0.001")
#ax.legend()

# Instead of filtering only X, get the full row where X == 0.001
#mask = data_matrix[:, 0] == 0.001  # Select rows where Learning Rate is 0.001
# Extract matching X, Y, Z values
#X_highlight = data_matrix[mask, 0]
#Y_highlight = data_matrix[mask, 1]
#Z_highlight = data_matrix[mask, 2]
#ax.scatter(X_highlight, Y_highlight, Z_highlight, c='blue', marker='o', s=200, label="X=0.001")

#ax.set_xticks([1e-6, 1e-5, 1e-4, 1e-3])  # Force correct tick locations
#ax.set_xticklabels(["1e-6", "1e-5", "1e-4", "1e-3"])  # Ensure proper display
ax.plot([0.001, 0.001], [min(Y), max(Y)], [min(Z), max(Z)], color='black', linestyle='dashed', linewidth=2, label="True X=0.001")
ax.legend()
#ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_zscale('log')

#Rotate the angle
#ax.view_init(elev=30, azim=45)   # Standard top-left perspective
ax.view_init(elev=10, azim=120)  # Low view, looking across
#ax.view_init(elev=60, azim=210)  # Higher, looking down

ax.set_title("RMS Acc for LR and WD. Batch Size: 128, Epochs: 200")
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Weight Decay")
ax.set_zlabel("Accuracy (RMS)")


pyplot.show()