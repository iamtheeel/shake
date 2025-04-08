import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# === Load image ===
dir = "/Volumes/Data/thesis/publish/2_gradShowCase_2025/images"
#img = "IMG_7573.png"
img = "cmorl10-0.8_preFilt_trial-0_ch.jpg"
img = Image.open(f"{dir}/{img}").convert("RGB")

# === Resize with correct aspect ===
max_dim = 200
w, h = img.size
scale = min(max_dim / w, max_dim / h)
img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
img_np = np.array(img)

# === Extract channels ===
r, g, b = img_np[..., 0], img_np[..., 1], img_np[..., 2]

# === Meshgrid ===
H, W = r.shape
X, Y = np.meshgrid(np.arange(W), np.arange(H))
Y = H - Y  # Flip for image orientation

# === Colorization ===
def build_facecolors(channel, color_mask):
    norm = channel / 255.0
    rgb = np.zeros((H, W, 3))
    for i in range(3):
        rgb[..., i] = norm * color_mask[i]
    return rgb

# === Offsets (X, Y, Z) with overlap ===
slide_x = - 1.01
slide_y = 0.01
slide_z = 0.01

offSet_x = int(W * slide_x)
offSet_y = int(H * slide_y)
offSet_z = int(W * slide_z)

x_offsets = [i * offSet_x for i in range(3)]
y_offsets = [i * offSet_y for i in range(3)]
z_offsets = [i * offSet_z for i in range(3)]

# === Plotting ===
fig = plt.figure(figsize=((W + 2 * abs(offSet_x)) / 20, (H + 2 * abs(offSet_y)) / 20))
ax = fig.add_subplot(111, projection="3d")

# Plot each channel with its offsets
for channel, color, z_off, y_off, x_off in zip(
    [b, g, r],            # red now on top
    [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        z_offsets, y_offsets, x_offsets):
        #z_offsets, y_offsets, x_offsets):

    facecolors = build_facecolors(channel, color)
    Z = np.full_like(channel, z_off)
    X_offset = X + x_off
    Y_offset = Y + y_off
    ax.plot_surface(X_offset, Y_offset, Z,
                    rstride=1, cstride=1,
                    facecolors=facecolors,
                    shade=False, antialiased=False)

# === Fix aspect and axis limits ===
# Compute maximum extents dynamically
x_total = W + abs(min(x_offsets)) + abs(max(x_offsets))
y_total = H + abs(min(y_offsets)) + abs(max(y_offsets))
z_total = offSet_z * len(z_offsets)

ax.set_xlim(min(x_offsets), W + max(x_offsets))
ax.set_ylim(0, y_total)
ax.set_zlim(0, z_total)

ax.set_box_aspect((x_total, y_total, z_total))
#fig.set_size_inches(x_total / 20, y_total / 20)
fig.set_size_inches(7, 7)  # or 5, 5 — whatever looks good

# === Final view settings ===
ax.view_init(elev=90, azim=-90)
#ax.view_init(elev=105, azim=-95)
ax.axis("off")
plt.tight_layout()
plt.show()