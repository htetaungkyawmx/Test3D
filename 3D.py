import numpy as np
import matplotlib
# Screen မပြဘဲ file တိုက်ရိုက်သိမ်းမယ်
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_circles, make_blobs

print("=" * 60)
print("3D DIMENSIONS VISUALIZATION")
print("=" * 60)

# ============================================
# 1. 2D Visualizations
# ============================================
print("\n1. Creating 2D visualizations...")

fig = plt.figure(figsize=(20, 12))
fig.suptitle('2D DIMENSIONS VISUALIZATION', fontsize=16, y=0.95)

# 2D Scatter Plot
ax1 = plt.subplot(2, 3, 1)
x = np.random.randn(200)
y = np.random.randn(200)
scatter = ax1.scatter(x, y, c=np.sqrt(x**2 + y**2), cmap='plasma', alpha=0.6, s=30)
ax1.set_title('2D Scatter Plot (Color by Distance)', fontsize=12)
ax1.set_xlabel('X Dimension')
ax1.set_ylabel('Y Dimension')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Distance from origin')

# 2D Grid
ax2 = plt.subplot(2, 3, 2)
x_grid, y_grid = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
ax2.scatter(x_grid, y_grid, c='red', s=15, alpha=0.5)
ax2.set_title('2D Grid Points (20x20)', fontsize=12)
ax2.set_xlabel('X Dimension')
ax2.set_ylabel('Y Dimension')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# 2D Circles Dataset
ax3 = plt.subplot(2, 3, 3)
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5)
ax3.scatter(X_circles[y_circles==0, 0], X_circles[y_circles==0, 1], 
            c='blue', label='Class 0', alpha=0.6, s=30)
ax3.scatter(X_circles[y_circles==1, 0], X_circles[y_circles==1, 1], 
            c='red', label='Class 1', alpha=0.6, s=30)
ax3.set_title('2D Circles Dataset (Two Classes)', fontsize=12)
ax3.set_xlabel('X Dimension')
ax3.set_ylabel('Y Dimension')
ax3.legend()
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

# 2D Gaussian Distribution
ax4 = plt.subplot(2, 3, 4)
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]  # correlation
data = np.random.multivariate_normal(mean, cov, 1000)
hist = ax4.hist2d(data[:, 0], data[:, 1], bins=40, cmap='Blues')
ax4.set_title('2D Gaussian Distribution (Correlated)', fontsize=12)
ax4.set_xlabel('X Dimension')
ax4.set_ylabel('Y Dimension')
plt.colorbar(hist[3], ax=ax4, label='Frequency')

# 2D Shapes
ax5 = plt.subplot(2, 3, 5)
# Point (0D)
ax5.scatter(0, 0, c='green', s=200, marker='*', label='Point (0D)', zorder=5)
# Line (1D)
ax5.plot([-2, 2], [-2, 2], 'b-', linewidth=3, label='Line (1D)')
# Rectangle (2D)
rect = plt.Rectangle((-1.5, -1.5), 3, 3, fill=False, edgecolor='red', 
                     linewidth=2, label='Rectangle (2D)')
ax5.add_patch(rect)
# Circle (2D)
circle = plt.Circle((1.5, 1.5), 0.8, fill=False, edgecolor='purple', 
                    linewidth=2, label='Circle (2D)')
ax5.add_patch(circle)
ax5.set_xlim(-3, 3)
ax5.set_ylim(-3, 3)
ax5.set_title('2D Space with 0D, 1D, and 2D Objects', fontsize=12)
ax5.set_xlabel('X Dimension')
ax5.set_ylabel('Y Dimension')
ax5.legend(loc='upper left')
ax5.grid(True, alpha=0.3)
ax5.set_aspect('equal')

# 2D Contour Plot
ax6 = plt.subplot(2, 3, 6)
x_contour = np.linspace(-3, 3, 100)
y_contour = np.linspace(-3, 3, 100)
X_contour, Y_contour = np.meshgrid(x_contour, y_contour)
Z_contour = np.sin(X_contour) * np.cos(Y_contour)
contour = ax6.contourf(X_contour, Y_contour, Z_contour, levels=20, cmap='RdYlBu')
ax6.set_title('2D Contour Plot: sin(X) * cos(Y)', fontsize=12)
ax6.set_xlabel('X Dimension')
ax6.set_ylabel('Y Dimension')
plt.colorbar(contour, ax=ax6, label='Z value')
ax6.set_aspect('equal')

plt.tight_layout()
plt.savefig('2d_dimensions_complete.png', dpi=200, bbox_inches='tight')
print("✓ 2D visualization saved as '2d_dimensions_complete.png'")

# ============================================
# 2. 3D Visualizations
# ============================================
print("\n2. Creating 3D visualizations...")

fig = plt.figure(figsize=(20, 15))
fig.suptitle('3D DIMENSIONS VISUALIZATION', fontsize=16, y=0.95)

# 3D Scatter Plot
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
n_points = 200
x3d = np.random.randn(n_points) * 2
y3d = np.random.randn(n_points) * 2
z3d = np.random.randn(n_points) * 2
colors = np.sqrt(x3d**2 + y3d**2 + z3d**2)
scatter = ax1.scatter(x3d, y3d, z3d, c=colors, cmap='viridis', s=30, alpha=0.7)
ax1.set_title('3D Scatter Plot\n(200 Random Points)', fontsize=12)
ax1.set_xlabel('X Dimension')
ax1.set_ylabel('Y Dimension')
ax1.set_zlabel('Z Dimension')
plt.colorbar(scatter, ax=ax1, shrink=0.5, label='Distance from origin')

# 3D Surface
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
X_surf, Y_surf = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
Z_surf = np.sin(np.sqrt(X_surf**2 + Y_surf**2))
surf = ax2.plot_surface(X_surf, Y_surf, Z_surf, cmap='coolwarm', alpha=0.9)
ax2.set_title('3D Surface: Z = sin(√(X² + Y²))', fontsize=12)
ax2.set_xlabel('X Dimension')
ax2.set_ylabel('Y Dimension')
ax2.set_zlabel('Z Dimension')
plt.colorbar(surf, ax=ax2, shrink=0.5, label='Z value')

# 3D Wireframe + Scatter
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
X_wire, Y_wire = np.meshgrid(np.linspace(-3, 3, 15), np.linspace(-3, 3, 15))
Z_wire = X_wire**2 - Y_wire**2
ax3.plot_wireframe(X_wire, Y_wire, Z_wire, color='gray', alpha=0.5, label='Wireframe')
# Add random points on surface
x_rand = np.random.uniform(-3, 3, 50)
y_rand = np.random.uniform(-3, 3, 50)
z_rand = x_rand**2 - y_rand**2
ax3.scatter(x_rand, y_rand, z_rand, c='red', s=20, alpha=0.8, label='Points on surface')
ax3.set_title('3D Wireframe: Z = X² - Y²\nwith Random Points', fontsize=12)
ax3.set_xlabel('X Dimension')
ax3.set_ylabel('Y Dimension')
ax3.set_zlabel('Z Dimension')
ax3.legend()

# 3D Sphere
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 40)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax4.plot_surface(x_sphere, y_sphere, z_sphere, cmap='plasma', alpha=0.8)
ax4.set_title('3D Sphere\n(Parametric Surface)', fontsize=12)
ax4.set_xlabel('X Dimension')
ax4.set_ylabel('Y Dimension')
ax4.set_zlabel('Z Dimension')

# 3D Bars
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
x_bar = np.arange(6)
y_bar = np.arange(6)
x_bar, y_bar = np.meshgrid(x_bar, y_bar)
x_bar = x_bar.flatten()
y_bar = y_bar.flatten()
z_bar = np.zeros_like(x_bar)
dz = np.random.randint(1, 8, size=len(x_bar))
colors = plt.cm.rainbow(dz / 8)
ax5.bar3d(x_bar, y_bar, z_bar, 0.7, 0.7, dz, color=colors, alpha=0.9)
ax5.set_title('3D Bar Chart\n(6x6 Grid)', fontsize=12)
ax5.set_xlabel('X Dimension')
ax5.set_ylabel('Y Dimension')
ax5.set_zlabel('Z Dimension (Height)')

# 3D Spiral and Clusters
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
# Spiral
t = np.linspace(0, 6*np.pi, 200)
x_spiral = np.cos(t) * t/3
y_spiral = np.sin(t) * t/3
z_spiral = t
ax6.plot(x_spiral, y_spiral, z_spiral, 'b-', linewidth=2, label='Spiral')
# Clusters
X_clusters, y_clusters = make_blobs(n_samples=150, n_features=3, centers=3, 
                                     cluster_std=1.0, random_state=42)
ax6.scatter(X_clusters[y_clusters==0, 0], X_clusters[y_clusters==0, 1], 
            X_clusters[y_clusters==0, 2], c='red', s=30, alpha=0.6, label='Cluster 1')
ax6.scatter(X_clusters[y_clusters==1, 0], X_clusters[y_clusters==1, 1], 
            X_clusters[y_clusters==1, 2], c='green', s=30, alpha=0.6, label='Cluster 2')
ax6.scatter(X_clusters[y_clusters==2, 0], X_clusters[y_clusters==2, 1], 
            X_clusters[y_clusters==2, 2], c='blue', s=30, alpha=0.6, label='Cluster 3')
ax6.set_title('3D Spiral + Clusters', fontsize=12)
ax6.set_xlabel('X Dimension')
ax6.set_ylabel('Y Dimension')
ax6.set_zlabel('Z Dimension')
ax6.legend()

plt.tight_layout()
plt.savefig('3d_dimensions_complete.png', dpi=200, bbox_inches='tight')
print("✓ 3D visualization saved as '3d_dimensions_complete.png'")

# ============================================
# 3. 2D to 3D Transformation
# ============================================
print("\n3. Creating 2D to 3D transformation demo...")

fig = plt.figure(figsize=(18, 6))
fig.suptitle('2D TO 3D TRANSFORMATION', fontsize=16, y=0.98)

# 2D Original Shapes
ax1 = plt.subplot(1, 3, 1)
# Square
square = plt.Rectangle((-2, -2), 4, 4, fill=False, edgecolor='blue', linewidth=2, label='Square')
ax1.add_patch(square)
# Circle
circle = plt.Circle((2, 2), 1.5, fill=False, edgecolor='red', linewidth=2, label='Circle')
ax1.add_patch(circle)
# Triangle
triangle = plt.Polygon([[-1, -1], [1, -1], [0, 1]], fill=False, edgecolor='green', linewidth=2, label='Triangle')
ax1.add_patch(triangle)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)
ax1.set_title('Original 2D Shapes', fontsize=12)
ax1.set_xlabel('X Dimension')
ax1.set_ylabel('Y Dimension')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Extrusion to 3D
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
# Extrude square
z_extrude = np.linspace(-2, 2, 10)
for z in z_extrude:
    ax2.plot([-2, 2, 2, -2, -2], [-2, -2, 2, 2, -2], z, 'b-', alpha=0.5)
# Extrude circle
t_circle = np.linspace(0, 2*np.pi, 30)
for z in np.linspace(-2, 2, 5):
    x_circ = 2 + 1.5 * np.cos(t_circle)
    y_circ = 2 + 1.5 * np.sin(t_circle)
    ax2.plot(x_circ, y_circ, z, 'r-', alpha=0.5)
ax2.set_title('2D Shapes Extruded to 3D', fontsize=12)
ax2.set_xlabel('X Dimension')
ax2.set_ylabel('Y Dimension')
ax2.set_zlabel('Z Dimension')

# Rotation in 3D
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
# Rotate square in 3D
angles = np.linspace(0, 2*np.pi, 12)
square_x = np.array([-2, 2, 2, -2, -2])
square_y = np.array([-2, -2, 2, 2, -2])
for i, angle in enumerate(angles):
    # Rotate around Y axis
    x_rot = square_x * np.cos(angle) - np.zeros_like(square_x) * np.sin(angle)
    z_rot = square_x * np.sin(angle) + np.zeros_like(square_x) * np.cos(angle)
    ax3.plot(x_rot, square_y, z_rot, color=plt.cm.rainbow(i/len(angles)), alpha=0.6)
ax3.set_title('2D Shapes Rotated in 3D Space', fontsize=12)
ax3.set_xlabel('X Dimension')
ax3.set_ylabel('Y Dimension')
ax3.set_zlabel('Z Dimension')

plt.tight_layout()
plt.savefig('2d_to_3d_transformation_complete.png', dpi=200, bbox_inches='tight')
print("✓ 2D to 3D transformation saved as '2d_to_3d_transformation_complete.png'")

# ============================================
# 4. Mathematical Functions in 3D
# ============================================
print("\n4. Creating mathematical functions in 3D...")

fig = plt.figure(figsize=(20, 10))
fig.suptitle('MATHEMATICAL FUNCTIONS IN 3D', fontsize=16, y=0.95)

# Sine-Cosine Surface
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
X, Y = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
Z1 = np.sin(X) * np.cos(Y)
surf1 = ax1.plot_surface(X, Y, Z1, cmap='RdYlBu', alpha=0.9)
ax1.set_title('Z = sin(X) * cos(Y)', fontsize=12)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
plt.colorbar(surf1, ax=ax1, shrink=0.5)

# Gaussian Function
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
Z2 = np.exp(-(X**2 + Y**2)/3)
surf2 = ax2.plot_surface(X, Y, Z2, cmap='viridis', alpha=0.9)
ax2.set_title('Z = exp(-(X² + Y²)/3) (Gaussian)', fontsize=12)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
plt.colorbar(surf2, ax=ax2, shrink=0.5)

# Ripple Function
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
R = np.sqrt(X**2 + Y**2)
Z3 = np.sinc(R)  # sin(πR)/(πR)
surf3 = ax3.plot_surface(X, Y, Z3, cmap='plasma', alpha=0.9)
ax3.set_title('Z = sinc(√(X² + Y²)) (Ripple)', fontsize=12)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
plt.colorbar(surf3, ax=ax3, shrink=0.5)

# Saddle Point
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
Z4 = X**2 - Y**2
surf4 = ax4.plot_surface(X, Y, Z4, cmap='coolwarm', alpha=0.9)
ax4.set_title('Z = X² - Y² (Saddle Point)', fontsize=12)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
plt.colorbar(surf4, ax=ax4, shrink=0.5)

# Sombrero Function
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
Z5 = np.sinc(R) * 3
surf5 = ax5.plot_surface(X, Y, Z5, cmap='hot', alpha=0.9)
ax5.set_title('Z = 3 * sinc(√(X² + Y²)) (Sombrero)', fontsize=12)
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Z')
plt.colorbar(surf5, ax=ax5, shrink=0.5)

# Wave Function
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
Z6 = np.sin(X) + np.cos(Y)
surf6 = ax6.plot_surface(X, Y, Z6, cmap='rainbow', alpha=0.9)
ax6.set_title('Z = sin(X) + cos(Y)', fontsize=12)
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.set_zlabel('Z')
plt.colorbar(surf6, ax=ax6, shrink=0.5)

plt.tight_layout()
plt.savefig('mathematical_3d_functions.png', dpi=200, bbox_inches='tight')
print("✓ Mathematical functions saved as 'mathematical_3d_functions.png'")

print("\n" + "=" * 60)
print("✅ ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\n📁 Files created:")
print("  1. 2d_dimensions_complete.png")
print("  2. 3d_dimensions_complete.png")  
print("  3. 2d_to_3d_transformation_complete.png")
print("  4. mathematical_3d_functions.png")
print("\n📍 These files are saved in your current folder:")
print("   /Users/macbookpro/Desktop/AI/Test3D/")
print("=" * 60)