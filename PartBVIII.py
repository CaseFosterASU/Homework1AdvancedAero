import numpy as np
import matplotlib.pyplot as plt

# Constants
U_inf = 1  # Freestream velocity
R = 1  # Cylinder radius
gamma_ratios = [0, 0.5, 1, 2]  # Different circulation values (Γ / (2πRU∞))

# Define grid for X and Y (flow field)
x_range = np.linspace(-2, 2, 200)  # x range (increase resolution to 200 for smoother output)
y_range = np.linspace(-2, 2, 200)  # y range
X, Y = np.meshgrid(x_range, y_range)  # Create grid

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 layout for 4 subplots

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Loop through the different circulation values
for i, gamma_ratio in enumerate(gamma_ratios):
    Gamma = gamma_ratio * 2 * np.pi * R * U_inf  # Compute circulation for this case
    
    # Convert Cartesian coordinates (X, Y) to Polar coordinates (r, theta)
    theta = np.arctan2(Y, X)
    r = np.sqrt(X**2 + Y**2)
    
    # Calculate velocity components in polar coordinates
    v_r = U_inf * np.cos(theta) * (1 - (R**2 / r**2))  # Radial component of velocity
    v_theta = -U_inf * np.sin(theta) * (1 + (R**2 / r**2)) + Gamma / (2 * np.pi * r)  # Tangential velocity with circulation
    
    # Mask the region inside the cylinder (r <= R)
    v_r[r <= R] = np.nan
    v_theta[r <= R] = np.nan
    
    # Convert polar velocities to Cartesian velocities
    U = v_r * np.cos(theta) - v_theta * np.sin(theta)  # x-component of velocity
    V = v_r * np.sin(theta) + v_theta * np.cos(theta)  # y-component of velocity
    
    # Plot streamlines in the corresponding subplot
    ax = axes[i]
    ax.streamplot(X, Y, U, V, density=2, linewidth=1)
    circle = plt.Circle((0, 0), R, color='b', fill=False, linewidth=2)  # Plot the cylinder boundary
    ax.add_artist(circle)
    ax.set_title(f'Γ/(2πRU∞) = {gamma_ratio}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
