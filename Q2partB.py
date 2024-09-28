import numpy as np
import matplotlib.pyplot as plt

# S1012 airfoil data
S1012_DATA = """
     1.00000   0.00000
     0.99640   0.00018
     0.98580   0.00094
     0.96872   0.00264
     0.94571   0.00536
     0.91728   0.00912
     0.88401   0.01398
     0.84664   0.01993
     0.80601   0.02677
     0.76300   0.03416
     0.71846   0.04139
     0.67266   0.04740
     0.62512   0.05182
     0.57587   0.05510
     0.52548   0.05748
     0.47451   0.05907
     0.42356   0.05994
     0.37324   0.06011
     0.32414   0.05961
     0.27682   0.05846
     0.23184   0.05667
     0.18974   0.05425
     0.15105   0.05120
     0.11622   0.04747
     0.08566   0.04299
     0.05966   0.03766
     0.03835   0.03142
     0.02177   0.02428
     0.00984   0.01640
     0.00251   0.00810
     0.00000   0.00000
     0.00251  -0.00810
     0.00984  -0.01640
     0.02177  -0.02428
     0.03835  -0.03142
     0.05966  -0.03766
     0.08566  -0.04299
     0.11622  -0.04747
     0.15105  -0.05120
     0.18974  -0.05425
     0.23184  -0.05667
     0.27682  -0.05846
     0.32414  -0.05961
     0.37324  -0.06011
     0.42356  -0.05994
     0.47451  -0.05907
     0.52548  -0.05748
     0.57587  -0.05510
     0.62512  -0.05182
     0.67266  -0.04740
     0.71846  -0.04139
     0.76300  -0.03416
     0.80601  -0.02677
     0.84664  -0.01993
     0.88401  -0.01398
     0.91728  -0.00912
     0.94571  -0.00536
     0.96872  -0.00264
     0.98580  -0.00094
     0.99640  -0.00018
     1.00000   0.00000
"""

def load_airfoil_data():
    data = np.array([list(map(float, line.split())) for line in S1012_DATA.strip().split('\n') if line.strip()])
    return data[:, 0], data[:, 1]

def calculate_source_strength(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    sigma = -2 * np.pi * np.sin(theta)
    return sigma, ds

# Updated function with epsilon to prevent division by zero
def calculate_induced_velocity(x, y, sigma, ds, x_field, y_field):
    u = np.zeros_like(x_field)
    v = np.zeros_like(y_field)
    epsilon = 1e-6  # Small value to prevent division by zero
    for i in range(len(sigma)):
        r = np.sqrt((x_field - x[i])**2 + (y_field - y[i])**2)
        r = np.maximum(r, epsilon)  # Ensure r is never smaller than epsilon
        u += sigma[i] * ds[i] * (x_field - x[i]) / (2 * np.pi * r**2)
        v += sigma[i] * ds[i] * (y_field - y[i]) / (2 * np.pi * r**2)
    return u, v

def calculate_pressure_coefficient(u, v, U_inf):
    V = np.sqrt(u**2 + v**2)
    return 1 - (V / U_inf)**2

def main():
    # Load airfoil data
    x, y = load_airfoil_data()

    # Calculate source strength
    sigma, ds = calculate_source_strength(x, y)

    # Create a grid for streamline calculation
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 0.5
    nx, ny = 100, 100
    x_field, y_field = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))

    # Calculate induced velocity
    U_inf = 1.0
    u_induced, v_induced = calculate_induced_velocity(x, y, sigma, ds, x_field, y_field)
    u_total = U_inf + u_induced
    v_total = v_induced

    # Mask the area inside the airfoil to remove streamlines within it
    for i in range(len(x)):
        mask = (x_field - x[i])**2 + (y_field - y[i])**2 < 1e-4  # Small threshold around the airfoil
        u_total[mask] = np.nan
        v_total[mask] = np.nan

    # Calculate pressure coefficient on the airfoil surface
    u_surface, v_surface = calculate_induced_velocity(x, y, sigma, ds, x, y)
    Cp = calculate_pressure_coefficient(U_inf + u_surface, v_surface, U_inf)

    # Check if the Cp values are being calculated properly
    print("Cp values: ", Cp)

    # Plot streamlines
    plt.figure(figsize=(12, 8))
    plt.streamplot(x_field, y_field, u_total, v_total, density=1.5, color='lightblue')
    plt.plot(x, y, 'k-', linewidth=2)  # Airfoil outline
    plt.title('Streamlines around S1012 Airfoil')
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # Plot pressure coefficient
    plt.figure(figsize=(12, 6))
    plt.plot(x, Cp, 'bo-', label="Cp")
    plt.title('Pressure Coefficient on S1012 Airfoil Surface')
    plt.xlabel('x/c')
    plt.ylabel('Cp')
    plt.grid(True)
    plt.gca().invert_yaxis()  # Invert y-axis to show negative Cp values at the top
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
