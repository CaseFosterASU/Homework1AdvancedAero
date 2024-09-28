import numpy as np
import matplotlib.pyplot as plt

# Airfoil shape equation
def airfoil_shape(x, t):
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2834 * x**3 - 0.1036 * x**4)

# Freestream velocity
U_inf = 1.0  

# Discretize the airfoil surface using a number of panels with cosine spacing
def create_panels(N, t):
    # Cosine spacing for better resolution near the leading and trailing edges
    beta = np.linspace(0, np.pi, N)
    x = (1 - np.cos(beta)) / 2  # x-coordinates (non-dimensional)
    y = airfoil_shape(x, t)  # y-coordinates from the airfoil shape equation
    return x, y

# Source panel method calculation
def panel_method(x, y, U_inf):
    N = len(x) - 1  # Number of panels
    A = np.zeros((N, N))
    b = np.zeros(N)

    # Panel geometry
    for i in range(N):
        xi, yi = (x[i] + x[i + 1]) / 2, (y[i] + y[i + 1]) / 2  # Control points (panel centers)
        length = np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2)  # Panel length

        for j in range(N):
            if i != j:
                # Influence of source j on panel i
                dx = xi - (x[j] + x[j + 1]) / 2
                dy = yi - (y[j] + y[j + 1]) / 2
                A[i, j] = np.log(np.sqrt(dx ** 2 + dy ** 2)) / (2 * np.pi)
            else:
                A[i, j] = 0.5  # Self-influence term

        # RHS (freestream influence)
        b[i] = -U_inf * (y[i + 1] - y[i]) / length

    # Solve for source strengths
    source_strengths = np.linalg.solve(A, b)
    return source_strengths

# Calculate velocity at each control point
def velocity_distribution(source_strengths, x, y, U_inf):
    N = len(source_strengths)
    velocity = np.zeros(N)

    for i in range(N):
        vi = U_inf
        for j in range(N):
            if i != j:
                dx = x[i] - (x[j] + x[j + 1]) / 2
                dy = y[i] - (y[j] + y[j + 1]) / 2
                vi += source_strengths[j] * np.log(np.sqrt(dx ** 2 + dy ** 2)) / (2 * np.pi)
        velocity[i] = vi
    return velocity

# Pressure coefficient calculation
def pressure_coefficient(velocity, U_inf):
    return 1 - (velocity / U_inf) ** 2

# Streamline calculation over a grid of points
def compute_velocity_field(source_strengths, x, y, X, Y, U_inf):
    u = np.ones_like(X) * U_inf  # X-component of velocity (freestream)
    v = np.zeros_like(Y)  # Y-component of velocity

    for j in range(len(source_strengths)):
        # Influence of each source on the grid
        dx = X - (x[j] + x[j + 1]) / 2
        dy = Y - (y[j] + y[j + 1]) / 2
        u += source_strengths[j] * dx / (2 * np.pi * (dx ** 2 + dy ** 2))
        v += source_strengths[j] * dy / (2 * np.pi * (dx ** 2 + dy ** 2))
    
    return u, v

# Plot airfoil shape, streamlines, and pressure coefficient
def plot_airfoil_and_streamlines(x, y, Cp, u, v, X, Y):
    plt.figure(figsize=(12, 6))
    
    # Airfoil shape and pressure coefficient plot
    plt.subplot(121)
    plt.plot(x, y, label="Airfoil Shape")
    plt.plot(x, -y, label="Airfoil Shape (lower)", linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Airfoil Shape and Pressure Coefficient')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()

    # Pressure coefficient plot
    plt.plot(x[:-1], Cp, label="Pressure Coefficient")
    plt.gca().invert_yaxis()
    plt.legend()

    # Streamline plot
    plt.subplot(122)
    plt.plot(x, y, color='k', lw=2)
    plt.plot(x, -y, color='k', lw=2)  # Lower surface
    plt.streamplot(X, Y, u, v, density=2, linewidth=1, arrowsize=1)
    plt.title('Streamlines Around the Airfoil')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Main function for running the analysis
def main(N, airfoil_type):
    # Airfoil thickness parameter
    if airfoil_type == 'NACA0004':
        t = 0.04
    elif airfoil_type == 'NACA0012':
        t = 0.12
    elif airfoil_type == 'NACA0024':
        t = 0.24
    else:
        raise ValueError("Unsupported airfoil type.")
    
    # Create panels and source distribution
    x, y = create_panels(N, t)
    source_strengths = panel_method(x, y, U_inf)
    
    # Calculate velocity and pressure coefficient
    velocity = velocity_distribution(source_strengths, x, y, U_inf)
    Cp = pressure_coefficient(velocity, U_inf)
    
    # Create grid for streamlines with larger domain and finer resolution
    X, Y = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-1.5, 1.5, 300))
    
    # Compute velocity field
    u, v = compute_velocity_field(source_strengths, x, y, X, Y, U_inf)
    
    # Plot results: Airfoil shape, streamlines, and Cp
    plot_airfoil_and_streamlines(x, y, Cp, u, v, X, Y)

# Run the program for three airfoil types with 50 panels
if __name__ == "__main__":
    main(100, 'NACA0004')  # For NACA0004
    main(100, 'NACA0012')  # For NACA0012
    main(100, 'NACA0024')  # For NACA0024
