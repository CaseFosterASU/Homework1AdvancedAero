import numpy as np
import matplotlib.pyplot as plt

# Define theta for 0 to 180 degrees
theta_half = np.linspace(0, np.pi, 500)  # 0 to 180 degrees in radians


def calculate_cp(gamma_ratio, theta_half):
    v_theta = -2 * np.sin(theta_half) - gamma_ratio  
    Cp = 1 - (v_theta / U_inf) ** 2
    return Cp

# Constants
U_inf = 1  # Freestream velocity
gamma_ratios = [0, 0.5, 1, 2]  # Different circulation values

# Plot for different circulation values, restricting to 0 to 180 degrees
plt.figure(figsize=(7, 6))

# Loop through the circulation values and plot for 0 to 180 degrees
for gamma_ratio in gamma_ratios:
    Cp_half = calculate_cp(gamma_ratio, theta_half)
    plt.plot(theta_half * 180/np.pi, Cp_half, label=f'Γ/(2πRU∞) = {gamma_ratio}')

# Set labels and title
plt.title(r"$C_p$ Distribution", fontsize=14)
plt.xlabel(r"$\theta$ in degrees", fontsize=12)
plt.ylabel(r"$C_p$", fontsize=12)
plt.legend(fontsize=10)

# Set limits to match the 0 to 180 degrees range and y-axis to show Cp reaching -14
plt.xlim([0, 180])
plt.ylim([-15, 4])  # Adjust the y-axis to show peaks near -14
plt.grid(True)

# Show the plot
plt.show()
