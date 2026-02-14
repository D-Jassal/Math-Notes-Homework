import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================
# DATA
# ============================================
x = np.array([11.2, 10.7, 9.9, 10.4, 12.1, 11.5, 10.9, 9.6, 10.1, 11.0])
n = len(x)
xbar = np.mean(x)

print("="*60)
print("PART I: LIKELIHOOD PLOT")
print("="*60)
print(f"Data: {x}")
print(f"Sample size (n): {n}")
print(f"Sample mean (MLE): {xbar:.2f}")
print("="*60)

# ============================================
# FOR PLOTTING: We need a value for sigma^2
# The likelihood shape depends on sigma^2, but the MLE (peak) does not
# We'll use sigma^2 = 1 for visualization
# ============================================
sigma2 = 0.58488889  # Known variance - using 1 for plotting
sigma = np.sqrt(sigma2)

print(f"\nNOTE: For visualization, we use σ² = {sigma2}")
print("The MLE (peak) is at μ = 10.74 regardless of σ²")
print("The spread of the likelihood depends on σ²/n\n")

# ============================================
# CREATE LIKELIHOOD FUNCTION
# ============================================

# Range of mu values for plotting (centered around MLE)
mu_range = np.linspace(xbar - 1.5, xbar + 1.5, 1000)

# Calculate likelihood (proportional to Normal PDF)
# L(μ|x) ∝ exp(-(n/(2σ²)) * (μ - xbar)²)
likelihood = stats.norm.pdf(mu_range, xbar, sigma/np.sqrt(n))

# For better visualization, we can also plot the log-likelihood
log_likelihood = -0.5 * ((mu_range - xbar)**2) / (sigma2/n)  # up to constant

# ============================================
# CREATE THE PLOT
# ============================================

# Create figure with two subplots (optional - can use just one)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Likelihood
ax1.plot(mu_range, likelihood, 'b-', linewidth=2.5)
ax1.axvline(x=xbar, color='red', linestyle='--', linewidth=2, label=f'MLE = {xbar:.2f}')
ax1.set_xlabel('μ', fontsize=12)
ax1.set_ylabel('L(μ | x)', fontsize=12)
ax1.set_title('Likelihood Function L(μ | x)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([xbar - 1.5, xbar + 1.5])

# Add text box with information
textstr = f'n = {n}\n$\\bar{{x}}$ = {xbar:.2f}\n$\\sigma^2=s^2$ = {sigma2}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Plot 2: Log-Likelihood (optional - shows the quadratic form)
ax2.plot(mu_range, log_likelihood, 'g-', linewidth=2.5)
ax2.axvline(x=xbar, color='red', linestyle='--', linewidth=2, label=f'MLE = {xbar:.2f}')
ax2.set_xlabel('μ', fontsize=12)
ax2.set_ylabel('log L(μ | x)', fontsize=12)
ax2.set_title('Log-Likelihood Function ℓ(μ | x)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([xbar - 1.5, xbar + 1.5])

plt.tight_layout()
plt.savefig('partI_likelihood_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# SINGLE PLOT VERSION (if you prefer just one)
# ============================================
plt.figure(figsize=(10, 6))
plt.plot(mu_range, likelihood, 'b-', linewidth=2.5)
plt.axvline(x=xbar, color='red', linestyle='--', linewidth=2, label=f'$\hat{{\mu}}_{{MLE}} = {xbar:.2f}$')
plt.fill_between(mu_range, 0, likelihood, alpha=0.2, color='blue')
plt.xlabel('μ', fontsize=14)
plt.ylabel('L(μ | x)', fontsize=14)
plt.title('Likelihood Function for Normal Mean', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim([xbar - 1.5, xbar + 1.5])

# Add annotation showing the maximum
plt.annotate(f'Maximum at μ = {xbar:.2f}', 
             xy=(xbar, np.max(likelihood)), 
             xytext=(xbar + 0.3, np.max(likelihood)*0.8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=11)

# Text box with data info
textstr = f'Data: n = {n}\nSample mean = {xbar:.2f}\nAssumed σ² = {sigma2}'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('partI_likelihood_single.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# PRINT ADDITIONAL INFORMATION
# ============================================
print("\n" + "="*60)
print("LIKELIHOOD FUNCTION DETAILS")
print("="*60)
print(f"The likelihood function is proportional to:")
print(f"L(μ | x) ∝ exp(-(n/(2σ²)) × (μ - {xbar:.2f})²)")
print(f"  where n = {n}, σ² = {sigma2} (assumed for plotting)")
print(f"\nThe MLE maximizes this function at μ = {xbar:.2f}")
print(f"\nThe curvature (2nd derivative) at the MLE is -n/σ² = -{n/sigma2:.2f}")
print(f"This relates to the variance of the MLE: Var(μ̂_MLE) = σ²/n = {sigma2/n:.4f}")
print("="*60)