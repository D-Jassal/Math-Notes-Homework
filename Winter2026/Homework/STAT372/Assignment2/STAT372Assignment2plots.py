import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.array([11.2, 10.7, 9.9, 10.4, 12.1, 11.5, 10.9, 9.6, 10.1, 11.0])
n = len(x)
xbar = np.mean(x)
print(f"Sample mean (MLE): {xbar:.2f}")

sigma2 = 0.58488889  
sigma = np.sqrt(sigma2)

print("\n=== PART II(A): Concentrated Prior ===")
mu0_A = 9
tau2_A = 0.25
tau_A = np.sqrt(tau2_A)

prior_prec_A = 1/tau2_A
data_prec = n/sigma2
post_mean_A = (prior_prec_A * mu0_A + data_prec * xbar) / (prior_prec_A + data_prec)
post_var_A = 1/(prior_prec_A + data_prec)
post_sd_A = np.sqrt(post_var_A)

print(f"Prior: N({mu0_A}, {tau2_A})")
print(f"Posterior: N({post_mean_A:.2f}, {post_var_A:.4f})")
print(f"Prior weight: {prior_prec_A/(prior_prec_A + data_prec):.3f}")
print(f"Data weight: {data_prec/(prior_prec_A + data_prec):.3f}")

mu_seq_A = np.linspace(5, 15, 1000)

prior_A = stats.norm.pdf(mu_seq_A, mu0_A, tau_A)
likelihood = stats.norm.pdf(mu_seq_A, xbar, sigma/np.sqrt(n))
posterior_A = stats.norm.pdf(mu_seq_A, post_mean_A, post_sd_A)

likelihood_rescaled_A = likelihood * np.max(prior_A) / np.max(likelihood)

fig_A, ax_A = plt.subplots(figsize=(10, 6))
ax_A.plot(mu_seq_A, prior_A, 'b-', linewidth=2, label='Prior')
ax_A.plot(mu_seq_A, likelihood_rescaled_A, 'g-', linewidth=2, label='Likelihood (rescaled)')
ax_A.plot(mu_seq_A, posterior_A, 'r-', linewidth=2, label='Posterior')
ax_A.axvline(x=xbar, color='gray', linestyle='--', linewidth=1.5, label=f'MLE = {xbar:.2f}')
ax_A.axvline(x=post_mean_A, color='red', linestyle='--', linewidth=1.5, label=f'Posterior mean = {post_mean_A:.2f}')
ax_A.set_xlabel('μ', fontsize=12)
ax_A.set_ylabel('Density', fontsize=12)
ax_A.set_title('Case II(A): Concentrated Prior N(9, 0.25)', fontsize=14)
ax_A.legend(loc='upper right')
ax_A.grid(True, alpha=0.3)
ax_A.set_xlim([5, 15])
plt.tight_layout()
plt.savefig('case_A_python.png', dpi=300)
plt.show()

print("\n=== PART II(B): Diffuse Prior ===")
mu0_B = 9
tau2_B = 25
tau_B = np.sqrt(tau2_B)

prior_prec_B = 1/tau2_B
post_mean_B = (prior_prec_B * mu0_B + data_prec * xbar) / (prior_prec_B + data_prec)
post_var_B = 1/(prior_prec_B + data_prec)
post_sd_B = np.sqrt(post_var_B)

print(f"Prior: N({mu0_B}, {tau2_B})")
print(f"Posterior: N({post_mean_B:.2f}, {post_var_B:.4f})")
print(f"Prior weight: {prior_prec_B/(prior_prec_B + data_prec):.4f}")
print(f"Data weight: {data_prec/(prior_prec_B + data_prec):.4f}")

mu_seq_B = np.linspace(0, 20, 1000)

prior_B = stats.norm.pdf(mu_seq_B, mu0_B, tau_B)
posterior_B = stats.norm.pdf(mu_seq_B, post_mean_B, post_sd_B)

likelihood_rescaled_B = likelihood * np.max(prior_B) / np.max(likelihood)

fig_B, ax_B = plt.subplots(figsize=(10, 6))
ax_B.plot(mu_seq_B, prior_B, 'b-', linewidth=2, label='Prior')
ax_B.plot(mu_seq_B, likelihood_rescaled_B, 'g-', linewidth=2, label='Likelihood (rescaled)')
ax_B.plot(mu_seq_B, posterior_B, 'r-', linewidth=2, label='Posterior')
ax_B.axvline(x=xbar, color='gray', linestyle='--', linewidth=1.5, label=f'MLE = {xbar:.2f}')
ax_B.axvline(x=post_mean_B, color='red', linestyle='--', linewidth=1.5, label=f'Posterior mean = {post_mean_B:.2f}')
ax_B.set_xlabel('μ', fontsize=12)
ax_B.set_ylabel('Density', fontsize=12)
ax_B.set_title('Case II(B): Diffuse Prior N(9, 25)', fontsize=14)
ax_B.legend(loc='upper right')
ax_B.grid(True, alpha=0.3)
ax_B.set_xlim([0, 20])
plt.tight_layout()
plt.savefig('case_B_python.png', dpi=300)
plt.show()

print("\n=== PART II(C): Flat Prior ===")
post_mean_C = xbar
post_sd_C = sigma / np.sqrt(n)
post_var_C = post_sd_C**2

print(f"Posterior: N({post_mean_C:.2f}, {post_var_C:.4f})")

mu_seq_C = np.linspace(5, 15, 1000)

posterior_C = stats.norm.pdf(mu_seq_C, post_mean_C, post_sd_C)

flat_prior = np.ones_like(mu_seq_C) * 0.05  

likelihood_rescaled_C = stats.norm.pdf(mu_seq_C, xbar, post_sd_C)

fig_C, ax_C = plt.subplots(figsize=(10, 6))
ax_C.plot(mu_seq_C, flat_prior, 'b-', linewidth=2, label='Prior (flat, scaled for visualization)')
ax_C.plot(mu_seq_C, likelihood_rescaled_C, 'g-', linewidth=2, label='Likelihood')
ax_C.plot(mu_seq_C, posterior_C, 'r--', linewidth=2, label='Posterior')
ax_C.axvline(x=xbar, color='gray', linestyle='--', linewidth=1.5, label=f'MLE/Posterior mean = {xbar:.2f}')
ax_C.set_xlabel('μ', fontsize=12)
ax_C.set_ylabel('Density', fontsize=12)
ax_C.set_title('Case II(C): Flat Prior (Non-informative)', fontsize=14)
ax_C.legend(loc='upper right')
ax_C.grid(True, alpha=0.3)
ax_C.set_xlim([5, 15])
plt.tight_layout()
plt.savefig('case_C_python.png', dpi=300)
plt.show()

print("\n" + "="*50)
print("SUMMARY OF RESULTS (with σ² = 1)")
print("="*50)
print(f"{'Case':<10} {'Prior':<20} {'Posterior Mean':<15} {'Posterior Var':<15} {'Prior Weight':<12}")
print("-"*70)
print(f"{'A':<10} {'N(9, 0.25)':<20} {post_mean_A:<15.2f} {post_var_A:<15.4f} {prior_prec_A/(prior_prec_A+data_prec):<12.3f}")
print(f"{'B':<10} {'N(9, 25)':<20} {post_mean_B:<15.2f} {post_var_B:<15.4f} {prior_prec_B/(prior_prec_B+data_prec):<12.4f}")
print(f"{'C':<10} {'Flat':<20} {post_mean_C:<15.2f} {post_var_C:<15.4f} {'0':<12}")
print(f"{'MLE':<10} {'—':<20} {xbar:<15.2f} {sigma2/n:<15.4f} {'—':<12}")