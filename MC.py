import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, expand, Rational, lambdify,diff,hessian,solve
from scipy.stats import norm

eps_sym, beta_sym = symbols('epsilon beta', positive=True)
states = [(0, 150, Rational(1))]
V1 = 0

for t in range(7):
    for L, U, p in states:
        x = (L + U) // 2
        p_plus = (1 + eps_sym) / 2 if x <= 100 else (1 - eps_sym) / 2
        V1 = V1 + beta_sym ** t * p * (100 - x) * p_plus
    new_states = []
    for L, U, p in states:
        x = (L + U) // 2
        p_plus = (1 + eps_sym) / 2 if x <= 100 else (1 - eps_sym) / 2
        new_states.append((x, U, p * p_plus))
        new_states.append((L, x, p * (1 - p_plus)))
    states = new_states

V1 = expand(V1)
V1_func = lambdify((eps_sym, beta_sym), V1, 'numpy')




def bayesian_update(mu, sigma, x, a, eps_val):
    z = (x - mu) / sigma
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    half = (1 - eps_val) / 2

    if a == +1:
        denom = eps_val * (1 - Phi) + half
        numer = eps_val * (mu * (1 - Phi) + sigma * phi) + half * mu
        second = eps_val * ((mu ** 2 + sigma ** 2) * (1 - Phi) + sigma * (mu + x) * phi) + half * (mu ** 2 + sigma ** 2)
    else:
        denom = eps_val * Phi + half
        numer = eps_val * (mu * Phi - sigma * phi) + half * mu
        second = eps_val * ((mu ** 2 + sigma ** 2) * Phi - sigma * (mu + x) * phi) + half * (mu ** 2 + sigma ** 2)

    mu_new = numer / denom
    var = second / denom - mu_new ** 2
    if var <= 0 or np.isnan(var):
        var = 1e-6
    sigma_new = np.sqrt(var)
    sigma_new = max(0.1, min(50, sigma_new))  # safety clip

    return mu_new, sigma_new


def V_phase2(eps_val, beta_val, T=1000, n_sim=100):
    """Estimate phase 2 value via Monte Carlo simulation."""
    total_value = 0.0
    for _ in range(n_sim):
        # Producer's posterior: start with mean=100, sigma=10 (wide initial)
        mu, sigma = 91.0, 10.0
        # Approximate true WTP = 100
        wtp = 100
        running_value = 0.0
        for t in range(T):
            price = mu  # producer charges posterior mean
            # Consumer's mixed action
            if np.random.rand() < eps_val:
                # Optimal
                a = +1 if price <= wtp else -1
            else:
                # Random
                a = np.random.choice([-1, +1])
                # Surplus
            if a == +1:
                running_value += (beta_val ** t) * (wtp - price)
            # Bayesian update (simplified Gaussian approximation)
            # Likelihood: P(a=+1 | wtp, price) under mixed policy
            # Update mu, sigma based on observation

            mu,sigma = bayesian_update(mu,sigma,price,a,eps_val)

        total_value += running_value
    return total_value / n_sim


# ============================================================
# Combine and find epsilon*(beta)
# ============================================================
betas = np.linspace(0.5, 0.99, 5)
eps_grid = np.linspace(0.01, 0.99, 11)

eps_star = []
V_star = []

for b in betas:
    V_total = []
    for e in eps_grid:
        v1 = float(V1_func(e, b))
        v2 = V_phase2(e, b, T=100, n_sim=100)
        V_total.append(v1 + (b ** 7) * v2)
    V_total = np.array(V_total)
    idx = np.argmax(V_total)
    eps_star.append(eps_grid[idx])
    V_star.append(V_total[idx])


    # ============================================================
# Plots
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(betas, eps_star, 'o-')
axes[0].set_xlabel('β (discount factor)')
axes[0].set_ylabel('ε* (optimal exploration)')
axes[0].set_title('Optimal ε* vs β')
axes[0].grid(True)

axes[1].plot(betas, V_star, 's-', color='orange')
axes[1].set_xlabel('β (discount factor)')
axes[1].set_ylabel('V*(β)')
axes[1].set_title('Optimal Value vs β')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('eps_star_vs_beta.png', dpi=120)
plt.show()

print(f"V_phase1 polynomial:\n{V1}")
print(f"\nε*(β) values: {list(zip(betas.round(2), [round(e, 3) for e in eps_star]))}")
a=diff(V1,eps_sym)
b=hessian(V1,(eps_sym,eps_sym))
dd=solve(a,eps_sym)
