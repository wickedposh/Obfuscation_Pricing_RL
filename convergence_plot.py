"""Convergence figure: (bar_eps_t, x_t) trajectories over training for several beta.
Shows the two-sided DQN dynamics converging to (different) fixed points.
Also overlays the solver fixed point as a reference line.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hetero_dynamic import train, Game, ProducerBelief, SharedConsumer  # reuse env+agents

# --- solver: independent fixed-point iteration (no learning), for reference ---
def solver_fixed_point(N=300, cost=50.0, xbar=150.0, iters=100):
    game = Game(N=N, cost=cost, xbar=xbar)
    xs = np.linspace(0, xbar, 151)
    eps_grid = np.linspace(0, 1, 101)
    bar_eps = 0.5
    for _ in range(iters):
        Stil = bar_eps*np.array([ (game.F_sampler(4000) >= x).mean() for x in xs]) + (1-bar_eps)/2
        x_star = xs[np.argmax((xs-cost)*Stil)]
        # each type best-responds individually (own surplus) at x_star
        W = game.F_sampler(20000)
        best = np.zeros_like(W); bestval = None
        for e in eps_grid:
            s = (W - x_star)*(e*(W>=x_star)+(1-e)/2)
            if bestval is None: bestval = s.copy(); best[:] = e
            else:
                m = s>bestval; bestval[m]=s[m]; best[m]=e
        bar_eps = best.mean()
    return bar_eps, x_star

betas = [0.0, 0.9]
colors = {0.0: "tab:blue", 0.9: "tab:red"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for beta in betas:
    # average trajectory over a few seeds for a clean line
    trajs = []
    for seed in range(3):
        _, _, traj = train(T=20000, gamma=beta, seed=seed)
        trajs.append(np.array([(t, e, x) for t, e, x in traj]))
    L = min(len(tr) for tr in trajs)
    arr = np.stack([tr[:L] for tr in trajs])          # (seeds, L, 3)
    t = arr[0, :, 0]
    eps_mean = arr[:, :, 1].mean(0); eps_sd = arr[:, :, 1].std(0)
    x_mean = arr[:, :, 2].mean(0)
    axes[0].plot(t, eps_mean, color=colors[beta], label=f"β={beta}")
    axes[0].fill_between(t, eps_mean-eps_sd, eps_mean+eps_sd, color=colors[beta], alpha=0.15)
    axes[1].plot(t, x_mean, color=colors[beta], label=f"β={beta}")

# solver reference for bar_eps (only meaningful for the static/myopic baseline)
se0, sx0 = solver_fixed_point()
axes[0].axhline(1.0, ls=":", color="gray", lw=1, label="truthful (ε=1)")

axes[0].set_xlabel("training step"); axes[0].set_ylabel(r"aggregate obfuscation  $\bar{\epsilon}_t$")
axes[0].set_title("Obfuscation converges to a $\\beta$-dependent level"); axes[0].set_ylim(0.4, 1.05); axes[0].legend(loc="lower right"); axes[0].grid(alpha=0.3)
axes[1].set_xlabel("training step"); axes[1].set_ylabel(r"price  $x_t$")
axes[1].set_title("Price converges"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("convergence.png", dpi=130)
print("saved convergence.png")
print(f"solver fixed point (β=0 / static, individual BR): bar_eps*={se0:.3f}, x*={sx0:.1f}")
