## Project 1 — Lying to a Bayesian: Optimal Randomisation in Repeated Pricing
**DOI:** https://doi.org/10.5281/zenodo.20695310

### Abstract
We apply reinforcement learning to a repeated pricing game between a
Bayesian-updating producer and a strategic consumer with private
willingness-to-pay. We establish existence of a mixed-strategy Nash equilibrium
for the continuous-action game via Reny (1999), then approximate it with three
methods (Monte Carlo, tabular Q-learning, DQN). Across all methods the optimal
mixing satisfies ε* < 1, supporting randomisation as an obfuscation strategy.
Monte Carlo and DQN recover a decreasing ε*(β) — more patient consumers
randomise more — while tabular Q-learning shows no significant trend. We trace
this discrepancy to estimator variance on a value surface that is flat in ε,
rather than to belief-state discretisation.

### Results
- **ε* < 1 across all methods** — randomisation improves consumer surplus over
  deterministic play.
- **ε*(β) decreases** in Monte Carlo and DQN (verified across seeds); tabular
  Q-learning's argmax is too high-variance to resolve the trend.
- The producer's posterior is regularised with a variance floor for numerical
  stability; the bounded-σ behaviour is a consequence of that floor, not an
  emergent property of randomisation.

### Files
- `MC.py` — Monte Carlo simulation
- `QLearning.py` — tabular Q-learning
- `dqn.py`, `dqn_agent.py` — Deep Q-Network
- `eps_star_vs_beta.png` — results plot

---
**DOI:** https://doi.org/10.5281/zenodo.20983755
## Project 2 — Heterogeneous Market: Does Obfuscation Survive a Rational Producer?
*(working title — in progress)*

### Idea
Extends Project 1 to N heterogeneous consumers with private WTP drawn from an
unknown distribution F. The producer is now strategic: it follows a
belief-greedy strategy, learning the acceptance curve online and pricing to
maximise profit. A two-sided Bayesian game.

### Theory
- The producer's optimal price depends on F only through a 2-D local statistic
  (price, effective hazard) — a sufficient-statistic result (DKW consistency
  for large N).
- Existence of a Bayes–Nash equilibrium via Berge's maximum theorem + Kakutani's
  fixed-point theorem; obfuscation enters through a price-insensitive "noise
  floor" in effective demand.

### Key finding
Obfuscation's survival depends on whether consumers are forward-looking:
- **Static / myopic (β = 0):** ε* ≈ 1 — obfuscation unravels to truthful play
  against a best-responding producer.
- **Dynamic / discounted (β > 0):** ε* < 1 — obfuscation survives, because
  corrupting the producer's belief today lowers future prices (a discounted
  intertemporal benefit). ε* decreases with patience.

| β | ε* (mean ± std, 24 seeds) |
|------|------|
| 0.00 | 0.914 ± 0.035 |
| 0.50 | 0.852 ± 0.048 |
| 0.90 | 0.740 ± 0.071 |
| 0.99 | 0.749 ± 0.059 |

Spearman ρ = −0.769, p < 10⁻⁶ (96 raw (β, ε*) pairs). Reproduced across four
independent seed sets. The trend decreases then plateaus (β ≈ 0.9): the
obfuscation incentive saturates once consumers are sufficiently patient.

### Files
- `hetero_dynamic.py` — two-sided dynamic game (producer belief-learning + consumer DQN)

---
