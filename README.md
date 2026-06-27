
## Project 2 — Heterogeneous Market: Does Obfuscation Survive a Rational Producer?

**Paper:** *Obfuscation in a Heterogeneous Pricing Market* (working title)

### Idea
Extends the single-consumer game to N heterogeneous consumers with private WTP
drawn from an unknown distribution F. The producer is now **strategic**: it
follows a belief-greedy strategy, learning the acceptance curve online and
pricing to maximise profit. A two-sided Bayesian game.

### Theory
- The producer's optimal price depends on F only through a 2-D local statistic
  (price, effective hazard) — a sufficient-statistic / dimension-reduction result
  (DKW consistency for large N).
- Existence of a Bayes–Nash equilibrium via Berge's maximum theorem + Kakutani's
  fixed-point theorem; obfuscation enters through a price-insensitive "noise floor"
  in effective demand.

### Key finding
Obfuscation's survival depends on whether consumers are forward-looking:
- **Static / myopic** (β = 0): equilibrium ε* ≈ 1 — obfuscation unravels to
  truthful play against a best-responding producer.
- **Dynamic / discounted** (β > 0): ε* < 1 — obfuscation survives, because
  corrupting the producer's belief today lowers future prices (a discounted,
  intertemporal benefit). ε* decreases with patience.

| β | ε* (mean ± std, 10 seeds) |
|------|------|
| 0.00 | 0.917 ± 0.027 |
| 0.50 | 0.875 ± 0.068 |
| 0.90 | 0.788 ± 0.091 |
| 0.99 | 0.790 ± 0.079 |

Spearman ρ = −0.630, p < 10⁻⁴. Trend decreases then plateaus (~β ≈ 0.9):
the obfuscation incentive saturates once consumers are sufficiently patient.

### Files
- `hetero_dynamic.py` — two-sided dynamic game (producer belief-learning + consumer DQN)


## Title : Application of RL algorithms to Game Theory
 
 
 ## Paper
  "Lying to a Bayesian: Optimal Randomisation in Repeated Pricing"
  DOI: https://doi.org/10.5281/zenodo.20695310   

## Abstract 
Reinforcement Learning has succeeded at a wide range of games ( Chess,
Go, Atari ). Here we apply RL methods to a game-theoretic problem in eco-
nomics : a repeated pricing game between a Bayesian-updating producer
and a consumer with private willingness-to-pay. We established existence
of the mixed strategy Nash equilibrium for the continuous-action extension
through Reny (1999), which we then approximate computationally via three
RL methods. We have used three different approaches (Monte Carlo, Q-
learning, Deep Q-Network) to simulate the benefits of randomisation for con-
sumer’s behaviour. All three approaches show a consistent result that ϵ∗ < 1,
supporting randomisation as an obfuscation strategy. Tabular Q-learning
shows no clear monotonic trend in ϵ∗(β), while DQN recovers the monotone
decrease predicted by Monte Carlo, reconciling the previously-observed dis-
crepancy. We empirically validate the Karp-Kleinberg lower bound on poste-
rior variance, confirming that mixed strategies prevent posterior collapse.





  ## Results
  All three methods find optimal ε* < 1 — randomisation improves consumer 
  surplus. DQN recovers the monotone ε*(β) trend that tabular Q-learning's
  discretisation misses.

 

  ## What's here
  - `MC.py` — Monte Carlo simulation
  - `QLearning.py` — tabular Q-learning
  - `dqn.py`, `dqn_agent.py` — Deep Q-Network
  - `eps_star_vs_beta.png` — results plot

## Reproduce

  Key result

  | β | ε* (mean) | std |
  |---|---|---|
  | 0.50 | 0.558 | 0.233 |
  | 0.70 | 0.520 | 0.247 |
  | 0.90 | 0.252 | 0.273 |
  | 0.99 | 0.118 | 0.050 |
  
DQN reconciles the previously observed discrepancy between Monte Carlo and tabular Q-learning.

