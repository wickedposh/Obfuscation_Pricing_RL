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
  [1-2 sentences of the headline finding]
  All three methods find optimal ε* < 1 — randomisation improves consumer 
  surplus. DQN recovers the monotone ε*(β) trend that tabular Q-learning's
  discretisation misses.

 


##Code

-"MC.py" - Monte Carlo simulation
-"QLearning.py" - tabular Q-learning simulation
-"dqn.py" - DQN simulation

## Reproduce

  Key result

  ┌──────┬───────────┬───────┐                                                                                                    
  │  β   │ ε* (mean) │  std  │
  ├──────┼───────────┼───────┤                                                                                                    
  │ 0.50 │ 0.558     │ 0.233 │
  ├──────┼───────────┼───────┤
  │ 0.70 │ 0.520     │ 0.247 │
  ├──────┼───────────┼───────┤
  │ 0.90 │ 0.252     │ 0.273 │
  ├──────┼───────────┼───────┤                                                                                                    
  │ 0.99 │ 0.118     │ 0.050 │
  └──────┴───────────┴───────┘   
DQN reconciles the previously observed discrepancy between Monte Carlo and tabular Q-learning.

