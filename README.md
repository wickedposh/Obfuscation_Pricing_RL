# Lying to a Bayesian: Optimal Randomisation in Repeated Pricing
**Yuil Ahn ** - wicekdposh@gmail.com - May 2026

## Abstract 
Reinforcement Learning solves a wide range of games ( Chess, Go, Atari ). Here we apply RL methods to a game-theoretic problem in economics : a repeated pricing game between a Bayesian-updating producer and a consumer with private willingness-to-pay. We established existence of Bayesian Nash Equilibrium for the discrete-action setting via Carbonell-Nicolau (2015), and of the mixed strategy Nash equilibrium for the continuous-action extension through Reny (1999). We have used three different approaches (Monte Carlo, Q-learning, Deep Q-Network) to simulate the benefits of randomisation for consumer’s behaviour. All three approaches show a consistent result that $\epsilon^{*}<1$, supporting randomisation as an obfuscation strategy. Tabular Q-learning shows no clear monotonic trend in $\epsilon^{*}(\beta)$, while DQN recovers the monotone decrease predicted by Monte Carlo, reconciling the previously-observed discrepancy. We empirically validate the Karp-Kleinberg lower bound on posterior variance, confirming that mixed strategies prevent posterior collapse.

##Paper
Project.pdf

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

