from scipy.stats import norm
import numpy as np
import random,torch
import torch.nn.functional as F
from collections import deque,namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def bayes(a,price,epsilon_star,mu,sigma):
    z = (price - mu) / sigma
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    if a == +1:
        denom = epsilon_star * (1 - Phi) + (1 - epsilon_star) / 2
        numer = epsilon_star * (mu * (1 - Phi) + sigma * phi) + (1 - epsilon_star) * mu / 2
        second = epsilon_star * ((mu ** 2 + sigma ** 2) * (1 - Phi) + sigma * (mu + price) * phi) + (1 - epsilon_star) * (
                mu ** 2 + sigma ** 2) / 2
    else:
        denom = epsilon_star * (Phi) + (1 - epsilon_star) / 2
        numer = epsilon_star * (mu * Phi - sigma * phi) + (1 - epsilon_star) * mu / 2
        second = epsilon_star * ((mu ** 2 + sigma ** 2) * (Phi) - sigma * (mu + price) * phi) + (1 - epsilon_star) * (
                mu ** 2 + sigma ** 2) / 2

    mu = float(numer / denom)
    var = max(float(second / denom) - mu ** 2,1e-6)
    sigma=float(np.sqrt(var))
    return mu,sigma


class PricingGame:
    def __init__(self, mu_0=91.0,sigma_0=10.0,wtp=100.0,cost=50.0):
        self.mu_0=mu_0
        self.sigma_0=sigma_0
        self.wtp=wtp
        self.cost=cost
        self.history_mu=[]
        self.history_sigma=[]
    def reset(self):
        self.mu=self.mu_0
        self.sigma=self.sigma_0
        return(self.mu,self.sigma)
    def step(self,eps,price):
        if np.random.rand()>eps:
            if np.random.rand()<=1/2:
                a=+1
                r=float(self.wtp-price)
            else:
                a=-1
                r=0
        else:
            if price<=self.wtp:
                a=+1
                r=float(self.wtp-price)
            else:
                a=-1
                r=0

        new_mu,new_sigma=bayes(a,price,eps,self.mu,self.sigma)
        self.history_mu.append(new_mu)
        self.history_sigma.append(new_sigma)
        self.mu=new_mu
        self.sigma=new_sigma

        return (self.mu,self.sigma),r,False

Transition=namedtuple('Transition',('state','action','reward','next_state'))
class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        batch=random.sample(self.memory, batch_size)
        batch=Transition(*zip(*batch))
        return batch

    def __len__(self):
        return len(self.memory)

class DQN(torch.nn.Module):
    def __init__(self, n_observations, n_actions,hidden=64):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, hidden)
        self.layer2 = torch.nn.Linear(hidden, hidden)
        self.layer3 = torch.nn.Linear(hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)




class DQNAgent:
    def __init__(self,n_observations,gamma,lr=3e-4):
        self.eps_grid = [float(i/100) for i in range(101)]
        self.n_actions=len(self.eps_grid)
        self.n_observations=n_observations
        self.lr=lr
        self.gamma=gamma
        self.online=DQN(self.n_observations,self.n_actions).to(device)
        self.target=DQN(self.n_observations,self.n_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer=torch.optim.AdamW(self.online.parameters(),lr=self.lr)
        self.memory = ReplayMemory(capacity=100000)
    def norm_state(s):
        mu,sigma=s
        return (mu/150.0,sigma/10.0)

    def select_action(self, state, nu):
        if np.random.rand() < nu:
            return np.random.randint(self.n_actions)
        else:
            state_norm=(state[0]/150.0,state[1]/10.0)
            with torch.no_grad():
                state_tensor = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = self.online(state_tensor)
                return q_values.argmax(dim=1).item()

    def store(self,s,a,r,s_next):
        self.memory.push(s,a,r,s_next)
    def learn(self,batch_size):
        if len(self.memory)<batch_size:
            return None
        batch=self.memory.sample(batch_size)
        states=torch.tensor(batch.state,dtype=torch.float32).to(device)
        states_normalized=states/torch.tensor([150.0,10.0],dtype=torch.float32).to(device)
        rewards=torch.tensor(batch.reward,dtype=torch.float32).to(device)
        actions=torch.tensor(batch.action,dtype=torch.long).to(device)
        next_states=torch.tensor(batch.next_state,dtype=torch.float32).to(device)
        next_state_normalized=next_states/torch.tensor([150.0,10.0],dtype=torch.float32).to(device)
        q_pred=self.online(states_normalized).gather(1,actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_state_values=self.target(next_state_normalized).max(1).values
            expected_values=(next_state_values*self.gamma)+rewards
        criterion=torch.nn.MSELoss()
        loss=criterion(q_pred,expected_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.online.parameters(),100)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())


T_max = 50000
nu_init, nu_min, decay = 1.0, 0.05, 0.9995
K = 1000

env = PricingGame()
agent = DQNAgent(n_observations=2, gamma=0.9)
state = env.reset()
losses, mu_trace, sigma_trace = [], [], []

for t in range(T_max):
    nu = max(nu_min, nu_init * (decay ** t))
    action_idx = agent.select_action(state, nu)
    eps = agent.eps_grid[action_idx]
    next_state, reward, _ = env.step(eps, env.mu)
    agent.store(state, action_idx, reward, next_state)

    loss = agent.learn(batch_size=32)
    if loss is not None:
        losses.append(loss)

    if t % K == 0:
        agent.update_target()

    state = next_state
    mu_trace.append(state[0])
    sigma_trace.append(state[1])

    # Diagnostics
print(f"Final mu: {mu_trace[-1]:.2f}")
print(f"Final sigma: {sigma_trace[-1]:.4f}")
print(f"Final loss: {losses[-1] if losses else 'N/A':.4f}")
print(f"Mean loss over last 100 steps: {np.mean(losses[-100:]):.4f}")
print(f"Loss trajectory: first={losses[0]:.4f}, mid={losses[len(losses) // 2]:.4f}, last={losses[-1]:.4f}")
# 1. What action does DQN pick at different states?
test_states = [(91, 10), (95, 5), (100, 1), (100, 0.001)]
for s in test_states:
    a = agent.select_action(s, nu=0.0)  # pure greedy
    print(f"state={s}: best ε = {agent.eps_grid[a]:.2f}")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(losses)
axes[0].set_title("Loss")
axes[0].set_yscale("log")
axes[1].plot(sigma_trace)
axes[1].set_title("Sigma")
plt.show()

results = {}
gammas=[0.5,0.7,0.9,0.99]
for gamma in gammas:
    eps_stars=[]
    for seed in range(5):
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = PricingGame()
        agent = DQNAgent(n_observations=2, gamma=gamma)
        state = env.reset()
        losses, mu_trace, sigma_trace = [], [], []
        for t in range(T_max):
            nu = max(nu_min, nu_init * (decay ** t))
            action_idx = agent.select_action(state, nu)
            eps = agent.eps_grid[action_idx]
            next_state, reward, _ = env.step(eps, env.mu)
            agent.store(state, action_idx, reward, next_state)

            loss = agent.learn(batch_size=32)
            if loss is not None:
                losses.append(loss)

            if t % K == 0:
                agent.update_target()

            state = next_state
        with torch.no_grad():
            s = torch.tensor([91.0 / 150.0, 10.0 / 10.0]).unsqueeze(0).to(
                    device)  # normalized
            q = agent.online(s)
            eps_stars.append(agent.eps_grid[q.argmax().item()])
    mean_eps=np.mean(eps_stars)
    std_eps=np.std(eps_stars)
    results[gamma] = (mean_eps, std_eps)
    print(f"γ={gamma}: ε* = {mean_eps:.3f} {std_eps:.4f}")




