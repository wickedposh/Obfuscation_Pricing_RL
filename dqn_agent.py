import random
from collections import deque,namedtuple
import numpy as np
from torch import device

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
import torch
import torch.nn.functional as F

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
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    def select_action(self, state, nu):
        if np.random.rand() < nu:
            return np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = self.online(state_tensor)
                return q_values.argmax(dim=1).item()

    def store(self,s,a,r,s_next):
        self.memory.push(s,a,r,s_next)
    def learn(self,batch_size):
        if len(self.memory)<batch_size:
            return None
        batch=self.memory.sample(batch_size)
        states=torch.tensor(batch.state,dtype=torch.float32).to(device)
        rewards=torch.tensor(batch.reward,dtype=torch.float32).to(device)
        actions=torch.tensor(batch.action,dtype=torch.long).to(device)
        next_states=torch.tensor(batch.next_state,dtype=torch.float32).to(device)
        q_pred=self.online(states).gather(1,actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_state_values=self.target(next_states).max(1).values
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


agent = DQNAgent(n_observations=2, gamma=0.9)
print("OK init")

# fake state
s = (91.0, 10.0)
a = agent.select_action(s, nu=0.5)
print(f"OK select_action: {a}")

# push a few transitions
for _ in range(50):
    s = (np.random.rand() * 100, np.random.rand() * 10)
    a = np.random.randint(101)
    r = np.random.rand()
    s_next = (np.random.rand() * 100, np.random.rand() * 10)
    agent.store(s, a, r, s_next)
print(f"OK store, memory size: {len(agent.memory)}")

# learn
loss = agent.learn(batch_size=32)
print(f"OK learn, loss: {loss}")



