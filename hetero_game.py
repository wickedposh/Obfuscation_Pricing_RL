import numpy as np
import random, torch
import torch.nn.functional as F
from collections import deque, namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Game:
    def __init__(self, cost=50.0, xbar=150.0, N=500, F_sampler=None):
        self.cost = cost
        self.xbar = xbar
        self.N = N
        self.F_sampler = F_sampler or (lambda n: np.clip(np.random.normal(100, 15, n), 0, xbar))

    def accept_prob(self, W, x, bar_eps):
        return bar_eps * (W >= x).astype(float) + (1 - bar_eps) / 2

    def effective_demand(self, x, bar_eps):
        W = self.F_sampler(self.N)
        return self.accept_prob(W, x, bar_eps).mean()

    def play(self, x, bar_eps):
        W = self.F_sampler(self.N)
        p = self.accept_prob(W, x, bar_eps)
        accept = (np.random.rand(self.N) < p).astype(float)
        producer_r = (x - self.cost) * accept.mean()
        consumer_r = ((W - x) * accept).mean()
        eff_hazard = self._eff_hazard(x, bar_eps, W)
        producer_state = (x, eff_hazard)
        return producer_state, producer_r, consumer_r

    def _eff_hazard(self, x, bar_eps, W, dx=1.0):
        S = self.accept_prob(W, x, bar_eps).mean()
        S2 = self.accept_prob(W, x + dx, bar_eps).mean()
        ftil = max((S - S2) / dx, 1e-6)
        return ftil / max(S, 1e-6)


class DQN(torch.nn.Module):
    def __init__(self, n_observations, n_actions, hidden=64):
        super().__init__()
        self.layer1 = torch.nn.Linear(n_observations, hidden)
        self.layer2 = torch.nn.Linear(hidden, hidden)
        self.layer3 = torch.nn.Linear(hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return Transition(*zip(*random.sample(self.memory, batch_size)))
    def __len__(self):
        return len(self.memory)


class _BaseAgent:
    """Shared DQN logic; subclasses set self.action_values."""
    def __init__(self, n_observations, n_actions, gamma, state_scale, lr=3e-4):
        self.n_actions = n_actions
        self.gamma = gamma
        self.online = DQN(n_observations, n_actions).to(device)
        self.target = DQN(n_observations, n_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=lr)
        self.memory = ReplayMemory(100000)
        self.state_scale = torch.tensor(state_scale, dtype=torch.float32).to(device)

    def norm_state(self, s):
        return s / self.state_scale

    def select_action(self, state, nu):
        if np.random.rand() < nu:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            return self.online(self.norm_state(st)).argmax(dim=1).item()

    def store(self, s, a, r, s_next):
        self.memory.push(s, a, r, s_next)

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        b = self.memory.sample(batch_size)
        states = self.norm_state(torch.tensor(np.array(b.state), dtype=torch.float32).to(device))
        next_states = self.norm_state(torch.tensor(np.array(b.next_state), dtype=torch.float32).to(device))
        rewards = torch.tensor(b.reward, dtype=torch.float32).to(device)
        actions = torch.tensor(b.action, dtype=torch.long).to(device)
        q_pred = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(next_states).max(1).values
            target = rewards + self.gamma * q_next
        loss = torch.nn.MSELoss()(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.online.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())


class ConsumerAgent(_BaseAgent):
    def __init__(self, gamma, xbar=150.0, lr=3e-4):
        self.eps_grid = [i / 100 for i in range(101)]
        super().__init__(n_observations=1, n_actions=len(self.eps_grid),
                         gamma=gamma, state_scale=[xbar], lr=lr)


class ProducerAgent(_BaseAgent):
    def __init__(self, gamma, xbar=150.0, h_max=0.1, lr=3e-4):
        self.price_grid = list(range(int(xbar) + 1))
        super().__init__(n_observations=2, n_actions=len(self.price_grid),
                         gamma=gamma, state_scale=[xbar, h_max], lr=lr)


# ---------- Solver cross-check (no learning): iterate the BR fixed point ----------
def solver_fixed_point(game, iters=60):
    xs = np.array(game.price_grid_ref)
    eps_grid = np.linspace(0, 1, 101)
    bar_eps = 0.5
    for _ in range(iters):
        # producer best-responds to current bar_eps
        Stil = np.array([game.effective_demand(x, bar_eps) for x in xs])
        x_star = xs[np.argmax((xs - game.cost) * Stil)]
        # aggregate consumer best-responds to x_star: maximize expected surplus over eps
        W = game.F_sampler(20000)
        surplus = [((W - x_star) * game.accept_prob(W, x_star, e)).mean() for e in eps_grid]
        bar_eps = eps_grid[int(np.argmax(surplus))]
    return float(bar_eps), float(x_star)


# ---------- Two-sided learning loop ----------
def train(T=40000, gamma=0.9, seed=0, log_every=200):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    game = Game()
    game.price_grid_ref = list(range(int(game.xbar) + 1))
    prod = ProducerAgent(gamma=gamma, xbar=game.xbar)
    cons = ConsumerAgent(gamma=gamma, xbar=game.xbar)

    nu_init, nu_min, decay, K = 1.0, 0.05, 0.9997, 500
    x = 100.0; h = 0.05
    traj = []
    for t in range(T):
        nu = max(nu_min, nu_init * (decay ** t))
        # consumer observes current price, picks bar_eps
        c_state = (x,)
        a_c = cons.select_action(c_state, nu)
        bar_eps = cons.eps_grid[a_c]
        # producer observes (x, h), picks next price
        p_state = (x, h)
        a_p = prod.select_action(p_state, nu)
        x_new = float(prod.price_grid[a_p])
        # play round at the producer's chosen price and consumer's bar_eps
        (x_obs, h_new), prod_r, cons_r = game.play(x_new, bar_eps)
        # next states
        c_next = (x_new,)
        p_next = (x_new, h_new)
        cons.store(c_state, a_c, cons_r, c_next)
        prod.store(p_state, a_p, prod_r, p_next)
        cons.learn(64); prod.learn(64)
        if t % K == 0:
            cons.update_target(); prod.update_target()
        x, h = x_new, h_new
        if t % log_every == 0:
            traj.append((t, bar_eps, x))
    return prod, cons, game, traj


if __name__ == "__main__":
    prod, cons, game, traj = train(T=40000, gamma=0.9, seed=0)
    game.price_grid_ref = list(range(int(game.xbar) + 1))
    # final greedy readout, averaged over last stretch of trajectory
    tail = traj[-20:]
    eps_tail = np.mean([e for _, e, _ in tail])
    x_tail = np.mean([x for _, _, x in tail])
    print(f"Two-sided DQN (last-20 avg):  bar_eps* = {eps_tail:.3f},  x* = {x_tail:.1f}")
    se, sx = solver_fixed_point(game)
    print(f"Solver fixed point:           bar_eps* = {se:.3f},  x* = {sx:.1f}")
    print()
    print("trajectory (t, bar_eps, x):")
    for tt, e, xx in traj[::max(1, len(traj)//15)]:
        print(f"  t={tt:6d}  bar_eps={e:.2f}  x={xx:6.1f}")
