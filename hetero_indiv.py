import numpy as np
import random, torch
import torch.nn.functional as F
from collections import deque, namedtuple

device = torch.device("cpu")


class Game:
    """Individual consumers (shared policy), producer faces aggregate bar_eps."""
    def __init__(self, cost=50.0, xbar=150.0, N=300, F_sampler=None):
        self.cost = cost
        self.xbar = xbar
        self.N = N
        self.F_sampler = F_sampler or (lambda n: np.clip(np.random.normal(100, 15, n), 0, xbar))
        self.price_grid_ref = list(range(int(xbar) + 1))

    def accept_prob(self, W, x, eps):
        return eps * (W >= x).astype(float) + (1 - eps) / 2

    def effective_demand(self, x, bar_eps):
        W = self.F_sampler(self.N)
        return self.accept_prob(W, x, bar_eps).mean()


class DQN(torch.nn.Module):
    def __init__(self, n_obs, n_act, hidden=64):
        super().__init__()
        self.l1 = torch.nn.Linear(n_obs, hidden); self.l2 = torch.nn.Linear(hidden, hidden); self.l3 = torch.nn.Linear(hidden, n_act)
    def forward(self, x):
        return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
class ReplayMemory:
    def __init__(self, cap): self.memory = deque([], maxlen=cap)
    def push(self, *a): self.memory.append(Transition(*a))
    def sample(self, n): return Transition(*zip(*random.sample(self.memory, n)))
    def __len__(self): return len(self.memory)


class _BaseAgent:
    def __init__(self, n_obs, n_act, gamma, state_scale, lr=3e-4):
        self.n_actions = n_act; self.gamma = gamma
        self.online = DQN(n_obs, n_act).to(device); self.target = DQN(n_obs, n_act).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=lr)
        self.memory = ReplayMemory(100000)
        self.state_scale = torch.tensor(state_scale, dtype=torch.float32).to(device)
    def norm_state(self, s): return s / self.state_scale
    def select_action(self, state, nu):
        if np.random.rand() < nu: return np.random.randint(self.n_actions)
        with torch.no_grad():
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            return self.online(self.norm_state(st)).argmax(dim=1).item()
    def select_action_batch(self, states, nu):
        """Vectorized greedy/epsilon action for many states at once (shared consumer policy)."""
        states = np.asarray(states, dtype=np.float32)
        n = states.shape[0]
        out = np.empty(n, dtype=np.int64)
        explore = np.random.rand(n) < nu
        out[explore] = np.random.randint(self.n_actions, size=explore.sum())
        if (~explore).any():
            with torch.no_grad():
                st = torch.tensor(states[~explore], dtype=torch.float32).to(device)
                out[~explore] = self.online(self.norm_state(st)).argmax(dim=1).cpu().numpy()
        return out
    def store(self, s, a, r, s_next): self.memory.push(s, a, r, s_next)
    def learn(self, bs):
        if len(self.memory) < bs: return None
        b = self.memory.sample(bs)
        states = self.norm_state(torch.tensor(np.array(b.state), dtype=torch.float32).to(device))
        nxt = self.norm_state(torch.tensor(np.array(b.next_state), dtype=torch.float32).to(device))
        rewards = torch.tensor(b.reward, dtype=torch.float32).to(device)
        actions = torch.tensor(b.action, dtype=torch.long).to(device)
        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            tgt = rewards + self.gamma * self.target(nxt).max(1).values
        loss = torch.nn.MSELoss()(q, tgt)
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_value_(self.online.parameters(), 100); self.optimizer.step()
        return loss.item()
    def update_target(self): self.target.load_state_dict(self.online.state_dict())


class ConsumerAgent(_BaseAgent):
    # state = (own type W, posted price x). action = own eps.
    def __init__(self, gamma, xbar=150.0, lr=3e-4):
        self.eps_grid = [i / 100 for i in range(101)]
        super().__init__(2, len(self.eps_grid), gamma, state_scale=[xbar, xbar], lr=lr)


class ProducerAgent(_BaseAgent):
    def __init__(self, gamma, xbar=150.0, h_max=0.1, lr=3e-4):
        self.price_grid = list(range(int(xbar) + 1))
        super().__init__(2, len(self.price_grid), gamma, state_scale=[xbar, h_max], lr=lr)


def solver_individual(game, iters=80):
    """Fixed point where EACH consumer best-responds individually (own surplus),
    producer best-responds to aggregate. Symmetric: every type's BR eps, then aggregate."""
    xs = np.array(game.price_grid_ref); eps_grid = np.linspace(0, 1, 101)
    bar_eps = 0.5
    for _ in range(iters):
        Stil = np.array([game.effective_demand(x, bar_eps) for x in xs])
        x_star = xs[np.argmax((xs - game.cost) * Stil)]
        # each individual type picks eps to maximize OWN expected surplus at x_star.
        # own surplus(W,eps) = (W-x*)*accept_prob(W,x*,eps).
        W = game.F_sampler(20000)
        best_eps = np.zeros_like(W)
        for j, e in enumerate(eps_grid):
            s = (W - x_star) * game.accept_prob(W, x_star, e)
            if j == 0:
                best_val = s.copy(); best_eps[:] = e
            else:
                better = s > best_val
                best_val[better] = s[better]; best_eps[better] = e
        bar_eps = best_eps.mean()
    return float(bar_eps), float(x_star)


def train(T=30000, gamma=0.9, seed=0, log_every=300, N=300):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    game = Game(N=N)
    prod = ProducerAgent(gamma, game.xbar); cons = ConsumerAgent(gamma, game.xbar)
    nu_init, nu_min, decay, K = 1.0, 0.05, 0.9997, 500
    x, h = 100.0, 0.05
    traj = []
    for t in range(T):
        nu = max(nu_min, nu_init * decay ** t)
        # producer picks price from (x,h)
        p_state = (x, h)
        a_p = prod.select_action(p_state, nu)
        x_new = float(prod.price_grid[a_p])
        # each consumer observes (own W, x_new), picks own eps via SHARED policy
        W = game.F_sampler(game.N)
        c_states = np.stack([W, np.full(game.N, x_new)], axis=1)
        a_c = cons.select_action_batch(c_states, nu)
        eps_i = np.array([cons.eps_grid[a] for a in a_c])
        bar_eps = eps_i.mean()
        # realize accepts; each consumer's reward = OWN surplus
        p = game.accept_prob(W, x_new, eps_i)
        accept = (np.random.rand(game.N) < p).astype(float)
        cons_r = (W - x_new) * accept                       # per-consumer
        prod_r = (x_new - game.cost) * accept.mean()
        # effective hazard for producer state
        S = (game.accept_prob(W, x_new, bar_eps)).mean()
        S2 = (game.accept_prob(W, x_new + 1.0, bar_eps)).mean()
        h_new = max((S - S2), 1e-6) / max(S, 1e-6)
        # store: producer one transition; consumers a subsample (keep buffer balanced)
        prod.store(p_state, a_p, prod_r, (x_new, h_new))
        idx = np.random.choice(game.N, size=min(16, game.N), replace=False)
        for i in idx:
            cons.store((W[i], x_new), int(a_c[i]), float(cons_r[i]), (W[i], x_new))
        prod.learn(64); cons.learn(64)
        if t % K == 0:
            prod.update_target(); cons.update_target()
        x, h = x_new, h_new
        if t % log_every == 0:
            traj.append((t, bar_eps, x))
    return prod, cons, game, traj


if __name__ == "__main__":
    prod, cons, game, traj = train(T=30000, gamma=0.9, seed=0)
    tail = traj[-15:]
    eps_tail = np.mean([e for _, e, _ in tail]); x_tail = np.mean([x for _, _, x in tail])
    print(f"Two-sided DQN (last-15 avg):  bar_eps* = {eps_tail:.3f},  x* = {x_tail:.1f}")
    se, sx = solver_individual(game)
    print(f"Solver (individual BR):       bar_eps* = {se:.3f},  x* = {sx:.1f}")
    print()
    print("trajectory (t, bar_eps, x):")
    for tt, e, xx in traj[::max(1, len(traj)//15)]:
        print(f"  t={tt:6d}  bar_eps={e:.2f}  x={xx:6.1f}")
