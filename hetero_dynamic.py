"""
Dynamic heterogeneous obfuscation game WITH producer memory.

Key difference from the static runs: the producer carries a BELIEF across rounds
and prices from it. So a consumer's obfuscation today corrupts the producer's
inference -> lowers future prices -> discounted private benefit (mediated by beta).
This is the channel that can make individual obfuscation rational.

Producer belief: a nonparametric estimate of the acceptance curve A(x) = P(accept | price x),
updated online from observed (price, accept-rate) pairs via a local (kernel) running average.
The producer prices to maximize (x - c) * Ahat(x) given its current belief.
Consumers (shared policy, own discounted surplus) choose eps_i from state (W_i, x_t).
"""
import numpy as np, random, torch
import torch.nn.functional as F
from collections import deque, namedtuple

device = torch.device("cpu")


class ProducerBelief:
    """Online nonparametric estimate of acceptance curve A(x) on a price grid."""
    def __init__(self, xbar=150.0, n_grid=151, lr=0.05):
        self.xs = np.linspace(0, xbar, n_grid)
        self.Ahat = np.clip(1 - self.xs / xbar, 0.01, 0.99)  # prior: roughly decreasing
        self.lr = lr
        self.bw = 6.0  # kernel bandwidth for credit assignment across nearby prices

    def update(self, price, accept_rate):
        # kernel-weighted online update of the whole curve near `price`
        w = np.exp(-0.5 * ((self.xs - price) / self.bw) ** 2)
        self.Ahat += self.lr * w * (accept_rate - self.Ahat)
        self.Ahat = np.clip(self.Ahat, 1e-3, 1 - 1e-3)

    def best_price(self, cost):
        profit = (self.xs - cost) * self.Ahat
        return float(self.xs[np.argmax(profit)])

    def hazard_at(self, price):
        i = int(np.clip(np.searchsorted(self.xs, price), 1, len(self.xs) - 1))
        S = self.Ahat[i]
        ftil = max(self.Ahat[i - 1] - self.Ahat[i], 1e-6)
        return ftil / max(S, 1e-6)


class Game:
    def __init__(self, cost=50.0, xbar=150.0, N=300, F_sampler=None):
        self.cost, self.xbar, self.N = cost, xbar, N
        self.F_sampler = F_sampler or (lambda n: np.clip(np.random.normal(100, 15, n), 0, xbar))

    def accept_prob(self, W, x, eps):
        return eps * (W >= x).astype(float) + (1 - eps) / 2


class DQN(torch.nn.Module):
    def __init__(self, n_obs, n_act, hidden=64):
        super().__init__()
        self.l1 = torch.nn.Linear(n_obs, hidden); self.l2 = torch.nn.Linear(hidden, hidden); self.l3 = torch.nn.Linear(hidden, n_act)
    def forward(self, x): return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))


Transition = namedtuple('T', ('s', 'a', 'r', 's2'))
class Mem:
    def __init__(self, cap): self.m = deque([], maxlen=cap)
    def push(self, *a): self.m.append(Transition(*a))
    def sample(self, n): return Transition(*zip(*random.sample(self.m, n)))
    def __len__(self): return len(self.m)


class SharedConsumer:
    """One shared policy net. State = (own W, price). Action = own eps. Reward = own DISCOUNTED surplus (gamma in bootstrap)."""
    def __init__(self, gamma, xbar=150.0, lr=3e-4):
        self.eps_grid = np.linspace(0, 1, 101)
        self.n = len(self.eps_grid); self.gamma = gamma
        self.online = DQN(2, self.n).to(device); self.target = DQN(2, self.n).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.AdamW(self.online.parameters(), lr=lr)
        self.mem = Mem(200000)
        self.scale = torch.tensor([xbar, xbar], dtype=torch.float32)

    def act_batch(self, states, nu):
        states = np.asarray(states, np.float32); m = len(states)
        out = np.empty(m, np.int64); expl = np.random.rand(m) < nu
        out[expl] = np.random.randint(self.n, size=expl.sum())
        if (~expl).any():
            with torch.no_grad():
                st = torch.tensor(states[~expl], dtype=torch.float32) / self.scale
                out[~expl] = self.online(st).argmax(1).numpy()
        return out

    def learn(self, bs=128):
        if len(self.mem) < bs: return
        b = self.mem.sample(bs)
        s = torch.tensor(np.array(b.s), dtype=torch.float32) / self.scale
        s2 = torch.tensor(np.array(b.s2), dtype=torch.float32) / self.scale
        r = torch.tensor(b.r, dtype=torch.float32); a = torch.tensor(b.a, dtype=torch.long)
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            tgt = r + self.gamma * self.target(s2).max(1).values
        loss = torch.nn.MSELoss()(q, tgt)
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_value_(self.online.parameters(), 100); self.opt.step()

    def sync(self): self.target.load_state_dict(self.online.state_dict())

    def greedy_eps(self, W_samples, price):
        st = torch.tensor(np.stack([W_samples, np.full(len(W_samples), price)], 1), dtype=torch.float32) / self.scale
        with torch.no_grad():
            idx = self.online(st).argmax(1).numpy()
        return self.eps_grid[idx].mean()


def train(T=40000, gamma=0.9, seed=0, N=300, K=500, log_every=400):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    game = Game(N=N); belief = ProducerBelief(xbar=game.xbar)
    cons = SharedConsumer(gamma, game.xbar)
    nu_init, nu_min, decay = 1.0, 0.05, 0.9998
    traj = []
    price = belief.best_price(game.cost)
    for t in range(T):
        nu = max(nu_min, nu_init * decay ** t)
        W = game.F_sampler(game.N)
        cur_states = np.stack([W, np.full(game.N, price)], 1)
        a = cons.act_batch(cur_states, nu)
        eps_i = cons.eps_grid[a]
        bar_eps = eps_i.mean()
        # realize accepts at CURRENT price
        p = game.accept_prob(W, price, eps_i)
        accept = (np.random.rand(game.N) < p).astype(float)
        accept_rate = accept.mean()
        # PRODUCER UPDATES BELIEF from observed (price, accept_rate) -> memory across rounds
        belief.update(price, accept_rate)
        next_price = belief.best_price(game.cost)   # endogenous: depends on accumulated obfuscation
        # consumer reward = own immediate surplus; next_state carries the NEW price (consequence of obfuscation)
        cons_r = (W - price) * accept
        next_states = np.stack([W, np.full(game.N, next_price)], 1)
        idx = np.random.choice(game.N, size=min(32, game.N), replace=False)
        for i in idx:
            cons.mem.push((W[i], price), int(a[i]), float(cons_r[i]), (W[i], next_price))
        cons.learn(128)
        if t % K == 0: cons.sync()
        price = next_price
        if t % log_every == 0:
            traj.append((t, bar_eps, price))
    # final greedy bar_eps at the converged price, averaged over many fresh types
    Wbig = game.F_sampler(20000)
    final_eps = cons.greedy_eps(Wbig, price)
    return final_eps, price, traj


if __name__ == "__main__":
    print("Dynamic game WITH producer memory. Sweeping beta (=gamma)...\n")
    for gamma in [0.0, 0.5, 0.9, 0.99]:
        eps_stars = []
        for seed in range(3):
            fe, px, traj = train(T=25000, gamma=gamma, seed=seed)
            eps_stars.append(fe)
        print(f"beta={gamma:>4}:  bar_eps* = {np.mean(eps_stars):.3f} +/- {np.std(eps_stars):.3f}   (final price ~ {px:.0f})")
    print("\n(If bar_eps* DECREASES as beta rises and is <1 for high beta -> obfuscation survives via reputation, beta-dependent.)")
