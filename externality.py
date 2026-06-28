"""
Externality experiment: dynamic game, INDIVIDUAL vs COOPERATIVE consumers.

- INDIVIDUAL: each consumer maximises its OWN discounted surplus (free-rider case).
              Shared policy keyed on (own W, price). [this is hetero_dynamic.py's setup]
- COOPERATIVE: one agent chooses bar_eps to maximise AGGREGATE discounted surplus
               (internalises the shared price benefit -> no free-rider loss).

If cooperative obfuscates MORE (lower bar_eps*) than individual, the gap = free-rider
externality. We sweep beta and report both.
"""
import numpy as np, random, torch
import torch.nn.functional as F
from collections import deque, namedtuple
from scipy.stats import spearmanr

device = torch.device("cpu")

# ---------- producer belief (shared) ----------
class ProducerBelief:
    def __init__(self, xbar=150.0, n_grid=151, lr=0.05):
        self.xs = np.linspace(0, xbar, n_grid)
        self.Ahat = np.clip(1 - self.xs / xbar, 0.01, 0.99)
        self.lr = lr; self.bw = 6.0
    def update(self, price, accept_rate):
        w = np.exp(-0.5 * ((self.xs - price) / self.bw) ** 2)
        self.Ahat += self.lr * w * (accept_rate - self.Ahat)
        self.Ahat = np.clip(self.Ahat, 1e-3, 1 - 1e-3)
    def best_price(self, cost):
        return float(self.xs[np.argmax((self.xs - cost) * self.Ahat)])

def F_sampler(n, xbar=150.0): return np.clip(np.random.normal(100, 15, n), 0, xbar)
def accept_prob(W, x, eps): return eps * (W >= x).astype(float) + (1 - eps) / 2

# ---------- DQN ----------
class DQN(torch.nn.Module):
    def __init__(self, n_obs, n_act, h=64):
        super().__init__(); self.l1=torch.nn.Linear(n_obs,h); self.l2=torch.nn.Linear(h,h); self.l3=torch.nn.Linear(h,n_act)
    def forward(self,x): return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))
Tr = namedtuple('Tr',('s','a','r','s2'))
class Mem:
    def __init__(s,c): s.m=deque([],maxlen=c)
    def push(s,*a): s.m.append(Tr(*a))
    def sample(s,n): return Tr(*zip(*random.sample(s.m,n)))
    def __len__(s): return len(s.m)

class Agent:
    def __init__(self, n_obs, gamma, scale, lr=3e-4):
        self.eps_grid=np.linspace(0,1,101); self.n=len(self.eps_grid); self.gamma=gamma
        self.on=DQN(n_obs,self.n); self.tg=DQN(n_obs,self.n); self.tg.load_state_dict(self.on.state_dict())
        self.opt=torch.optim.AdamW(self.on.parameters(),lr=lr); self.mem=Mem(200000)
        self.scale=torch.tensor(scale,dtype=torch.float32)
    def act_batch(self, S, nu):
        S=np.asarray(S,np.float32); m=len(S); out=np.empty(m,np.int64); ex=np.random.rand(m)<nu
        out[ex]=np.random.randint(self.n,size=ex.sum())
        if (~ex).any():
            with torch.no_grad():
                out[~ex]=self.on(torch.tensor(S[~ex])/self.scale).argmax(1).numpy()
        return out
    def learn(self, bs=128):
        if len(self.mem)<bs: return
        b=self.mem.sample(bs)
        s=torch.tensor(np.array(b.s),dtype=torch.float32)/self.scale
        s2=torch.tensor(np.array(b.s2),dtype=torch.float32)/self.scale
        r=torch.tensor(b.r,dtype=torch.float32); a=torch.tensor(b.a,dtype=torch.long)
        q=self.on(s).gather(1,a.unsqueeze(1)).squeeze(1)
        with torch.no_grad(): tgt=r+self.gamma*self.tg(s2).max(1).values
        loss=torch.nn.MSELoss()(q,tgt); self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_value_(self.on.parameters(),100); self.opt.step()
    def sync(self): self.tg.load_state_dict(self.on.state_dict())

def train(mode, T=20000, gamma=0.9, seed=0, N=300, cost=50.0, xbar=150.0, K=500):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    belief=ProducerBelief(xbar)
    if mode=="individual":
        agent=Agent(2, gamma, [xbar,xbar])      # state (own W, price)
    else:  # cooperative: state = price only, one bar_eps choice
        agent=Agent(1, gamma, [xbar])
    nu0,numin,decay=1.0,0.05,0.9998
    price=belief.best_price(cost)
    for t in range(T):
        nu=max(numin,nu0*decay**t)
        W=F_sampler(N,xbar)
        if mode=="individual":
            states=np.stack([W,np.full(N,price)],1)
            a=agent.act_batch(states,nu); eps_i=agent.eps_grid[a]; bar=eps_i.mean()
        else:
            a=agent.act_batch([[price]],nu)[0]; bar=agent.eps_grid[a]; eps_i=np.full(N,bar)
        p=accept_prob(W,price,eps_i); acc=(np.random.rand(N)<p).astype(float)
        belief.update(price, acc.mean()); nxt=belief.best_price(cost)
        if mode=="individual":
            r=(W-price)*acc
            idx=np.random.choice(N,size=min(32,N),replace=False)
            for i in idx: agent.mem.push((W[i],price),int(a[i]),float(r[i]),(W[i],nxt))
        else:
            r_agg=((W-price)*acc).mean()           # AGGREGATE surplus -> internalises externality
            agent.mem.push((price,),int(a),float(r_agg),(nxt,))
        agent.learn(128)
        if t%K==0: agent.sync()
        price=nxt
    # final greedy bar_eps
    Wb=F_sampler(20000,xbar)
    if mode=="individual":
        st=torch.tensor(np.stack([Wb,np.full(len(Wb),price)],1),dtype=torch.float32)/agent.scale
        with torch.no_grad(): idx=agent.on(st).argmax(1).numpy()
        return agent.eps_grid[idx].mean()
    else:
        st=torch.tensor([[price]],dtype=torch.float32)/agent.scale
        with torch.no_grad(): idx=agent.on(st).argmax(1).item()
        return agent.eps_grid[idx]

if __name__=="__main__":
    betas=[0.0,0.5,0.9,0.99]; SEEDS=8
    print(f"Externality: INDIVIDUAL vs COOPERATIVE consumers, {SEEDS} seeds, T=20000.\n")
    print(f"{'beta':>5} | {'individual':>22} | {'cooperative':>22} | {'gap (free-rider loss)':>22}")
    for beta in betas:
        ind=[train('individual',gamma=beta,seed=s) for s in range(SEEDS)]
        coop=[train('cooperative',gamma=beta,seed=s) for s in range(SEEDS)]
        gi,gc=np.mean(ind),np.mean(coop)
        print(f"{beta:>5} | {gi:.3f} +/- {np.std(ind):.3f}        | {gc:.3f} +/- {np.std(coop):.3f}        | {gi-gc:+.3f}", flush=True)
