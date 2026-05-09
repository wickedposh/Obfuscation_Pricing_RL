import numpy as np
from scipy.stats import norm
q_table=np.random.randn(151,11)*0.01
T,mu,sigma,lr,delta,eta=10000,91,10,0.2,0.0001,0.2
betas=[0.5,0.6,0.7,0.8,0.9,0.99]
results={}
for beta in betas:
    t=0
    q_table=np.random.randn(151,11)*0.01
    mu,sigma=91,10
    eps_history=[]
    mu_history=[]
    sigma_history=[]
    done=False
    ca=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
    pa=[i for i in range(151)]
    while not done and t<T:
        q_old=q_table.copy()

        x=max(0,min(150,int(round(mu))))
        if np.random.rand()<eta:
            eind=np.random.randint(11)
        else:
            eind=int(np.argmax(q_table[int(x),:]))
        epsilon_star=ca[eind]
        eps_history.append(epsilon_star)
        if x<=100:
            c=epsilon_star+(1-epsilon_star)/2
            if np.random.rand()<c:
                u=(100-x)
                a=+1
            else:
                u=0
                a=-1
        else:
            c=(1-epsilon_star)/2
            if np.random.rand()<c:
                u=(100-x)
                a=+1
            else:
                u=0
                a=-1
        z=(x-mu)/sigma
        phi=norm.pdf(z)
        Phi=norm.cdf(z)
        if a==+1:
            denom=epsilon_star*(1-Phi)+(1-epsilon_star)/2
            numer=epsilon_star*(mu*(1-Phi)+sigma*phi)+(1-epsilon_star)*mu/2
            second=epsilon_star*((mu**2+sigma**2)*(1-Phi)+sigma*(mu+x)*phi)+(1-epsilon_star)*(mu**2+sigma**2)/2
        else:
            denom=epsilon_star*(Phi)+(1-epsilon_star)/2
            numer=epsilon_star*(mu*Phi-sigma*phi)+(1-epsilon_star)*mu/2
            second=epsilon_star*((mu**2+sigma**2)*(Phi)-sigma*(mu+x)*phi)+(1-epsilon_star)*(mu**2+sigma**2)/2

        mu=float(numer/denom)
        var=float(second/denom)-mu**2
        if var<=0 or np.isnan(var):

            var=1e-6
        sigma=np.sqrt(var)
        mu_history.append(mu)
        sigma_history.append(sigma)
        t+=1
        x_next_ind=max(0,min(150,int(round(mu))))
        next_max=np.max(q_table[int(x_next_ind),:])
        q_table[x,eind]+=lr*(u+beta*next_max-q_table[x,eind])

        if np.max(np.abs(q_table-q_old))<delta:
            done=True

    results[beta]={'eps_star_final':eps_history[-1],
                   'eps_star_avg_late':np.mean(eps_history[-20:]),
                   'mu_final':mu_history[-1],
                   'sigma_final':sigma_history[-1],
                   'eps_history':eps_history,
                   'sigma_history':sigma_history}

for beta,res in results.items():
    print(f"beta={beta} | res={res['eps_star_avg_late']:.2f},mu={res['mu_final']:.2f},sigma={res['sigma_final']:.2f}")