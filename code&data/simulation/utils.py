import argparse
import os
import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import pickle as pkl
import pathlib
from modules import dataset, Net_s
import rpy2.robjects as ro # type: ignore
from rpy2.robjects.packages import importr # type: ignore
from rpy2.robjects import numpy2ri # type: ignore
import argparse
import os
import shutil
import yaml
from sklearn.linear_model import LinearRegression # type: ignore
from scipy.stats import ortho_group # type: ignore


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--dim', type=int, default=30, help='The dimension of the feature')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to the config file')
    parser.add_argument('--fig_config', type=str, default='config.yml', help='Path to the config file of the figure')
    parser.add_argument('--seed', type=str, default='1234', help='Random seed, N means no seed.')
    parser.add_argument('--sample_size', type=int, default=300, help='Size of the training set')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--type', type=str, default='ln', help='The type of the link functions: ln | nln_m | nln_r')
    parser.add_argument('--dist', type=str, default='MVN', help='Distribution: MVN | MHYP | MVT')
    parser.add_argument('--dev', type=str, default='cpu', help='Device: cpu | cuda:0 | ......')
    parser.add_argument('--init', type=int, default=300, help='The initial number of samples')
    parser.add_argument('--clean_par', type=str, default='N', help='Whether to clean the parameter folder: Y | N')
    parser.add_argument('--clean_out', type=str, default='Y', help='Whether to clean the outcome folder: Y | N')
    parser.add_argument('--gen_comb', type=str, default='Y', help='Whether generate a combination of link functions in case 3: Y | N')
    parser.add_argument('--nln_r_comb_dir', type=str, default='configs/nln_r', help='direction to save the combination of the link functions in case 3')
    args = parser.parse_args()

    dir =  os.path.join(os.getcwd(), 'configs', args.type, args.config)
   
    with open(dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    new_config = dict2namespace(config)

    return args, new_config

    

def gen_par(p,m,b,sd):
    Q = ortho_group.rvs(dim = p,)
        #diag_LAMBDA = np.diag(np.abs(np.random.normal(0,1,p)))
    diag_LAMBDA = np.diag(b + np.abs(np.random.normal(0,sd,p)))
        #diag_LAMBDA = np.diag(np.ones(p))
    Sigma = Q @ diag_LAMBDA @ Q.T
    mu = np.ones(p) * m
    return mu, Sigma




def gen_fet(dist, n, par):

    if dist == 'MVN':
        feature = np.random.multivariate_normal(par[0], par[1],n)
        feature = torch.from_numpy(feature).to(torch.float)


    return feature        

def f_0(prod,w):
    return torch.matmul(torch.sin(prod-1),w)  
def f_1(prod,w):   
    return torch.matmul(torch.cosh(prod-1) ,w)
def f_2(prod,w):
    return torch.matmul(torch.cos(prod-1),w)
def f_3(prod,w):
    return torch.matmul(torch.tanh(prod-1),w)
def f_4(prod,w):
    return torch.matmul(torch.arctan(prod-1),w)   
def f_5(prod,w):
    return torch.matmul((prod-1)**3,w)
def f_6(prod,w):  
    return torch.matmul((prod-1)**5,w)
def f_7(prod,w):
    return torch.matmul(torch.log(1 + torch.exp(prod)),w)  
def f_8(prod,w):
    return torch.matmul(((prod-1)**2 + 1)**0.5,w)  
# 2. Polynomials 
def f_9(prod,w):
    return torch.matmul(torch.exp(prod),w) 
def f_01(prod,w):
    return torch.matmul(torch.sin(prod-1) + torch.cosh(prod-1),w)
def f_12(prod,w):
    return torch.matmul(torch.cosh(prod-1) + torch.cos(prod-1),w)
def f_23(prod,w):
    return torch.matmul(torch.cos(prod-1) + torch.tanh(prod-1),w)
# Polynomials 
def f_34(prod,w):
    return torch.matmul(torch.tanh(prod-1) + torch.arctan(prod-1),w)
def f_45(prod,w):
    return torch.matmul(torch.arctan(prod-1) + (prod-1)**3,w)
def f_56(prod,w):
    return torch.matmul((prod-1)**3 + (prod-1)**5,w)
def f_67(prod,w):
    return torch.matmul((prod-1)**5 + torch.log(1 + torch.exp(prod)),w)
def f_78(prod,w):
    return torch.matmul(torch.log(1 + torch.exp(prod)) + ((prod-1)**2 + 1)**0.5,w)
def f_89(prod,w):
    return torch.matmul(((prod-1)**2 + 1)**0.5 + torch.exp(prod),w)
def f_90(prod,w):
    return torch.matmul(torch.sin(prod-1) + torch.exp(prod),w)
def gen_res(prod,W):   
    return torch.vstack((f_0(prod,W[0,:]),f_1(prod,W[1,:]), f_2(prod,W[2,:]), f_3(prod,W[3,:]), f_4(prod,W[4,:]), f_5(prod,W[5,:]), f_6(prod,W[6,:]), f_7(prod,W[7,:]), f_8(prod,W[8,:]), f_9(prod,W[9,:]),f_01(prod,W[10,:]),f_12(prod,W[11,:]),f_23(prod,W[12,:]),f_34(prod,W[13,:]),f_45(prod,W[14,:]),f_56(prod,W[15,:]),f_67(prod,W[16,:]),f_78(prod,W[17,:]),f_89(prod,W[18,:]),f_90(prod,W[19,:]))).T    




 
def gen_comb(dir):
    dir_f = os.path.join(os.getcwd(), dir, 'par_nlf.pkl')
    combs = []
    for i in np.arange(10):
        combs.append(np.random.choice(a=10, size=2, replace=True, p=None))
    with open(dir_f, 'wb') as f: 
        pkl.dump(combs, f)
    f.close()
    return 0




def g_0(prod):
    return torch.sin(prod-1)  
def g_1(prod):   
    return torch.cosh(prod-1)
def g_2(prod):
    return torch.cos(prod-1) 
def g_3(prod):
    return torch.tanh(prod-1)
def g_4(prod):
    return torch.arctan(prod-1)   
def g_5(prod):
    return (prod-1)**3
def g_6(prod):  
    return (prod-1)**5
def g_7(prod):
    return torch.log(1 + torch.exp(prod))  
def g_8(prod):
    return ((prod-1)**2 + 1)**0.5  
# 2. Polynomials 
def g_9(prod):
    return torch.exp(prod)

def gen_res_r(prod, W, combs):
    
    features = torch.zeros_like(prod[:,0])
    f = []
    for i in np.arange(10):
        f.append(eval('g_' + '{}'.format(i)))
        features = torch.vstack([features, torch.matmul(f[i](prod),W[i,:])])

    for i in np.arange(10):

        comb = combs[i]
        
        feature = f[comb[0]](prod) + f[comb[1]](prod)        
        feature =  torch.matmul(feature,W[(10+i),:])
        features = torch.vstack([features, feature])
        
    features = (features[1:,:]).T
    
    return features  


def gen_score(dist, feature, par):

    if dist == 'MVN':       
        score =  par[1] @ (feature-par[0])
    if dist == 'MVT':
        p = len(feature)
        nu = (feature - par[0]).reshape(-1,1)
        score = (p + par[2])/(par[2] - 2 +  nu.T @ par[1] @ nu)* torch.matmul(par[1], feature - par[0])       

    if dist == 'MHYP':
        a = feature - par[0]
        score = par[1] @ a 
        a = a.reshape(-1,1)
        score = score / ((par[2] + a.T @ par[1] @ a) ** 0.5) * (par[3] ** 0.5)

    return score  


def gen_s_score(dist, feature, par):

    
    if dist == 'MVN':   
        a = (par[1] @ (feature-par[0])).reshape(-1,1)   
        score = a @ a.T - par[1]
    if dist == 'MVT':
        p = len(feature)
        
        nu = (feature-par[0]).reshape(-1,1)
        nnu = par[1] @ nu
        cons = par[2] - 2 + nu.T @ par[1] @ nu   
        score = ((p + par[2])*(p + par[2] + 2)*(nnu @ nnu.T) - (p + par[2]) * cons * par[1])/(cons**2)

    if dist == 'MHYP':
        a = feature - par[0]
        a = a.reshape(-1,1)

        c1 = par[3] / (par[2] + a.T @ par[1] @ a)
        c3 = c1 ** 0.5
        c2 = (par[3] ** 0.5) / ((par[2] + a.T @ par[1] @ a) ** 1.5)
        
        score = (c1 + c2) * (par[1] @ a @ a.T @ par[1]) - c3 * par[1] 

    return score  





# Find B* = argmin_B |Y - XB|²
def solve_ols(X, Y):
    return np.linalg.pinv(X.T@X)@X.T@Y
def solve_rrr(X, Y, rank):
    
    # get OLS solution
    B_ols = solve_ols(X, Y)
    
    # get OLS estimate
    Y_ols = X@B_ols
    
    # get singular values and vectors of Y_ols
    _, _, V = np.linalg.svd(Y_ols)
    
    # construct P_r
    P_r = np.sum([np.outer(V[i], V[i]) for i in range(rank)], axis=0)
    return torch.from_numpy(B_ols@P_r).to(torch.float)


def reg_distance(A, B):
    model = LinearRegression(fit_intercept=False)
    model.fit(B, A)
    H = model.coef_.T
    loss = np.linalg.norm(A - B.dot(H)) 
   
    return loss




def set_seed(seed=666):
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ln(args, config):
    dist = args.dist
    dir_outcome = os.path.join(os.getcwd(), 'outcomes', args.type, dist)
    if args.clean_out == 'Y' and os.path.exists(dir_outcome):
        shutil.rmtree(dir_outcome)

    dir_par = os.path.join(os.getcwd(), 'parameters', args.type, dist)
    if args.clean_par == 'Y' and os.path.exists(dir_par):
        shutil.rmtree(dir_par)



    pathlib.Path(dir_outcome).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(dir_par).mkdir(parents=True, exist_ok=True)

    if args.seed != 'N':
        set_seed(int(args.seed)) 
    p = args.dim
    q = config.q
    k = config.k
    r = config.r

    n = args.sample_size


    dev = args.dev 
    device = torch.device(dev)
    batch_size = int(config.batch_ratio * n)

    dir_par_val = os.path.join(dir_par, 'pars_' + str(p) + '.pkl')


    if n == args.init:
        par_sampler = torch.distributions.normal.Normal(torch.zeros(1), torch.tensor(config.sdb))
        B = par_sampler.sample(torch.tensor((p*q,))).reshape(p,q)
        B, _, _ = torch.linalg.svd(B)
        B = B[:,0:k]
        normal_func = torch.distributions.normal.Normal(torch.zeros(1), torch.tensor(config.sdf))
        O = normal_func.sample(torch.tensor((k*q,))).reshape(k,q)
        mu, Sigma = gen_par(p,config.m,config.b,config.sd)
        Pars = {'B':B, 'O':O, 'par':[mu, Sigma]}
        with open(dir_par_val, 'wb') as f:  
            pkl.dump(Pars, f)
        f.close()
    else: 
        with open(dir_par_val,'rb') as f:
            Pars = pkl.load(f)
        f.close() 

        B = Pars['B']
        O = Pars['O']
        mu, Sigma = Pars['par']


    B_n = B.numpy()

    

    mse = nn.MSELoss()


    Dis_s = torch.zeros(r)
    Dis_n = torch.zeros(r)
    Dis_r = torch.zeros(r)

    normal_eps = torch.distributions.normal.Normal(torch.zeros(1), torch.tensor(config.sde))
    for t in torch.arange(r):
    
        eps = normal_eps.sample(torch.tensor((n*q,))).reshape(n,q)

        if dist == 'MVN':
    
            feature = gen_fet(dist, n, [mu, Sigma])
    
        if dist == 'MVT':

            ghyp = importr('ghyp')

            n1, c1 = ro.vectors.IntVector([n, config.nv])
  
            numpy2ri.activate()

            loc = ro.vectors.FloatVector(mu)

            sigma= numpy2ri.py2rpy(Sigma)

            mvt = ro.r['student.t']


            mvt_ob = mvt(nu=c1, mu = loc, sigma=sigma)
        
            feature = ghyp.rghyp(n = n1, object=mvt_ob)
            fit_tmv = ro.r['fit.tmv']

            fitted = fit_tmv(feature, nu=c1, mu=loc, sigma=sigma, symmetric=True)

            fitted = ghyp.coef(fitted, "chi.psi")

            est_loc = fitted.rx2('mu')
            est_loc = torch.from_numpy(est_loc).to(torch.float32).to(device)
            est_scale = fitted.rx2('sigma')
            est_scale = torch.from_numpy(est_scale).to(torch.float32).to(device)
            inv_est_scale = torch.linalg.inv(est_scale)

            est_nu = fitted.rx2('lambda')[0]
            est_nu = -2*est_nu
        
  

            feature = torch.from_numpy(feature).to(torch.float32)        
  
        if dist == 'MHYP':
        
            ghyp = importr('ghyp')

            n1, c1, c2 = ro.vectors.IntVector([n, 2*p+1, p])
  
            numpy2ri.activate()

            loc = ro.vectors.FloatVector(mu)

            sigma= numpy2ri.py2rpy(Sigma)
            hyp = ghyp.hyp(chi = c1, psi = c2, mu = loc,  sigma = Sigma)
        
            feature = ghyp.rghyp(n = n1, object=hyp)
            fit_hypmv = ro.r['fit.hypmv']

            fitted = fit_hypmv(feature, symmetric = True)

            fitted = ghyp.coef(fitted, "chi.psi")

            est_loc = fitted.rx2('mu')
            est_loc = torch.from_numpy(est_loc).to(torch.float32).to(device)
            est_scale = fitted.rx2('sigma')
            est_scale = torch.from_numpy(est_scale).to(torch.float32).to(device)
            inv_est_scale = torch.linalg.inv(est_scale)

            est_chi = fitted.rx2('chi')[0]
            est_psi = fitted.rx2('psi')[0]


            feature = torch.from_numpy(feature).to(torch.float32)


        label = torch.matmul(torch.matmul(feature, B), O) 
    
        label = (label - torch.mean(label, dim=0))/torch.std(label, dim=0)
    
        label = label + eps  

   
        train_set = dataset(feature, label)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        model = Net_s(k,q,p,config.h)
        optimizer = optim.Adam(model.parameters())

        model = model.to(device)

        model.train(True)

        epoch_number = 0

        for _ in range(config.EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))
        
            for data in train_loader:
        
                features, labels = data
                features = features.to(device) 
                labels = labels.to(device) 

                optimizer.zero_grad()

                outputs = model(features)

                loss = mse(outputs, labels)
                loss.backward()
       
                optimizer.step()

            epoch_number += 1
     
        esB = (torch.transpose((list(model.parameters())[0]),0,1)).detach().cpu().numpy()
    
        dis = reg_distance(B_n, esB)

        dis = dis/(k ** 0.5)
        Dis_n[t] = dis
        print('NN DIS {}'.format(dis))


        esB = solve_rrr(feature.numpy(), label.numpy(), k)

        esB, _, _ = np.linalg.svd(esB)

        esB = esB[:,0:k]

        dis = reg_distance(B_n, esB)
        dis = dis / (k ** 0.5)
        Dis_r[t] = dis
    
   
        print('rr DIS {}'.format(dis))
    
    

        esB =  torch.zeros(p*q).reshape(p,q).to(device)
        feature = feature.to(device)
        label = label.to(device)

        if dist == 'MVT':                        
            for i in torch.arange(n):
                S = gen_score(dist,feature[i,:], [est_loc, inv_est_scale, est_nu])  
                esB += torch.kron(label[i,:].reshape(1,q), S.reshape(p,1))         
#           
        if dist == 'MVN':
            sample_mean = torch.mean(feature, dim=0)
            inv_sample_cov = torch.linalg.inv(torch.cov(feature.T))     
            for i in torch.arange(n):
                S = gen_score(dist,feature[i,:], [sample_mean, inv_sample_cov])
                esB += torch.kron(label[i,:].reshape(1,q), S.reshape(p,1))

        if dist == 'MHYP':
            for i in torch.arange(n):
                S = gen_score(dist,feature[i,:], [est_loc, inv_est_scale, est_chi, est_psi])
                esB += torch.kron(label[i,:].reshape(1,q), S.reshape(p,1))

        esB = esB/n
        esB, L, _ = torch.linalg.svd(esB)
        esB = (esB[:,0:k]).cpu().numpy()


        dis = reg_distance(B_n, esB)
        dis = dis / (k ** 0.5)
        Dis_s[t] = dis

        print('S DIS {}'.format(dis))



        print('Times {}'.format(t))

        dir_out_val = os.path.join(dir_outcome, 'outcome_' + str(p) + '_'+ str(n) + '.pkl')
  
        output = {'Dis_NN':Dis_n, 'Dis_S':Dis_s, 'Dis_RR':Dis_r}
        with open(dir_out_val, 'wb') as f: 
            pkl.dump(output, f)
        f.close()  
    return 0  

def nln_m(args, config):
    dist = args.dist
    dir_outcome = os.path.join(os.getcwd(), 'outcomes', args.type, dist)
    if args.clean_out == 'Y' and os.path.exists(dir_outcome):
        shutil.rmtree(dir_outcome)

    dir_par = os.path.join(os.getcwd(), 'parameters', args.type, dist)
    if args.clean_par == 'Y' and os.path.exists(dir_par):
        shutil.rmtree(dir_par)



    pathlib.Path(dir_outcome).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(dir_par).mkdir(parents=True, exist_ok=True)

    if args.seed != 'N':
        set_seed(int(args.seed)) 

    p = args.dim
    q = config.q
    k = config.k
    r = config.r

    n = args.sample_size


    dev = args.dev 
    device = torch.device(dev)
    batch_size = int(config.batch_ratio * n)

    dir_par_val = os.path.join(dir_par, 'pars_' + str(p) + '.pkl')



    if n == args.init:

        weight_sampler = torch.distributions.normal.Normal(torch.zeros(1), torch.tensor(config.sdw))
        W = config.bw + torch.abs(weight_sampler.sample(torch.tensor((k*q,))).reshape(q,k))

        par_sampler = torch.distributions.normal.Normal(torch.zeros(1), torch.tensor(config.sdp))
        B = par_sampler.sample(torch.tensor((p*q,))).reshape(p,q)
        B, _, _ = torch.linalg.svd(B)
        B = B[:,0:k]
        mu, Sigma = gen_par(p,config.m,config.b,config.sd)

        Pars = {'W':W, 'B':B, 'par':[mu, Sigma]}
    
        with open(dir_par_val, 'wb') as f:  
            pkl.dump(Pars, f)
        f.close()
    else: 
        with open(dir_par_val,'rb') as f:
            Pars = pkl.load(f)
        f.close()  
        W = Pars['W']
        B = Pars['B']
        mu, Sigma = Pars['par']

    B_n = B.cpu().numpy()
    normal_eps = torch.distributions.normal.Normal(torch.zeros(1), torch.tensor(config.sde))
    mse = nn.MSELoss()


    Dis_r  = torch.zeros(r)
    Dis_s = torch.zeros(r)
    Dis_n = torch.zeros(r)
    Dis_1 = torch.zeros(r)

    for t in torch.arange(r):
     
        eps = normal_eps.sample(torch.tensor((n*q,))).reshape(n,q)
 

        if dist == 'MVN':
    
            feature = gen_fet(dist, n, [mu, Sigma])

        if dist == 'MVT':

            ghyp = importr('ghyp')

            n1, c1 = ro.vectors.IntVector([n, config.nv])
  
            numpy2ri.activate()

            loc = ro.vectors.FloatVector(mu)

            sigma= numpy2ri.py2rpy(Sigma)

            mvt = ro.r['student.t']


            mvt_ob = mvt(nu=c1, mu = loc, sigma=sigma)
        
            feature = ghyp.rghyp(n = n1, object=mvt_ob)
            fit_tmv = ro.r['fit.tmv']

            fitted = fit_tmv(feature, nu=c1, mu=loc, sigma=sigma, symmetric=True)

            fitted = ghyp.coef(fitted, "chi.psi")

            est_loc = fitted.rx2('mu')
            est_loc = torch.from_numpy(est_loc).to(torch.float32).to(device)
            est_scale = fitted.rx2('sigma')
            est_scale = torch.from_numpy(est_scale).to(torch.float32).to(device)
            inv_est_scale = torch.linalg.inv(est_scale)

            est_nu = fitted.rx2('lambda')[0]
            est_nu = -2*est_nu
        
  
      

            feature = torch.from_numpy(feature).to(torch.float32)        
  

        if dist == 'MHYP':
        
            ghyp = importr('ghyp')

            n1, c1, c2 = ro.vectors.IntVector([n, 2*p+1, p])
  
            numpy2ri.activate()

            loc = ro.vectors.FloatVector(mu)

            sigma= numpy2ri.py2rpy(Sigma)
            hyp = ghyp.hyp(chi = c1, psi = c2, mu = loc,  sigma = Sigma)
        
            feature = ghyp.rghyp(n = n1, object=hyp)
            fit_hypmv = ro.r['fit.hypmv']

            fitted = fit_hypmv(feature, symmetric = True)

            fitted = ghyp.coef(fitted, "chi.psi")

            est_loc = fitted.rx2('mu')
            est_loc = torch.from_numpy(est_loc).to(torch.float32).to(device)
            est_scale = fitted.rx2('sigma')
            est_scale = torch.from_numpy(est_scale).to(torch.float32).to(device)
            inv_est_scale = torch.linalg.inv(est_scale)

            est_chi = fitted.rx2('chi')[0]
            est_psi = fitted.rx2('psi')[0]

       

            feature = torch.from_numpy(feature).to(torch.float32)


        label = gen_res(torch.matmul(feature, B), W) 
    
        label = (label - torch.mean(label, dim=0))/torch.std(label, dim=0)
    
        label = label + eps  





        feature = feature.to(device)
    

        sample_mean = torch.mean(feature,dim=0)

        label = label.to(device)

        train_set = dataset(feature, label)


        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

        model = Net_s(k,q,p,config.h).to(device)

        optimizer = optim.Adam(model.parameters())

        model.train(True)


        epoch_number = 0

        for _ in range(config.EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))
        
            for data in train_loader:
        
                features, labels = data
                features = features.to(device) 
                labels = labels.to(device) 

                optimizer.zero_grad()

                outputs = model(features)

                loss = mse(outputs, labels)
                loss.backward()
       
                optimizer.step()

            epoch_number += 1
     
        esB = (torch.transpose((list(model.parameters())[0]),0,1)).detach().cpu().numpy()
        
        dis = reg_distance(B_n, esB)
        dis = dis / (k ** 0.5)
        Dis_n[t] = dis
        print('NN DIS {}'.format(dis))






        esB = solve_rrr(feature.cpu().numpy(), label.cpu().numpy(), k).to(device)

        esB, _, _ = torch.linalg.svd(esB)

        esB = (esB[:,0:k]).cpu().numpy()
        dis = reg_distance(B_n, esB)
        dis = dis / (k **0.5)
        Dis_r[t] = dis
        print('rr DIS {}'.format(dis))
    
    
        esB =  torch.zeros(p*q).reshape(p,q).to(device)
        if dist == 'MVT':                        
            for i in torch.arange(n):
                S = gen_score(dist,feature[i,:], [est_loc, inv_est_scale, est_nu])  
                esB += torch.kron(label[i,:].reshape(1,q), S.reshape(p,1))         
#           
        if dist == 'MVN':
            inv_sample_cov = torch.linalg.inv(torch.cov(feature.T))     
            for i in torch.arange(n):
                S = gen_score(dist,feature[i,:], [sample_mean, inv_sample_cov])
                esB += torch.kron(label[i,:].reshape(1,q), S.reshape(p,1))

        if dist == 'MHYP':
            for i in torch.arange(n):
                S = gen_score(dist,feature[i,:], [est_loc, inv_est_scale, est_chi, est_psi])
                esB += torch.kron(label[i,:].reshape(1,q), S.reshape(p,1))

        esB = esB/n
        esB, _, _ = torch.linalg.svd(esB)
        esB = (esB[:,0:k]).cpu().numpy()


        dis = reg_distance(B_n, esB)
        dis = dis / (k ** 0.5)
        Dis_s[t] = dis     
    
        print('S DIS {}'.format(dis))


        esB =  torch.zeros(p*p).reshape(p,p).to(device)
        if dist == 'MVT':                 
            for i in torch.arange(n):
                S = gen_s_score(dist,feature[i,:], [est_loc, inv_est_scale, est_nu])
                y = label[i,:].reshape(-1,1,1)
                esB += torch.sum((y*S),dim=0) 

        if dist == 'MVN':
            inv_sample_cov = torch.linalg.inv((torch.cov(feature.T))).to(device)       
            for i in torch.arange(n):
                S = gen_s_score(dist,feature[i,:], [sample_mean, inv_sample_cov])
                y = label[i,:].reshape(-1,1,1)
                esB += torch.sum((y*S),dim=0) 

        if dist == 'MHYP':   
            for i in torch.arange(n):
                S = gen_s_score(dist,feature[i,:], [est_loc, inv_est_scale, est_chi, est_psi])  
                y = label[i,:].reshape(-1,1,1) 
                esB += torch.sum( (y*S),dim=0)

        esB = esB/n/q

        esB, _, _ = torch.linalg.svd(esB)
        esB = esB[:,0:k].cpu().numpy()
        dis = reg_distance(B_n, esB)
        dis = dis/(k ** 0.5) 
        Dis_1[t] = dis 
        print('Original {}'.format(dis))

        print('Times {}'.format(t)) 
 

        dir_out_val = os.path.join(dir_outcome, 'outcome_' + str(p) + '_'+ str(n) + '.pkl')
  
        output = {'Dis_r':Dis_r, 'Dis_s':Dis_s, 'Dis_n':Dis_n,'Dis_1':Dis_1}
        with open(dir_out_val, 'wb') as f: 
            pkl.dump(output, f)
        f.close()  
    return 0  


    
def nln_r(args, config):
    dist = args.dist
    dir_outcome = os.path.join(os.getcwd(), 'outcomes', args.type, dist)
    if args.clean_out == 'Y' and os.path.exists(dir_outcome):
        shutil.rmtree(dir_outcome)

    dir_par = os.path.join(os.getcwd(), 'parameters', args.type, dist)
    if args.clean_par == 'Y' and os.path.exists(dir_par):
        shutil.rmtree(dir_par)



    pathlib.Path(dir_outcome).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(dir_par).mkdir(parents=True, exist_ok=True)

    if args.seed != 'N':
        set_seed(int(args.seed)) 

    p = args.dim
    q = config.q
    k = config.k
    r = config.r

    n = args.sample_size


    dev = args.dev 
    device = torch.device(dev)
    batch_size = int(config.batch_ratio * n)

    dir_par_val = os.path.join(dir_par, 'pars_' + str(p) + '.pkl')





    if n == args.init:

        weight_sampler = torch.distributions.normal.Normal(torch.zeros(1), torch.tensor(config.sdw))
        W = config.bw + torch.abs(weight_sampler.sample(torch.tensor((k*q,))).reshape(q,k))

        par_sampler = torch.distributions.normal.Normal(torch.zeros(1), torch.tensor(config.sdp))
        B = par_sampler.sample(torch.tensor((p*q,))).reshape(p,q)
        B, _, _ = torch.linalg.svd(B)
        B = B[:,0:k]
        mu, Sigma = gen_par(p,config.m,config.b,config.sd)

        Pars = {'W':W, 'B':B, 'par':[mu, Sigma]}
    
        with open(dir_par_val, 'wb') as f:  
            pkl.dump(Pars, f)
        f.close()
    else: 
        with open(dir_par_val,'rb') as f:
            Pars = pkl.load(f)
        f.close()  
        W = Pars['W']
        B = Pars['B']
        mu, Sigma = Pars['par']

    B_n = B.cpu().numpy()
    normal_eps = torch.distributions.normal.Normal(torch.zeros(1), torch.tensor(config.sde))
    mse = nn.MSELoss()



    dir_comb = os.path.join(os.getcwd(), args.nln_r_comb_dir, 'par_nlf.pkl')


    with open(dir_comb,'rb') as f:
        combs = pkl.load(f)
    f.close() 


    Dis_r  = torch.zeros(r)
    Dis_s = torch.zeros(r)
    Dis_n = torch.zeros(r)
    Dis_1 = torch.zeros(r)



    for t in torch.arange(r):
     
        eps = normal_eps.sample(torch.tensor((n*q,))).reshape(n,q)
 

        if dist == 'MVN':
    
            feature = gen_fet(dist, n, [mu, Sigma])
    
        if dist == 'MVT':

            ghyp = importr('ghyp')

            n1, c1 = ro.vectors.IntVector([n, config.nv])
  
            numpy2ri.activate()

            loc = ro.vectors.FloatVector(mu)

            sigma= numpy2ri.py2rpy(Sigma)

            mvt = ro.r['student.t']


            mvt_ob = mvt(nu=c1, mu = loc, sigma=sigma)
        
            feature = ghyp.rghyp(n = n1, object=mvt_ob)
            fit_tmv = ro.r['fit.tmv']

            fitted = fit_tmv(feature, nu=c1, mu=loc, sigma=sigma, symmetric=True)

            fitted = ghyp.coef(fitted, "chi.psi")

            est_loc = fitted.rx2('mu')
            est_loc = torch.from_numpy(est_loc).to(torch.float32).to(device)
            est_scale = fitted.rx2('sigma')
            est_scale = torch.from_numpy(est_scale).to(torch.float32).to(device)
            inv_est_scale = torch.linalg.inv(est_scale)

            est_nu = fitted.rx2('lambda')[0]
            est_nu = -2*est_nu
        
  
 

            feature = torch.from_numpy(feature).to(torch.float32)        
  
        if dist == 'MHYP':
        
            ghyp = importr('ghyp')

            n1, c1, c2 = ro.vectors.IntVector([n, 2*p+1, p])
  
            numpy2ri.activate()

            loc = ro.vectors.FloatVector(mu)

            sigma= numpy2ri.py2rpy(Sigma)
            hyp = ghyp.hyp(chi = c1, psi = c2, mu = loc,  sigma = Sigma)
        
            feature = ghyp.rghyp(n = n1, object=hyp)
            fit_hypmv = ro.r['fit.hypmv']

            fitted = fit_hypmv(feature, symmetric = True)

            fitted = ghyp.coef(fitted, "chi.psi")

            est_loc = fitted.rx2('mu')
            est_loc = torch.from_numpy(est_loc).to(torch.float32).to(device)
            est_scale = fitted.rx2('sigma')
            est_scale = torch.from_numpy(est_scale).to(torch.float32).to(device)
            inv_est_scale = torch.linalg.inv(est_scale)

            est_chi = fitted.rx2('chi')[0]
            est_psi = fitted.rx2('psi')[0]

    

            feature = torch.from_numpy(feature).to(torch.float32)

        label = gen_res_r(torch.matmul(feature, B), W, combs) 


        label = (label - torch.mean(label, dim=0))/torch.std(label, dim=0)
    
        label = label + eps    


        feature = feature.to(device)
    

        sample_mean = torch.mean(feature,dim=0)

        label = label.to(device)

        train_set = dataset(feature, label)


        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

        model = Net_s(k,q,p,config.h).to(device)

        optimizer = optim.Adam(model.parameters())

        model.train(True)


        epoch_number = 0

        for _ in range(config.EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))
        
            for data in train_loader:
        
                features, labels = data
                features = features.to(device) 
                labels = labels.to(device) 

                optimizer.zero_grad()

                outputs = model(features)

                loss = mse(outputs, labels)
                loss.backward()
       
                optimizer.step()

            epoch_number += 1
     
        esB = (torch.transpose((list(model.parameters())[0]),0,1)).detach().cpu().numpy()
        
        dis = reg_distance(B_n, esB)

        dis = dis / (k ** 0.5)
        Dis_n[t] = dis
        print('NN DIS {}'.format(dis))






        esB = solve_rrr(feature.cpu().numpy(), label.cpu().numpy(), k).to(device)

        esB, _, _ = torch.linalg.svd(esB)

        esB = (esB[:,0:k]).cpu().numpy()
        dis = reg_distance(B_n, esB)
        dis = dis / (k **0.5)
        Dis_r[t] = dis
        print('rr DIS {}'.format(dis))
    
    
        esB =  torch.zeros(p*q).reshape(p,q).to(device)
        if dist == 'MVT':                        
            for i in torch.arange(n):
                S = gen_score(dist,feature[i,:], [est_loc, inv_est_scale, est_nu])  
                esB += torch.kron(label[i,:].reshape(1,q), S.reshape(p,1))         
#           
        if dist == 'MVN':
            inv_sample_cov = torch.linalg.inv(torch.cov(feature.T))     
            for i in torch.arange(n):
                S = gen_score(dist,feature[i,:], [sample_mean, inv_sample_cov])
                esB += torch.kron(label[i,:].reshape(1,q), S.reshape(p,1))

        if dist == 'MHYP':
            for i in torch.arange(n):
                S = gen_score(dist,feature[i,:], [est_loc, inv_est_scale, est_chi, est_psi])
                esB += torch.kron(label[i,:].reshape(1,q), S.reshape(p,1))

        esB = esB/n
        esB, L, _ = torch.linalg.svd(esB)
        esB = (esB[:,0:k]).cpu().numpy()


        dis = reg_distance(B_n, esB)
        dis = dis / (k ** 0.5)
        Dis_s[t] = dis     
    
        print('S DIS {}'.format(dis))


        esB =  torch.zeros(p*p).reshape(p,p).to(device)
        if dist == 'MVT':                 
            for i in torch.arange(n):
                S = gen_s_score(dist,feature[i,:], [est_loc, inv_est_scale, est_nu])
                y = label[i,:].reshape(-1,1,1)
                esB += torch.sum((y*S),dim=0) 

        if dist == 'MVN':
            inv_sample_cov = torch.linalg.inv((torch.cov(feature.T))).to(device)       
            for i in torch.arange(n):
                S = gen_s_score(dist,feature[i,:], [sample_mean, inv_sample_cov])
                y = label[i,:].reshape(-1,1,1)
                esB += torch.sum((y*S),dim=0) 

        if dist == 'MHYP':   
            for i in torch.arange(n):
                S = gen_s_score(dist,feature[i,:], [est_loc, inv_est_scale, est_chi, est_psi])  
                y = label[i,:].reshape(-1,1,1) 
                esB += torch.sum( (y*S),dim=0)

        esB = esB/n/q

        esB, _, _ = torch.linalg.svd(esB)
        esB = esB[:,0:k].cpu().numpy()
        dis = reg_distance(B_n, esB)
        dis = dis/(k ** 0.5) 
        Dis_1[t] = dis 
        print('Original {}'.format(dis))

        print('Times {}'.format(t)) 

        dir_out_val = os.path.join(dir_outcome, 'outcome_' + str(p) + '_'+ str(n) + '.pkl')
  
        output = {'Dis_r':Dis_r, 'Dis_s':Dis_s, 'Dis_n':Dis_n,'Dis_1':Dis_1}
        with open(dir_out_val, 'wb') as f: 
            pkl.dump(output, f)
        f.close()  
    return 0  
