import numpy as np 
from kernels.curlfree_imq import CurlFreeIMQ
from estimators.nu_method import NuMethod 
import torch # type: ignore
import pickle as pkl
from sklearn import model_selection as ms # type: ignore
import numpy as np # type: ignore
import os
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
import argparse
import yaml

def set_seed(seed=666):
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




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
    parser.add_argument('--config', type=str, default='config.yml', help='Path to the config file')
    parser.add_argument('--data', type=str, default='data/processed_data.pickle', help='Path to the data file')
    parser.add_argument('--t', type=int, default=100, help='Times of experiments')
    parser.add_argument('--seed', type=str, default='1234', help='Random seed, N means no seed.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--dev', type=str, default='cpu', help='Device: cpu | cuda:0 | ......')
    parser.add_argument('--out_dir', type=str, default='outcome/output.pkl', help='Directory to save the output')
    parser.add_argument('--figure_dir', type=str, default='figure/seq.png', help='Directory to save the figure')    
    args = parser.parse_args()

    dir =  os.path.join(os.getcwd(),  args.config)
   
    with open(dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    new_config = dict2namespace(config)

    return args, new_config




def preprocess(data):
    X = data['counts'][:,data['mostVariableGenes']] / np.sum(data['counts'], axis=1) * 1e+6
    X = X.toarray()
    X = np.log2(X + 1)
    X = X - np.mean(X, axis=0)
    X = X / np.std(X, axis=0)

    Y = data['ephys']
    Y = Y - np.mean(Y, axis=0)
    Y = Y / np.std(Y, axis=0)
    
    return (X,Y)

def train_one_epoch(model, opt, loss_fun, training_loader):

    for i, data in enumerate(training_loader):
        print(i)
        # Every data instance is an input + label pair
        feature, label = data


        opt.zero_grad()
        outputs = model(feature)
       
        loss = loss_fun(outputs, label)
        loss.backward()
        opt.step()
    
    
    return model, opt



def M1(args, config):

    if args.seed != 'N':
        set_seed(int(args.seed)) 

    data_path = os.path.join(os.getcwd(), args.data)

    with open(data_path,'rb') as f:
        data = pkl.load(f)
    f.close() 
    feature, label = preprocess(data)
    print('Load data successfully!')
    p = feature.shape[1]
    q = label.shape[1]

    label_sizes = config.label_sizes
   
    t = args.t
    outdims = config.outdims

    mses = torch.zeros(t, len(label_sizes),len(outdims),4)

    for j in torch.arange(t):
    
        train_feature, test_feature, train_label, test_label = ms.train_test_split(feature, label, test_size=212.5/1213)

        test_feature = torch.from_numpy(test_feature).to(torch.float32)
        test_label = torch.from_numpy(test_label).to(torch.float32)

        n = len(train_label)
        n_tr = len(train_label)
        unlabel_feature, label_feature, unlabel_label, label_label = ms.train_test_split(train_feature, train_label, test_size=(label_sizes[-1])/n_tr)
        print(len(unlabel_label))
        l = n - label_sizes[-1]
        
        train_feature_g = torch.from_numpy(np.vstack((unlabel_feature, label_feature))).to(torch.float32)

        train_label_g = torch.from_numpy(np.vstack((unlabel_label, label_label))).to(torch.float32)
        nu_estimator = NuMethod(lam=config.lam, kernel=CurlFreeIMQ())
        nu_estimator.fit(train_feature_g)
        score = nu_estimator.compute_gradients(train_feature_g)
        for k, label_size in enumerate(label_sizes):
            label_1 = torch.hstack((train_feature_g[l:(l+label_size)], train_label_g[l:(l+label_size)]))
            B_a =  torch.zeros(p,p+q)
            for i in torch.arange(label_size): 
                B_a += torch.kron(label_1[i,:].reshape(1,p+q), score[(l+i),:].reshape(p,1))
            for i in torch.arange(l): 
                B_a[:,:p] += torch.kron(train_feature_g[i,:].reshape(1,p), score[i,:].reshape(p,1))

            B_a[:,:p] = B_a[:,:p]/(n)
            B_a[:,p:(p+q)] = B_a[:,p:(p+q)]/(label_size)
    
            B_l =  torch.zeros(p,q)
            for i in torch.arange(label_size): 
                B_l += torch.kron(train_label_g[(l+i),:].reshape(1,q), score[(l+i),:].reshape(p,1))
            B_l = B_l/label_size  

            B_u =  torch.zeros(p,p)
            for i in torch.arange(n): 
                B_u += torch.kron(train_feature_g[i,:].reshape(1,p), score[i,:].reshape(p,1))
            B_u = B_u/n 

            B_p = torch.cov(train_feature_g.T)
        
            B_a, _, _ = torch.linalg.svd(B_a)

            B_l, _, _ = torch.linalg.svd(B_l)

            B_u, _, _ = torch.linalg.svd(B_u)

            B_p, _, _ = torch.linalg.svd(B_p)

            for i, outdim in enumerate(outdims):
                U_a = (B_a[:,:outdim]).numpy()
                U_l = (B_l[:,:outdim]).numpy()
                U_u = (B_u[:,:outdim]).numpy()
        
                U_p = (B_p[:,:outdim]).numpy() 
     
                r_a = LinearRegression()
                r_a.fit(unlabel_feature @ U_a, unlabel_label)
                pred_a = r_a.predict(test_feature @ U_a)
                loss_a = mean_squared_error(pred_a, test_label)


                r_l = LinearRegression()
                r_l.fit(unlabel_feature @ U_l, unlabel_label)
                pred_l = r_l.predict(test_feature @ U_l)
                loss_l = mean_squared_error(pred_l, test_label)


                r_u = LinearRegression()
                r_u.fit(unlabel_feature @ U_u, unlabel_label)
                pred_u = r_u.predict(test_feature @ U_u)
                loss_u = mean_squared_error(pred_u, test_label)


                r_p = LinearRegression()
                r_p.fit(unlabel_feature @ U_p, unlabel_label)
                pred_p = r_p.predict(test_feature @ U_p)
                loss_p = mean_squared_error(pred_p, test_label)

                mses[j, k, i, 0] = loss_a
                mses[j, k, i, 1] = loss_l
                mses[j, k, i, 2] = loss_u
                mses[j, k, i, 3] = loss_p


                print(mses[j, k, i, :])                



                dir_out = os.path.join(os.getcwd(),args.out_dir)

                with open(dir_out, 'wb') as f: 
                    pkl.dump(mses, f)
                f.close()
    return 0