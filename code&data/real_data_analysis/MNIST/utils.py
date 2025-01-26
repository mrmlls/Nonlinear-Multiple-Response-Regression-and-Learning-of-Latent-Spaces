import os
import torch # type: ignore
from modules import data_set, AE, Decoder, Classifier,  CondRefineNetDilated
import yaml
import pickle as pkl
import copy
import torch.nn as nn # type: ignore
import pathlib
from torchmetrics.image import StructuralSimilarityIndexMeasure  # type: ignore
import gc
import torch.nn.functional as F # type: ignore
import shutil
import argparse
import numpy as np
import torchvision # type: ignore
import torchvision.transforms as transforms # type: ignore
#import matplotlib.pyplot as plt
 
def pre_process(dir_data):
    transform = transforms.Compose([transforms.ToTensor()])     
 
    train_set = torchvision.datasets.MNIST(root=dir_data, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=dir_data, train=False, download=True, transform=transform)

    n_tr,h,w = train_set.data.shape
    c=1
    n_te = test_set.data.shape[0] 
    n_cl = len(train_set.classes) 
 
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=n_tr, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=n_te, shuffle=False)
    for data in train_loader:
        train_feature, train_label = data
    for data in test_loader:
        test_feature, test_label = data

    train_features = torch.zeros(n_tr,c*h*w)
    train_labels = torch.zeros(n_tr)
    test_features = torch.zeros(n_te, c*h*w)
    test_labels = torch.zeros(n_te)

    for i in torch.arange(n_tr):
        if i < n_te:
            test_features[i,:] = (test_feature[i]).reshape(-1)
            test_labels[i] = (test_label[i])
        train_features[i,:] = (train_feature[i]).reshape(-1)
        train_labels[i] = (train_label[i])

        
    train_feature = torch.zeros(n_tr,c*h*w)
    train_label = torch.zeros(n_tr)
    test_feature = torch.zeros(n_te, c*h*w)
    test_label = torch.zeros(n_te)


    que_train = torch.zeros(n_cl+1, dtype=torch.int32)
    que_test = torch.zeros(n_cl+1, dtype=torch.int32)
    for i in torch.arange(n_cl):
        idx_train = torch.where(train_labels == i)[0]
        idx_test = torch.where(test_labels == i)[0]
        n_train = len(idx_train)
        n_test = len(idx_test)
        que_train[i+1] = que_train[i] + n_train
        que_test[i+1] = que_test[i] + n_test 

        train_feature[que_train[i]:que_train[i+1],:] = train_features[idx_train,:]
        train_label[que_train[i]:que_train[i+1]] = train_labels[idx_train]

        test_feature[que_test[i]:que_test[i+1],:] = test_features[idx_test,:]
        test_label[que_test[i]:que_test[i+1]] = test_labels[idx_test]
    Pars = {'train_feature':train_feature, 'test_feature':test_feature, 'train_label':train_label, 'test_label':test_label, 'que_train':que_train, 'que_test':que_test}

    dir_data = os.path.join(os.getcwd(), dir_data, 'processed_data.pkl')
    with open(dir_data, 'wb') as f:  
        pkl.dump(Pars, f)
    f.close()
    return 0

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
    parser.add_argument('--model_config', type=str, default='checkpoint/config.yml', help='Path to the score model config file')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to the config file')
    parser.add_argument('--n', type=int, default=10000, help='Training sample size')
    parser.add_argument('--state_dir', type=str, default='checkpoint', help='Path to the checkpoint folder')
    parser.add_argument('--score_dir', type=str, default='scores', help='Path to the score folder')
    parser.add_argument('--t', type=int, default=100, help='The number of simulations')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to the checkpoint file')
    parser.add_argument('--outcome_dir', type=str, default='outcome/outcome.pkl', help='Path to the outcome folder')
    parser.add_argument('--seed', type=str, default='1234', help='Random seed, N means no seed.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--dev', type=str, default='cpu', help='Device: cpu | cuda:0 | ......')
    parser.add_argument('--pre_process', type=str, default='N', help='Wheterh pre-process raw data: Y | N')
    parser.add_argument('--figure_dir', type=str, default='figure/mnist.png', help='Directory to save the figure')
    args = parser.parse_args()
    dir_config =  os.path.join(os.getcwd(), args.config)
    dir_model_config =  os.path.join(os.getcwd(), args.model_config)
   
    with open(dir_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    new_config = dict2namespace(config)

    with open(dir_model_config, 'r') as f:
        model_config = yaml.unsafe_load(f)  
    f.close()
#    new_model_config = dict2namespace(model_config)

    return args, new_config, model_config


def gen_s_score(dist, feature, par):

    
    if dist == 'MVN':   
        a = (par[1] @ (feature-par[0])).reshape(-1,1)   
        score = a @ a.T - par[1]

    return score


def train_one_epoch(model1, model2, model3, opt1, opt2, opt3, loss_fun, training_loader, U_f, U_s, device):


    for i, data in enumerate(training_loader):
        print(i)
 
        inputs, _ = data
        inputs = inputs.reshape(inputs.shape[0],-1).to(device)
        

        opt1.zero_grad()
        outputs_1 = model1(inputs)
       
        loss_1 = loss_fun(outputs_1, inputs)
        loss_1.backward(retain_graph=True)
        opt1.step()
    

        opt2.zero_grad() 
        input_sf = (inputs @ U_f)
        outputs_sf= model2(input_sf)
        loss_2 = loss_fun(outputs_sf, inputs)
        loss_2.backward(retain_graph=True)
        opt2.step()

        opt3.zero_grad()
        
        input_ss = (inputs @ U_s)
        outputs_3 = model3(input_ss)        
        
        loss_3 = loss_fun(outputs_3, inputs)
        
        loss_3.backward()
       
        # Adjust learning weights
    
        opt3.step()
       
        # Gather data and report
    
    return model1, model2,  model3, opt1, opt2, opt3




#model1 AE, model 2 stein's method, model 3 2nd order stein's method1, model 4 2nd order stein's method2


def train_c_one_epoch(model0, model1, model2, model3, model4, opt1, opt2, opt3, opt4, loss_fun, training_loader, U_f, U_s, U_pca, device):

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        print(i)
        # Every data instance is an input + label pair
        inputs, label = data
        inputs = inputs.to(device)
    
        label = label.to(torch.long).to(device)

        opt1.zero_grad()
        # Make predictions for this batch
        input_ae = model0(inputs.reshape(inputs.shape[0],-1), f='c')

        outputs_1 = model1(input_ae)
        loss_1 = loss_fun(outputs_1, label)
        loss_1.backward(retain_graph=True)
        opt1.step()
    #    outputs_2 = model2(inputs)
        opt2.zero_grad()
        input_sf = inputs.reshape(inputs.shape[0],-1)
        input_sf = (input_sf @ U_f)
        outputs_2 = model2(input_sf)
        loss_2 = loss_fun(outputs_2, label)
        loss_2.backward(retain_graph=True)
        opt2.step()



        opt3.zero_grad()
        input_ss = inputs.reshape(inputs.shape[0],-1)
        input_ss = (input_ss @ U_s)
        outputs_3 = model3(input_ss)
        # Compute the loss and its gradie      
        loss_3 = loss_fun(outputs_3, label)        
        loss_3.backward(retain_graph=True)      
        # Adjust learning weights     
        opt3.step()

        opt4.zero_grad()
        input_pca = inputs.reshape(inputs.shape[0],-1)
        input_pca = (input_pca @ U_pca)
        outputs_4 = model4(input_pca)
        # Compute the loss and its gradie      
        loss_4 = loss_fun(outputs_4, label)        
        loss_4.backward()      
        # Adjust learning weights     
        opt4.step()



    
    return model1, model2,  model3, model4, opt1, opt2, opt3, opt4





def mnist(args, config, model_config):

    if args.seed != 'N':
        set_seed(int(args.seed)) 
    
    state_root = os.path.join(os.getcwd(), args.state_dir)     
    
    state_path = os.path.join(state_root, 'checkpoint.pth')
    state = torch.load(state_path)
    state = state[0]

    state = dict(state)
    state_p = {}
    for key, value in state.items():
        key_p = key.split('.')
        key_p = '.'.join(key_p[1:])
        state_p[key_p] = value

    score_model = CondRefineNetDilated(model_config.data.image_size, model_config.model.num_classes, model_config.data.channels, model_config.model.ngf) 
    score_model.load_state_dict(state_p)
    print('Load score model successfully') 

    data_root = os.path.join(os.getcwd(), args.data_dir)

    data_path = os.path.join(data_root, 'processed_data.pkl')

    with open(data_path,'rb') as f:
        Pars = pkl.load(f)
    f.close() 

    print('Load data successfully!')

    n = args.n

    t = args.t
    n_classes = config.n_classes

    fold = config.fold

    batch_size = config.batch_size
    mse = nn.MSELoss()
    p = config.channels * (config.image_size ** 2)
    alpha = config.alpha
    sigma = config.sigma
    sigmas = torch.ones(n_classes) * sigma

    EPOCHS = config.EPOCHS

    cross_entropy = nn.CrossEntropyLoss()

    shape = (-1, config.channels, config.image_size, config.image_size)

    dev = args.dev
    device = (dev) 
    ssim = StructuralSimilarityIndexMeasure().to(device)

    outdims = config.outdims

    outcome = torch.zeros(t, len(outdims),3,4)

    for k in torch.arange(t):
        print(k)

        train_feature = Pars['train_feature']
        test_feature = Pars['test_feature']
        train_label = Pars['train_label']
        test_label = Pars['test_label']

        idx = torch.randperm(len(train_feature))
        train_feature = train_feature[idx]
        train_label = train_label[idx]

        train_feature = train_feature[:n]
        train_label = train_label[:n]

        sample_mean = torch.mean(train_feature,dim=0).to(device)
        sample_cov = torch.cov(train_feature.T).to(device)   

        idx_p = torch.where(sample_mean > 0) 

        train_set = data_set(train_feature.reshape(shape), train_label)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)

        del train_set
        gc.collect()

        n_tr = len(train_label)
 
        n_sub = int(n_tr/fold)

        score_path = os.path.join(os.getcwd(), args.score_dir)

        shutil.rmtree(score_path)

        pathlib.Path(args.score_dir).mkdir(parents=True, exist_ok=True) 
        
        score_model.eval()

        for j in torch.arange(fold):

            print(j)
        
            sub_feature = train_feature[(j*n_sub):((j+1)*n_sub)].reshape(shape)
    
            sub_labels = torch.randint(len(sigmas), (len(sub_feature),))

            score_est = score_model(sub_feature, sub_labels).reshape(len(sub_feature),-1)
            score_est = torch.clamp(score_est, min=torch.quantile(score_est,(alpha/2)), max=torch.quantile(score_est,(1-alpha/2)))
            sub_feature = sub_feature.reshape(sub_feature.shape[0],-1) 
            B_e = score_est.T @ sub_feature
    
            with open(os.path.join(score_path,'{}.pkl'.format(j)), 'wb') as f:  
                pkl.dump(B_e, f)
            f.close()
            torch.cuda.empty_cache()

        B_ft =  torch.zeros(p*p).reshape(p,p)
        for j in torch.arange(fold):
            with open(os.path.join(score_path,'{}.pkl'.format(j)),'rb') as f:
                pars = pkl.load(f)
            f.close()
            B_ft += pars 

        B_ft, _, _ = torch.linalg.svd(B_ft/n_tr)
        B_ft = B_ft.to(device)

        print('Estimate first order successfully!')

        U, L, V = torch.svd(sample_cov)
  
        print('Estimate second order successfully!')
        B_pt, _, _ = torch.linalg.svd(torch.cov(train_feature.T))
        B_pt =  B_pt.to(device)
 
        for o, outdim in enumerate(outdims):

            B_st =  torch.zeros(p*p).reshape(p,p).to(device)

            L_copy = copy.deepcopy(L)

            L_copy[:outdim] = 1/L_copy[:outdim]
            L_copy[outdim:] = 0

            inv_sample_cov = (U @ torch.diag(L_copy) @ V.T).to(device) 
            train_feature = train_feature.to(device)

            for j in torch.arange(n):
                print(j)

                S = gen_s_score('MVN', train_feature[j,:], [sample_mean, inv_sample_cov])
                y = train_feature[j]
                y = y[idx_p]
                y = y.reshape(-1,1,1)
                B_st += torch.mean(y*S,dim=0)
        
            B_st, _, _ = torch.linalg.svd(B_st/n)

            h = 2 * outdim
            B_f = B_ft[:,:outdim]       
            B_p = B_pt[:,:outdim]

            B_s = B_st[:,:outdim]
            B_s = B_s.to(device)

            ae = AE(config.image_size**2, h, outdim).to(device) 
            aesf = Decoder(config.image_size**2, h, outdim).to(device) 
            aess = Decoder(config.image_size**2, h, outdim).to(device) 

            opt_ae = torch.optim.Adam(ae.parameters())
            opt_aesf = torch.optim.Adam(aesf.parameters())
            opt_aess = torch.optim.Adam(aess.parameters())

            epoch_number = 0
            for _ in range(EPOCHS):
                print('EPOCH {}:'.format(epoch_number + 1))

                epoch_number += 1
                ae.train(True)
                aesf.train(True)
                aess.train(True)

                ae, aesf, aess, opt_ae, opt_aesf, opt_aess = train_one_epoch(ae, aesf, aess, opt_ae, opt_aesf, opt_aess, mse, train_loader, B_f, B_s, device)
       
            clas_ae = Classifier(outdim, n_classes).to(device) 
            clas_aesf = Classifier(outdim, n_classes).to(device) 
            clas_aess = Classifier(outdim, n_classes).to(device) 
            clas_pca = Classifier(outdim, n_classes).to(device) 
    
            clas_ae.train(True)
            clas_aesf.train(True)
            clas_aess.train(True)
            clas_pca.train(True)
    
            opt_ae = torch.optim.Adam(clas_ae.parameters())
            opt_aesf = torch.optim.Adam(clas_aesf.parameters())
            opt_aess = torch.optim.Adam(clas_aess.parameters())
            opt_pca = torch.optim.Adam(clas_pca.parameters())
    
            epoch_number = 0
            ae_copy = copy.deepcopy(ae)
            ae_copy.to(device)
            ae_copy.eval()
            for _ in range(EPOCHS):
                print('EPOCH {}:'.format(epoch_number + 1))
                epoch_number += 1

                clas_ae, clas_aesf, clas_aess, clas_pca, opt_ae, opt_aesf, opt_aess, opt_pca = train_c_one_epoch(ae_copy, clas_ae, clas_aesf, clas_aess, clas_pca,  opt_ae, opt_aesf, opt_aess, opt_pca, cross_entropy, train_loader, B_f, B_s, B_p, device)

            print('Train classifier successfully!')

            aesf.eval()
            aess.eval()

            clas_ae.eval()
            clas_aesf.eval()
            clas_aess.eval()
            clas_pca.eval()
    
            test_feature = test_feature.to(device)
            test_label = test_label.to(device)
    
            with torch.no_grad():

                toutputs_ae = torch.clamp(ae(test_feature.reshape(test_feature.shape[0],-1)), min=0, max=1)
                toutputs_aesf = torch.clamp(aesf(test_feature.reshape(test_feature.shape[0],-1) @ B_f), min=0, max=1)
                toutputs_aess = torch.clamp(aess(test_feature.reshape(test_feature.shape[0],-1) @ B_s), min=0, max=1)
                toutputs_pca = torch.clamp(test_feature @ B_p @ B_p.T, min=0, max=1)
    
                o_p = test_feature.reshape(shape)
                ae_p = toutputs_ae.reshape(shape)
                aesf_p = toutputs_aesf.reshape(shape)
                aess_p = toutputs_aess.reshape(shape)
                pca_p = toutputs_pca.reshape(shape)
                tssim_ae = ssim(ae_p, o_p)
                tssim_aesf = ssim(aesf_p, o_p)
                tssim_aess = ssim(aess_p, o_p)
                tssim_pca = ssim(pca_p, o_p)
                avg_tssim_ae = torch.mean(tssim_ae) 
                avg_tssim_aesf = torch.mean(tssim_aesf)
                avg_tssim_aess = torch.mean(tssim_aess)
                avg_tssim_pca = torch.mean(tssim_pca) 

                tloss_ae = torch.norm(toutputs_ae -  test_feature,dim=1)/torch.norm(test_feature, dim=1)
                tloss_aesf = torch.norm(toutputs_aesf -  test_feature,dim=1)/torch.norm(test_feature, dim=1)            
                tloss_aess = torch.norm(toutputs_aess -  test_feature,dim=1)/torch.norm(test_feature, dim=1)
                tloss_pca = torch.norm(toutputs_pca -  test_feature,dim=1)/torch.norm(test_feature, dim=1)

                avg_tloss_ae = torch.mean(tloss_ae)
                avg_tloss_aesf = torch.mean(tloss_aesf)
                avg_tloss_aess = torch.mean(tloss_aess)
                avg_tloss_pca = torch.mean(tloss_pca)
   
                test_feature_1 = ae(test_feature.reshape(test_feature.shape[0],-1), f='c').clone().detach()
                clas_toutputs_ae = torch.argmax(F.softmax(clas_ae(test_feature_1), dim=1), dim=1)
                clas_toutputs_aesf = torch.argmax(F.softmax(clas_aesf(test_feature.reshape(test_feature.shape[0],-1) @ B_f), dim=1), dim=1)
                clas_toutputs_aess = torch.argmax(F.softmax(clas_aess(test_feature.reshape(test_feature.shape[0],-1) @ B_s), dim=1), dim=1)
                clas_toutputs_pca = torch.argmax(F.softmax(clas_pca(test_feature.reshape(test_feature.shape[0],-1) @ B_p), dim=1), dim=1)
        
                clas_avg_tloss_ae = torch.mean(torch.tensor(clas_toutputs_ae == test_label, dtype=torch.float))
                clas_avg_tloss_aesf = torch.mean(torch.tensor(clas_toutputs_aesf == test_label, dtype=torch.float))
                clas_avg_tloss_aess = torch.mean(torch.tensor(clas_toutputs_aess == test_label, dtype=torch.float))
                clas_avg_tloss_pca = torch.mean(torch.tensor(clas_toutputs_pca == test_label, dtype=torch.float))

                outcome[k,o, 0, 0] = avg_tssim_ae
                outcome[k,o, 0, 1] = avg_tssim_aesf
                outcome[k,o, 0, 2] = avg_tssim_aess
                outcome[k,o, 0, 3] = avg_tssim_pca

                outcome[k,o, 1, 0] = avg_tloss_ae
                outcome[k,o, 1, 1] = avg_tloss_aesf
                outcome[k,o, 1, 2] = avg_tloss_aess
                outcome[k,o, 1, 3] = avg_tloss_pca

                outcome[k,o, 2, 0] = clas_avg_tloss_ae
                outcome[k,o, 2, 1] = clas_avg_tloss_aesf
                outcome[k,o, 2, 2] = clas_avg_tloss_aess
                outcome[k,o, 2, 3] = clas_avg_tloss_pca

                dir_outcome = os.path.join(os.getcwd(), args.outcome_dir)
                with open(dir_outcome, 'wb') as f:  
                    pkl.dump(outcome, f)
                f.close()
    return 0