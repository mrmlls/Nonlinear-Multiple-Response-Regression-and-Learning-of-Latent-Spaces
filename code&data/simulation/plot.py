import torch # type: ignore
import pickle as pkl
import matplotlib.pyplot as plt # type: ignore
import yaml
from utils import parse_args_and_config, dict2namespace
import os

args, _ = parse_args_and_config()

dir = os.path.join(os.getcwd(), args.fig_config) 
with open(dir, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
f.close()
config = dict2namespace(config)


#plt.rc('text', usetex=True)
x_axis = config.x_axis
items = config.items
dists = config.dists
order = config.order

x_ticket = config.x_ticket 





if args.type == 'ln':
    result = torch.zeros(len(dists),len(items),len(x_axis),3)

    for l, dist in enumerate(dists):
        for i, item in enumerate(items):
            for j, num in enumerate(x_axis):
                dir_outcome = os.path.join(os.getcwd(), 'outcomes', args.type, dist)
                out_dir = os.path.join(dir_outcome, 'outcome_' + str(item) + '_'+ str(num) + '.pkl')
                pkl_file = open(out_dir, 'rb')
                data = pkl.load(pkl_file)
                pkl_file.close()
                Dis_n = data['Dis_NN']
                Dis_s = data['Dis_S']
                Dis_r = data['Dis_RR']
                result[l,i,j,0] = torch.quantile(Dis_r,0.5)
                result[l,i,j,1] = torch.quantile(Dis_n,0.5)
                result[l,i,j,2] = torch.quantile(Dis_s,0.5)

    

    for i, item in enumerate(items):
        plt.figure(figsize=(24,8))
        for j in torch.arange(len(dists)):
            j = j.item()
            plt.subplot(1, 3, j+1)


            plt.ylabel('Distance', fontsize=22)
    
            plt.xlabel('Sample Size', fontsize=22)


            plt.plot(x_axis, result[order[j],i,:,0], marker="^", alpha=1,markersize=14, linewidth=3,linestyle='solid', label="Reduced Rank Regression")
            plt.plot(x_axis, result[order[j],i,:,1],marker="s", alpha=1,markersize=14, c='green', linewidth=3,linestyle='solid',label="Neural Network")
            plt.plot(x_axis, result[order[j],i,:,2], marker="D", alpha=1,markersize=14, c='red', linewidth=3,linestyle='solid', label="First Order Stein's Method")
            plt.tick_params(axis='x', labelsize=22)


            plt.tick_params(axis='y', labelsize=22)

            plt.xticks(ticks=x_ticket)

            plt.legend(fontsize=20, loc='upper right')



            plt.tight_layout(pad=1.08)
        fig_dir = os.path.join(os.getcwd(), 'figures', args.type + '_'+ str(item) + '.png')
        plt.savefig(fig_dir)



else:
    result = torch.zeros(len(dists),len(items),len(x_axis),4)

    for l, dist in enumerate(dists):
        for i, item in enumerate(items):
            for j, num in enumerate(x_axis):
                dir_outcome = os.path.join(os.getcwd(), 'outcomes', args.type, dist)
                out_dir = os.path.join(dir_outcome, 'outcome_' + str(item) + '_'+ str(num) + '.pkl')
                pkl_file = open(out_dir, 'rb')
                data = pkl.load(pkl_file)
                pkl_file.close()
                Dis_n = data['Dis_n']
                Dis_s = data['Dis_s']
                Dis_r = data['Dis_r']
                Dis_1 = data['Dis_1']
                result[l,i,j,0] = torch.quantile(Dis_r,0.5)
                result[l,i,j,1] = torch.quantile(Dis_n,0.5)
                result[l,i,j,2] = torch.quantile(Dis_s,0.5)
                result[l,i,j,3] = torch.quantile(Dis_1,0.5)
    

    for i, item in enumerate(items):
        plt.figure(figsize=(24,8))
        for j in torch.arange(len(dists)):
            j = j.item()
            plt.subplot(1, 3, j+1)


            plt.ylabel('Distance', fontsize=22)
    
            plt.xlabel('Sample Size', fontsize=22)


            plt.plot(x_axis, result[order[j],i,:,0], marker="^",markersize=14,  alpha=1, linewidth=3,linestyle='solid', label="Reduced Rank Regression")
            plt.plot(x_axis, result[order[j],i,:,1],marker="s", markersize=14, alpha=1, c='green', linewidth=3,linestyle='solid',label="Neural Network")
            plt.plot(x_axis, result[order[j],i,:,2], marker="D",markersize=14,  alpha=1, c='red', linewidth=3,linestyle='solid', label="First Order Stein's Method")
            plt.plot(x_axis, result[order[j],i,:,3], marker="o", markersize=14, alpha=1, linewidth=3, linestyle='solid', label="Second Order Stein's Method")
            plt.tick_params(axis='x', labelsize=22)


            plt.tick_params(axis='y', labelsize=22)

            plt.xticks(ticks=x_ticket)

            plt.legend(fontsize=20, loc='upper right')



        plt.tight_layout(pad=1.08)
        fig_dir = os.path.join(os.getcwd(), 'figures', args.type + '_'+ str(item) + '.png')
        plt.savefig(fig_dir)
