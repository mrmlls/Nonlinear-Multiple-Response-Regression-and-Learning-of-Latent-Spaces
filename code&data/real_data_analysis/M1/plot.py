import matplotlib.pyplot as plt # type: ignore
import pickle as pkl
import torch # type: ignore
from utils import parse_args_and_config
import os

args, config = parse_args_and_config()



x_axis = config.outdims


dir_out = os.path.join(os.getcwd(), args.out_dir)


pkl_file = open(dir_out, 'rb')

data = pkl.load(pkl_file)
    
pkl_file.close()
mses = torch.quantile(data, 0.5, dim=0)



plt.figure(figsize=(24,8))



for i in torch.arange(3):

    plt.subplot(1, 3, i.item()+1)

    plt.plot(x_axis, mses[i,:,0], marker="^", markersize=10, alpha=1, linewidth=3,linestyle='solid', label='Semi-supervised')

    plt.plot(x_axis, mses[i,:,1],marker="s", markersize=10,alpha=1, linewidth=3,linestyle='solid', label="Supervised")

    plt.plot(x_axis, mses[i,:,2], marker="D", markersize=10,alpha=1, linewidth=3,linestyle='solid', label="Unsupervised")

    plt.plot(x_axis, mses[i,:,3], marker="o", markersize=10,alpha=1, linewidth=3, linestyle='solid', label="PCA")


    plt.ylabel('PMSE', fontsize= 25)
    
    plt.xlabel('Dimension of Embedding Space', fontsize= 25)
    plt.legend(loc = 'upper right', fontsize= 18)

    plt.tick_params(axis='x', labelsize=25)

    plt.tick_params(axis='y', labelsize=25)
    
plt.tight_layout(pad=1.08)

dir = os.path.join(os.getcwd(), args.figure_dir)
plt.savefig(dir)
plt.show()