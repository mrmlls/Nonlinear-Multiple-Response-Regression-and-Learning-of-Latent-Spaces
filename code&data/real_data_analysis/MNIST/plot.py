# We plot the results of the experiments in the paper.
import matplotlib.pyplot as plt # type: ignore
import pickle as pkl
import torch # type: ignore
import os
from utils import parse_args_and_config


args, config, _ = parse_args_and_config()

x_axis = config.outdims
dir = os.path.join(os.getcwd(), args.outcome_dir)
pkl_file = open(dir, 'rb')
outcome = pkl.load(pkl_file)
pkl_file.close()

data = torch.quantile(outcome, 0.5, dim=0)

plt.figure(figsize=(24,8))

plt.subplot(1, 3,1)

plt.plot(x_axis, data[:,1,0], marker="^",  markersize=14,alpha=1, linewidth=3,linestyle='solid', label='Autoencoder')

plt.plot(x_axis, data[:,1,1],marker="s",  markersize=14,alpha=1, linewidth=3,linestyle='solid', label="First-order Stein's Method")

plt.plot(x_axis, data[:,1,2], marker="D", markersize=14, alpha=1, linewidth=3,linestyle='solid', label="Second-order Stein's Method")

plt.plot(x_axis, data[:,1,3], marker="o", markersize=14,alpha=1, linewidth=3, linestyle='solid', label="PCA")


plt.tick_params(axis='x', labelsize=21)


plt.tick_params(axis='y', labelsize=21)

plt.ylabel('NRSE', fontsize=21)

plt.xlabel('Dimension of Embedding Space', fontsize=21)
plt.legend(loc = 'upper right', fontsize= 17)

plt.subplot(1, 3, 2)

plt.plot(x_axis, data[:,0,0], marker="^",markersize=14, alpha=1, linewidth=3,linestyle='solid', label='Autoencoder')

plt.plot(x_axis, data[:,0,1],marker="s", markersize=14,alpha=1, linewidth=3,linestyle='solid', label="First-order Stein's Method")

plt.plot(x_axis, data[:,0,2], marker="D",markersize=14, alpha=1, linewidth=3,linestyle='solid', label="Second-order Stein's Method")

plt.plot(x_axis, data[:,0,3], marker="o",markersize=14, alpha=1, linewidth=3,label="PCA")

plt.ylabel('SSIM', fontsize=21)
plt.ylim(0.4, 0.7)
plt.xlabel('Dimension of Embedding Space', fontsize=21)
plt.legend(loc = 'lower right', fontsize= 17)

plt.tick_params(axis='x', labelsize=21)


plt.tick_params(axis='y', labelsize=21)



plt.subplot(1, 3, 3)

plt.plot(x_axis, data[:,2,0], marker="^", markersize=14, alpha=1, linewidth=3, linestyle='solid', label='Autoencoder')

plt.plot(x_axis, data[:,2,1],marker="s",  markersize=14, alpha=1, linewidth=3,linestyle='solid', label="First-order Stein's Method")

plt.plot(x_axis, data[:,2,2], marker="D",  markersize=14, alpha=1, linewidth=3,linestyle='solid', label="Second-order Stein's Method")

plt.plot(x_axis, data[:,2,3], marker="o",  markersize=14, alpha=1, linewidth=3, linestyle='solid', label="PCA")

plt.tick_params(axis='x', labelsize=21)


plt.tick_params(axis='y', labelsize=21)


plt.ylabel('Classification Accuracy', fontsize=21)
plt.ylim(0.79, 0.87)
plt.xlabel('Dimension of Embedding Space', fontsize=21)
plt.legend(loc = 'lower right', fontsize= 17)

plt.tight_layout(pad=1.08)


dir = os.path.join(os.getcwd(), args.figure_dir)
plt.savefig(dir)
plt.show()


