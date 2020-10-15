import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import pickle as pkl
# Random state.
RS = 20150101
 
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from scipy import sparse
 
# We import seaborn to make nice plots.
dataset = 'acm'
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
digits = load_digits()
# We first reorder the data points according to the handwritten numbers.
# X = np.vstack([digits.data[digits.target==i]
#                for i in range(10)])
if dataset == 'acm':
    X_vis_1 = np.loadtxt('visdata/acm.txt')
    y = np.loadtxt('visdata/acm_label.txt')
    X_vis_2 = np.load('visdata/{}_features_out.npy'.format(dataset),allow_pickle=True)
    clusternum = 3
    num1max = 0
    numlist = []
    indexlist = []
    # print(type(X_vis_1.shape[0]))
    # for i in range(X_vis_1.shape[0]):
    #     num=0
    #     for j in range(X_vis_1.shape[1]):
    #         if X_vis_1[i][j]==1:
    #             num = num+1
    #             numlist.append(num)
    #     indexlist.append(i)
    # print(max(numlist))
    print('X_vis_1',type(X_vis_1),X_vis_1.shape)
    print('X_vis_2',type(X_vis_2),X_vis_2.shape)
    print('y',type(y),y.shape)
if dataset == 'cora':
    X_vis_1 = np.load('visdata/{}_features_in.npy'.format(dataset),allow_pickle=True)
    y = np.loadtxt('visdata/cora_label.txt')
    X_vis_2 = np.load('visdata/{}_features_out.npy'.format(dataset),allow_pickle=True)
    clusternum = 7
    print('X_vis_1',type(X_vis_1),X_vis_1.shape)
    print('X_vis_2',type(X_vis_2),X_vis_2.shape)
    print('y',type(y),y.shape)

if dataset == 'dblp_hg_new':
    X_vis_1 = np.load('visdata/{}_features_in.npy'.format(dataset),allow_pickle=True)
    X_vis_2 = np.load('visdata/{}_features_out.npy'.format(dataset),allow_pickle=True)
    y = pkl.load(open('visdata/true_labels.pkl', 'rb'))
    y = np.array(y)
    clusternum = 3
    print('X_vis_1',type(X_vis_1),X_vis_1.shape)
    print('X_vis_2',type(X_vis_2),X_vis_2.shape)
    print('y',type(y),y.shape)


digits_proj = TSNE(random_state=RS).fit_transform(X_vis_1)
 
def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", clusternum))
 
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    colors  = np.delete(colors,np.where(x[:,1]>20),axis=0)
    x  = np.delete(x,np.where(x[:,1]>20),axis=0)
    print(x)
    print(max(x[:,1]))
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-3, 3)
    plt.ylim(-2, 2)
    ax.axis('off')
    ax.axis('tight')
 
    # We add the labels for each digit.
    txts = []
    for i in range(clusternum):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, "", fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
 
    return f, ax, sc, txts
 
scatter(digits_proj, y)
plt.savefig('acm_features_in.png', dpi=120)
