import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot import *

plt.style.use('seaborn-whitegrid')


def LDA(train_data):
    X_data = train_data.iloc[:,0:2]
    y_data = train_data.iloc[:,2:3]
    pi_ = np.mean(y_data == 0)
    mu1_ = np.mean(train_data[train_data['y'] == 0], axis=0).iloc[0:2]
    mu2_ = np.mean(train_data[train_data['y'] == 1], axis=0).iloc[0:2]
    sigma_ = np.cov(X_data, rowvar=False)

    return float(pi_), np.array(mu1_), np.array(mu2_), sigma_

def decision_boundary(pi_, mu1_, mu2_, sigma_,precision_matrix_):
    
    normal_vec = precision_matrix_ @ (mu1_ - mu2_)
    renormed_vec = normal_vec / np.linalg.norm(normal_vec)
    e = np.random.rand(2); e /= np.linalg.norm(e)
    support_vec = e - np.dot(renormed_vec, e) * renormed_vec

    renormed_vec.dot(support_vec)

    mu_avg = 0.5 * (mu1_ + mu2_)

    b_ = normal_vec @ mu_avg + np.log(pi_ / (1-pi_))
    b_ = np.array([b_])

    base_point = np.linalg.lstsq(normal_vec[None], b_, rcond=None)[0]

    line_points_ = [
        base_point + t * support_vec
        for t in [6, 63]
    ]

    line_points_ = np.array(line_points_)
    
    return line_points_

def postprob_grid_lda(data, pi_, mu1_, mu2_, precision_matrix_):
    
    def lda_posterior_prob(x, pi_, mu1_, mu2_, precision_matrix_):
        dx0 = x - mu1_
        dx1 = x - mu2_
        c0 = pi_ * np.exp(-0.5 * np.dot(dx0, precision_matrix_ @ dx0))
        c1 = (1-pi_) * np.exp(-0.5 * np.dot(dx1, precision_matrix_ @ dx1))
        return c1 / (c0 + c1)
    
    grid, xx, yy, xlim, ylim = create_grid(np.array(data))
    
    postprob_grid = np.array([
        lda_posterior_prob(x, pi_, mu1_, mu2_, precision_matrix_)
        for x in grid
    ])

    postprob_grid = postprob_grid.reshape(*xx.shape)
    
    return postprob_grid, xx, yy, xlim, ylim

def LDA_plot(data, pi_, mu1_, mu2_, line_points_, postprob_grid, xg, yg, xlims_, ylims_):
    
    fig = plt.figure(figsize=(8, 6), dpi=80)

    cf_ = plt.contourf(xg, yg, postprob_grid, levels=40,
                   zorder=-1, alpha=0.9,
                   cmap='viridis')

    cbar_ = fig.colorbar(cf_)
    cbar_.set_label("Posterior probability $p(y=1|x)$")

    ## Add the class centroids
    for i, mu in enumerate([mu1_,mu2_]):
        plt.scatter(*mu, s=100, marker='D',
                    edgecolors='k', linewidths=.8,
                    label="$\\mu_{%d}$" % i)

    plt.scatter(x = data[data['y']==0]['x1'], y=data[data['y']==0]['x2'], s=20, edgecolors='k',c='r', linewidths=.5, label="Class $0$")

    plt.scatter(x = data[data['y']==1]['x1'], y=data[data['y']==1]['x2'], s=20, edgecolors='k',
                c='b', linewidths=.5, label="Class $1$")

    plt.plot(*line_points_.T, c='k', lw=2.2, ls='--',
             label="Decision boundary")


    legend_ = plt.legend(frameon=1)
    frame_ = legend_.get_frame()
    frame_.set_color('white')

    plt.grid(False)
    plt.axis('scaled')

    #plt.title("Data and posterior probability (dataset %s)" % dataset_id)

    plt.xlim(*xlims_)
    plt.ylim(*ylims_)
    fig.tight_layout();
    
    return fig


def plot_maestrati(data, test, mu0, mu1, Sigma, alpha, postprod_grid):
    
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Generative Train points '+char)
    
    
    
    xmin, xmax =np.min(data[:, 0])-1, np.max(data[:, 0])+1
    ymin, ymax = np.min(data[:, 1])-1, np.max(data[:, 1])+1
    x_space = np.linspace(xmin, xmax, 40+1)
    y_space = np.linspace(ymin, ymax, 50+1)
    
    xx, yy = np.meshgrid(x_space,y_space)
    grid = np.dstack((xx,yy)) 
    
    cf_ = axs[0].contourf(xx, yy, postprod_grid, levels=40,
                   zorder=-1, alpha=0.9,
                   cmap='viridis')
    cbar_ = fig.colorbar(cf_)
    cbar_.set_label("Posterior probability $p(y=1|x)$")
    grid = grid.reshape(np.prod(xx.shape),2)
    w=np.linalg.inv(Sigma)@(mu1-mu0)
    b=np.log(alpha/(1-alpha))-0.5*(mu1-mu0).T@(np.linalg.inv(Sigma))@(mu1+mu0)
    
    m=-w[0]/w[1]
    c=-b/w[1]
    
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    #Train
    axs[0].plot(xd, yd, 'k', lw=1, ls='--')
    axs[0].scatter(data[:, 0], data[:, 1], c=data[:, 2], s=20 , alpha=0.8)
    axs[0].scatter(mu0[0], mu0[1], label="mu_0")
    axs[0].scatter(mu1[0], mu1[1], label="mu_1")
    axs[0].legend()
    axs[0].set_ylabel(r'$x_2$')
    axs[0].set_xlabel(r'$x_1$')
    axs[0].set_title('Train set')
    #Test
    xmin, xmax =np.min(test[:, 0])-1, np.max(test[:, 0])+1
    ymin, ymax = np.min(test[:, 1])-1, np.max(test[:, 1])+1
    x_space = np.linspace(xmin, xmax, 40+1)
    y_space = np.linspace(ymin, ymax, 50+1)
    
    xx, yy = np.meshgrid(x_space,y_space)
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    cf_ = axs[1].contourf(xx, yy, postprod_grid, levels=40,
                   zorder=-1, alpha=0.9,
                   cmap='viridis')
    axs[1].plot(xd, yd, 'k', lw=1, ls='--')
    axs[1].scatter(test[:, 0], test[:, 1], c=test[:, 2], s=10 , alpha=0.8)
    axs[1].scatter(mu0[0], mu0[1], label="mu_0")
    axs[1].scatter(mu1[0], mu1[1], label="mu_1")
    axs[1].set_ylabel(r'$x_2$')
    axs[1].set_xlabel(r'$x_1$')
    axs[1].set_title('Test set')
    plt.show()
    

if __name__ == '__main__' :
    data1 = pd.read_csv('data/trainA',header=None,sep=' ', names=["x1", "x2", "y"])
    data11 = pd.read_csv('data/testA', header = None, sep = ' ', names=['x1', 'x2', 'y'])
    data2 = pd.read_csv('data/trainB',header=None,sep=' ', names=["x1", "x2", "y"])
    data22 = pd.read_csv('data/testB' , header = None, sep = ' ', names = ['x1','x2','y'])
    data3 = pd.read_csv('data/trainC',header=None,sep=' ', names=["x1", "x2", "y"])
    data33 = pd.read_csv('data/testC',header=None,sep=' ', names=["x1", "x2", "y"])
    data = [(data1,data11), (data2,data22), (data3, data33)]
    
    char ='A'
    for (dat, test) in data:
        pi, mu1, mu2, sigma_ = LDA(dat)

        precision_matrix_ = np.linalg.inv(sigma_)
    
        decision_bound = decision_boundary(pi, mu1, mu2, sigma_,precision_matrix_)
    
        postprod_grid, xx, yy, xlim, ylim = postprob_grid_lda(dat, pi, mu1, mu2, precision_matrix_)
        
        plot_maestrati(np.array(dat), np.array(test), mu1, mu2, sigma_, pi, postprod_grid)
        #fig = plot(dat, mu1, mu2, decision_bound, postprob_grid, xx, yy, xlim, ylim)
        
        fig.savefig('./images/train%s.png' % char)
        i = ord(char[0])
        i += 1
        char = chr(i)