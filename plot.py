import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def create_grid(data):
    

    xmin, xmax =np.min(data[:, 0])-1, np.max(data[:, 0])+1
    ymin, ymax = np.min(data[:, 1])-1, np.max(data[:, 1])+1
  
    # create mesh
    x_space = np.linspace(xmin, xmax, 40+1)
    y_space = np.linspace(ymin, ymax, 50+1)
    
    xx, yy = np.meshgrid(x_space,y_space)
    grid = np.dstack((xx,yy))    
    
    grid = grid.reshape(np.prod(xx.shape),2)
    
    return grid, xx, yy, [xmin, xmax], [ymin, ymax]

def plot(data, mu1, mu2, decision_boundary, postprob_grid, xx, yy, xlim, ylim, title='Posterior probability $p(y=1|x)$'):
    
    fig = plt.figure(figsize=(8, 6), dpi=80)

    cf_ = plt.contourf(xx, yy, postprob_grid, levels=40,
                   zorder=-1, alpha=0.9,
                   cmap='viridis')

    cbar_ = fig.colorbar(cf_)
    cbar_.set_label(title)

    ## Add the class centroids
    for i, mu in enumerate([mu1,mu2]):
        plt.scatter(*mu, s=100, marker='D', edgecolors='k', linewidths=.8, label="$\\mu_{%d}$" % i)

    plt.scatter(x = data[data['y']==0]['x1'], y=data[data['y']==0]['x2'], s=20, 
                edgecolors='k', c='r', linewidths=.5, label="Class $0$")

    plt.scatter(x = data[data['y']==1]['x1'], y=data[data['y']==1]['x2'], s=20,
                edgecolors='k', c='b', linewidths=.5, label="Class $1$")

    """plt.plot(*decision_boundary.T, c='k', lw=2.2, ls='--',
             label="Decision boundary")"""

    cs_ = plt.contour(cf_, levels=[0.5],  colors='k', linestyles='--')

    
    legend_ = plt.legend(frameon=1)
    frame_ = legend_.get_frame()
    frame_.set_color('white')

    plt.grid(False)
    plt.axis('scaled')

    #plt.title("Data and posterior probability (dataset %s)" % dataset_id)

    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0], ylim[1])
    fig.tight_layout()
    
    return fig


