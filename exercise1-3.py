import numpy as np
from plot import *
from exercise1_2 import add_noise

def linear_regression(X, y):
    w_star, *_ = np.linalg.lstsq(X, y, rcond=None)
    return w_star  

def postprod_grid_linear(data, w_star, grid, xx):
    
    postprod_grid = np.empty((grid.shape[0], 3)) 
    postprod_grid[:, 0] = 1
    postprod_grid[:, 1:] = grid
    
    grid_linear = (postprod_grid @ w_star).reshape(*xx.shape)
    
    return grid_linear

def plot_maestrati(data, test, mu0, mu1, postprod_grid):
    
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
    
    return fig
    
if __name__ == '__main__':
    
    data1 = pd.read_csv('data/trainA',header=None,sep=' ', names=["x1", "x2", "y"])
    data1_1 = pd.read_csv('data/testA',header=None,sep=' ', names=["x1", "x2", "y"])
    data2 = pd.read_csv('data/trainB',header=None,sep=' ', names=["x1", "x2", "y"])
    data2_1 = pd.read_csv('data/testB',header=None,sep=' ', names=["x1", "x2", "y"])
    data3 = pd.read_csv('data/trainC',header=None,sep=' ', names=["x1", "x2", "y"])
    data3_1 = pd.read_csv('data/testC',header=None,sep=' ', names=["x1", "x2", "y"])
    data = [(data1, data1_1), (data2, data2_1), (data3, data3_1)]
    
    char ='A'

    for dat, dat_test in data:
        X, y = add_noise(np.array(dat.iloc[:,0:2])), np.squeeze(np.array(dat.iloc[:,2:]))
        
        w_star = linear_regression(X, y)
        print(w_star)
        grid, xx, yy, xlim, ylim = create_grid(np.array(dat))
    
        postprob_grid = postprod_grid_linear(dat, w_star, grid, xx)
        
        mu1 = np.mean(dat[dat['y'] == 0], axis=0).iloc[0:2]
        mu2 = np.mean(dat[dat['y'] == 1], axis=0).iloc[0:2]
    
        
        fig = plot_maestrati(np.array(dat), np.array(dat_test), mu1, mu2, postprob_grid)
        fig.savefig('./images/train_linear_%s.png' % char)
        i = ord(char[0])
        i += 1
        char = chr(i)
        
        
        
    