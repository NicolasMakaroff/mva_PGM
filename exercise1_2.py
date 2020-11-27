from scipy.special import expit
import numpy as np
from plot import *
import pandas as pd

def add_noise(X):
    X_noise = np.empty((X.shape[0], 3))
    X_noise[:, 1:] = X
    X_noise[:, 0] = 1.
    return X_noise

def logit_func(x):
    return -np.log(1 + np.exp(-x))

def log_likelihood(w, X, y):
    s = y * X.dot(w)
    return np.sum(logit_func(s))

def gradient(w, X, y):
    s = y * X.dot(w)
    z = (1 - expit(s)) * y
    prod = z[:, None] * X
    return np.sum(prod, axis=0)

def hessian(w, X, y):
    s = y * X.dot(w)
    sigma = expit(s)
    z = (sigma - 1) * sigma
    outers = np.matmul(X[:,:,None], X[:,None])
    prod = z[:,None,None] * outers
    return np.sum(prod, axis=0)

def newton_raphson(w_init, X, epsilon, num_iters=50, stopping_criterion=1e-10, alpha=2.0):
    
    w_iters = []
    w_iters.append(w_init)
    w = w_init
    values = [log_likelihood(w, X, epsilon)]
    
    for i in range(num_iters):
        
        grad = gradient(w, X, epsilon)
        hess = hessian(w, X, epsilon)
        
        delta_w = np.linalg.solve(hess, -grad)
        lbda = np.dot(delta_w, grad)
        step = .5 * lbda ** 2
        
        w_new = w + alpha * delta_w
        w = w_new
        w_iters.append(w)
        
        values.append(log_likelihood(w, X, epsilon))
        
        
        if step <= stopping_criterion:
            break
    values = np.array(values)      
    return w_iters, values

def logistic_regression(X, y):
    r"""
    
    Returns w_opt, (w_iters, value_iters) where:
        w_star: trained weights
        w_iters: Newton iterates of the weights
        values: value of the log-likelihood objective at each training step
    """
    epsilon = 2 * y - 1
    
    w_init = np.zeros(3)

    w_iters, values = newton_raphson(w_init, X, epsilon)
    
    w_star = w_iters[-1]
    
    return w_star, (w_iters, values)

def postprod_grid_logit(data, w_star, grid, xx):
    
    def logistic_prob(x, w):
        xaug = np.ones(3)
        xaug[1:] = x
        sigma = expit(w @ xaug)
        return sigma
    
    post_grid = np.array([
        logistic_prob(x, w_star)
        for x in grid
    ])

    post_grid = post_grid.reshape(*xx.shape)
    
    return post_grid

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
                   cmap='jet')
    cbar_ = fig.colorbar(cf_)
    cbar_.set_label("Posterior probability $p(y=1|x)$")
    grid = grid.reshape(np.prod(xx.shape),2)
    #w=np.linalg.inv(Sigma)@(mu1-mu0)
    #b=np.log(alpha/(1-alpha))-0.5*(mu1-mu0).T@(np.linalg.inv(Sigma))@(mu1+mu0)
    
    #m=-w[0]/w[1]
    #c=-b/w[1]
    
    #xd = np.array([xmin, xmax])
    #yd = m*xd + c
    #Train
    #axs[0].plot(xd, yd, 'k', lw=1, ls='--')
    axs[0].scatter(data[:, 0], data[:, 1], c=data[:, 2], s=20 , alpha=1)
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
    #xd = np.array([xmin, xmax])
    #yd = m*xd + c
    cf_ = axs[1].contourf(xx, yy, postprod_grid, levels=40,
                   zorder=-1, alpha=0.9,
                   cmap='jet')
    #axs[1].plot(xd, yd, 'k', lw=1, ls='--')
    axs[1].scatter(test[:, 0], test[:, 1], c=test[:, 2], s=10 , alpha=0.8)
    axs[1].scatter(mu0[0], mu0[1], label="mu_0")
    axs[1].scatter(mu1[0], mu1[1], label="mu_1")
    axs[1].set_ylabel(r'$x_2$')
    axs[1].set_xlabel(r'$x_1$')
    axs[1].set_title('Test set')
    plt.show()

if __name__ == '__main__':
    
    data1 = pd.read_csv('data/trainA',header=None,sep=' ', names=["x1", "x2", "y"])
    data11 = pd.read_csv('data/testA', header = None, sep = ' ', names=['x1', 'x2', 'y'])
    data2 = pd.read_csv('data/trainB',header=None,sep=' ', names=["x1", "x2", "y"])
    data22 = pd.read_csv('data/testB' , header = None, sep = ' ', names = ['x1','x2','y'])
    data3 = pd.read_csv('data/trainC',header=None,sep=' ', names=["x1", "x2", "y"])
    data33 = pd.read_csv('data/testC',header=None,sep=' ', names=["x1", "x2", "y"])
    data = [(data1,data11), (data2,data22), (data3, data33)]
    
    char ='A'
    for (dat, test) in data:
        X, y = add_noise(np.array(dat.iloc[:,0:2])), np.squeeze(np.array(dat.iloc[:,2:]))

        w_star, (w_iters, values) = logistic_regression(X, y)

        grid, xx, yy, xlim, ylim = create_grid(np.array(dat))
    
        postprob_grid = postprod_grid_logit(dat, w_star, grid, xx)
        
        mu1 = np.mean(dat[dat['y'] == 0], axis=0).iloc[0:2]
        mu2 = np.mean(dat[dat['y'] == 1], axis=0).iloc[0:2]
    
        #fig = plot(dat, mu1, mu2, 0, postprob_grid, xx, yy, xlim, ylim)
        plot_maestrati(np.array(dat), np.array(test), mu1, mu2, postprob_grid)
        
        fig.savefig('./images/train_logit_%s.png' % char)
        i = ord(char[0])
        i += 1
        char = chr(i)