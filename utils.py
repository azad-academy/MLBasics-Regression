'''
Author: Azad-Academy
jrs@azaditech.com
https://www.azaditech.com

'''
import sys
from matplotlib import pyplot as plt
import numpy as np
import math
import matplotlib


def assemble_points(X,Y):
    model_pts = np.array(np.zeros((len(X),2)))    # Assembling a set of points to plot the model line
    for i in range(len(X)):
        model_pts[i][0] = X[i]
        model_pts[i][1] = Y[i]
    return model_pts

def plot_data(X,Y,model_pts=None,classification=False,canvas=None,xtitle=None,ytitle=None,colors=None,plt_title=None):
    
    
    if(classification):
        if(colors is None):
            colors = np.random.rand(max(Y)+1,3)
    
        
    if(canvas is None):
        fig, ax = plt.subplots(figsize=(11,8))
    else:
        ax = canvas
        ax.cla()
    
    if(plt_title is not None):
        ax.set_title(plt_title)

    for i in range(len(Y)):
        if(classification):
            ax.scatter(X[i,0],X[i,1],color=colors[Y[i]],alpha=0.5)
        else:
            ax.scatter(X,Y,color=colors,alpha=0.5)
        
    if(model_pts is not None):
        ax.plot(model_pts[:,0],model_pts[:,1],'r',linewidth=4)
    
    if(xtitle is not None):
        ax.set_xlabel(xtitle,fontweight='bold',fontsize=16)
    else:
        ax.set_xlabel(r'$X_1^i$',fontweight='bold',fontsize=16)
    if(xtitle is not None):
        ax.set_ylabel(ytitle,fontweight='bold',fontsize=16)
    else:    
        ax.set_ylabel(r'$Y^i$',fontweight='bold',fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def linear_save_animation(X,Y,t):
    import PIL
    LR = linear_model.SGDRegressor()
    fig,ax = plt.subplots(figsize=(11,8))
    imgs = []
    for i in range(t):
        LR.partial_fit(X,Y)
        y_hat = LR.predict(X)
        model_pts = assemble_points(X,y_hat)
        plot_data(X,Y,model_pts,canvas=ax)
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        img = PIL.Image.frombytes('RGB',canvas.get_width_height(), canvas.tostring_rgb())
        imgs.append(img)
    imgs[0].save('regression.gif',format='GIF',append_images=imgs,save_all=True,duration=200,loop=0)


def gen_non_linear_data(N):
    X = 0.55*np.random.normal(size=N)+0.5

    Xt = 0.65*X-0.35
    X = X.reshape((N,1))

    Y = -(6 * Xt**2 + 0.1*Xt + 0.1) + 0.2 * np.random.normal(size=N)
    Y = np.exp(Y) + 0.05 * np.random.normal(size=N)
    Y /= max(np.abs(Y))
    return X, Y

def plot_gaussian(X,Y,y_hat_mean,y_hat_std,canvas=None,title=None):
    if(canvas==None):
        figure,ax = plt.subplots()
    else:
        ax = canvas
    if(title is not None):
        ax.set_title(title)
    indices = np.argsort(X[:,0],axis=0)
    X_sorted = np.array(X)[indices]
    Y_sorted=np.array(Y)[indices]
    ax.scatter(X_sorted,Y_sorted)
    ax.plot(X_sorted,y_hat_mean[indices],'r.')
    ax.fill_between(
        X_sorted.ravel(),
        np.squeeze(y_hat_mean[indices]) - 1.96 * np.squeeze(y_hat_std[indices]),
        np.squeeze(y_hat_mean[indices]) + 1.96 * np.squeeze(y_hat_std[indices]),
        color="tab:orange",
        alpha=0.5,
        label=r"95% confidence interval")