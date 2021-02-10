import numpy as np
import pandas as pd

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

def plot_graphs(y,t,title,ylabel,xlabel,i=0,save=False,file_path="",filename="",label="",show_label=True,scale=False,
                color=None,ls="solid"):

    if filename:
        data = pd.read_csv(filename,sep=' ')
    ax = plt.figure(i)
    if t is -1:
        if color is None:
            plt.plot(y,label=label,ls=ls)
        else:
            plt.plot(y,label=label,color=color,ls=ls)
    else:
        if color is None:
            plt.plot(t,y,label=label,ls=ls)
        else:
            plt.plot(t,y,label=label,color=color,ls=ls)
    if scale:
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    if show_label:
        plt.legend()
    plt.draw()
    if save:
        plt.savefig(file_path+".png")

def zoom(i,x,y,X1, X2, Y1, Y2):
    y0 = y[0]
    y1 = y[1]
    y2 = y[2]
    y3 = y[3]
    plt.figure(i)
    ax = plt.axes()
    axins = zoomed_inset_axes(ax, 10, loc=3) # zoom = 2

    axins.plot(x,y0,color="C0")
    axins.plot(x,y1,color="C1")
    axins.plot(x,y2,color="C2")
    axins.plot(x,y3,color="C3")
    axins.set_xlim(X1, X2)
    axins.set_ylim(Y1, Y2)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=1, loc2=2, fc='b', ec='0.4')
    plt.draw()


def save_plot(save,path="",i=None):

    if path and save:
        if i is not None:
            plt.figure(i)
        plt.savefig(path+".png")


def horizontal_line(y,label="",c='b'):
    plt.axhline(y, linestyle='--',label=label,c=c)
    plt.legend()

def plot_grid(y,t,i,label="",save=False,path="",color=None):
    plt.figure(i)
    plt.plot(t,y,label=label,ls="dashed",color=color)
    plt.legend()
    if save:
        save_plot(save,path)


def text_on_plot(string, i=None):
    if i is not None:
        plt.figure(i)
    ax = plt.gca()
    plt.text(0.3, 0.05, string, horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes,style="oblique")#,
              # bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})

def show_plots(plot):
    if plot:
        plt.show()
    plt.clf()
    plt.cla()
    plt.close('all')

def legend_loc(n,i=None):
    if i is not None:
        plt.figure(i)
    plt.legend(loc=n)

def plot_2_axis(xlabel,ylabel1,y1,ylabel2,y2,t,title="",i=0,ax1=None,fig=None,conf=1):
    if not ax1 or not fig:
        fig, ax1 = plt.subplots()

    if conf == 1:
        color1 = 'k'
        color2 = 'g'
        line = "dashed"
    elif conf == 2:
        color1 = 'r'
        color2 = 'b'
        line = "solid"

    # ax1.set_xlabel('time (s)')
    ax1.set_ylabel(ylabel1, color=color1)
    ax1.plot(t, y1, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(ylabel2, color=color2)  # we already handled the x-label with ax1
    ax2.plot(t, y2, color=color2,linestyle=line)
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.title(title)
    return ax2


# def subplots(title,x,y1_0,y1_1,ylabel1_0,ylabel1_1,yaxis1,yaxis2,xlabel):
#     fig, axs = plt.subplots(2)
#     fig.suptitle('Vertically stacked subplots')
#     axs[0].plot(x, y1)
#     axs[1].plot(x, y2)
#     ax2[1].set_ylabel(ylabel2)
#     ax2[0].set_ylabel(ylabel1)
#     ax2[1].sex_xlabel(xlabel)

def subplots_plot(title,x,array_y,x_label,array_ylabel,disposition,i=0):
    """
    title - title of the full experiment
    x - list with x axis for time
    array_y - list with all the plots for the y axis
    disposition - list with the configuration for the plots
        [1,2,2] means 1st subplot with array_y[0], 2nd subplot with array[1] and
        array[2] and 3rd subplot with arrays[3,4]
    array_ylabel - list with the labels for the axis of the different ys
    """
    n = len(disposition)
    fig,a =  plt.subplots(n,1)

    j = k = 0
    while(disposition):
        i = disposition.pop(0)
        if i == 1:
            a[j].plot(x,array_y[k])
            j += 1; k+= 1;
        elif i ==2:
            if 'rank' in array_ylabel[k] or "Condition Number" in array_ylabel[k]:
                conf = 1
            else:
                conf = 2
            ret = plot_2_axis(x_label,array_ylabel[k],array_y[k],array_ylabel[k+1],array_y[k+1],x,ax1=a[j],fig=fig,conf=conf)
            if 'Range' in array_ylabel[k] or 'Range Rate' in array_ylabel[k]:
                ax2 = ret
            j += 1; k+= 2;
        else:
            print("ERROR\ndisposition is either 1 or 2.")
    a[0].set_title(title)
    # L=a[0].axvline(x=415,linestyle='dotted',color='g',linewidth=2,label="ola")
    # D=a[0].axvline(x=970,linestyle='dotted',color='y',linewidth=2,label="ola")
    # a[0].axvline(x=5980,linestyle='dotted',color='g',linewidth=2)
    # a[0].axvline(x=6700,linestyle='dotted',color='y',linewidth=2)
    # a[1].axvline(x=415,linestyle='dotted',color='g',linewidth=2)
    # a[1].axvline(x=970,linestyle='dotted',color='y',linewidth=2)
    # a[1].axvline(x=5980,linestyle='dotted',color='g',linewidth=2)
    # a[1].axvline(x=6700,linestyle='dotted',color='y',linewidth=2)
    a[1].set_xlabel('time (s)')
    # a[0].legend([L,D],['Beginning of LOS','End of LOS'],loc=1)

def trajectory_plot(title,x,y,z,xlabel,ylabel,zlabel,i=0,label=""):

    fig = plt.figure(i)
    mpl.rcParams['legend.fontsize'] = 10
    ax = fig.gca(projection='3d')
    # for i,passage in enumerate(passages):
    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.plot(x, y, z, label=label)
    if label:
        ax.legend()



# # Uncomment this code and run this script just to build the plots
# data = '../Data/results/07-04/CDUKF.csv'
# args = ('pos','vel')
# M = 60
# n = 200
# plot_graphs(data,args,M,n,save=False,file_path='../Data/results/')
# y,t,title,ylabel,xlabel,i=0,save=False,file_path="",filename=""
