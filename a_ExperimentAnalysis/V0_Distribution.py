import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
def myBinsEdge(data):
    return np.histogram_bin_edges(data, bins='scott')

def gaussFit(data, xmin, xmax):
    kde = gaussian_kde(data)
    x = np.linspace(xmin, xmax, num=100)
    y = kde(x)
    expected = (1, .2, 0.7)
    params, cov, infodict, mesg, ier = curve_fit(gauss, x, y, expected, full_output=True)
    sigma = np.sqrt(np.diag(cov))
    return params, sigma, infodict['fvec']

def gauss(x, mu, sigma, A):
    return A * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)


def plotVmaxDistribution(trajs):
    framerate=25
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.22, hspace=0.28)
    widths = [1.2, 3.4, 4.5, 5.6]
    motivations = ['high', 'low']
    for traj in trajs:
        print("Plot: %s" % os.path.basename(traj))
        data = np.loadtxt(traj)
        # Get the parameters of the trajectory
        traj = os.path.basename(traj)
        index = int(traj.split('_')[4])
        # the condition can be crowd(c):0 or queue(q):1
        condition = traj.split('_')[5]
        if condition == 'c':
            condition = 0
        else:
            condition = 1
        # the width of the exp
        width = float(traj.split('_')[6]) / 10
        # the motivation can be h0(0) or h-(1)
        motivation = traj.split('_')[7][0:2]
        if motivation == 'h0':
            motivation = 0
        else:
            motivation = 1
        ax = axes[motivation, widths.index(width)]
        ids = np.unique(data[:, 0])
        vmaxs=[]
        for id in ids:
            data_id = data[data[:, 0] == id]
            data_id = data_id[np.argsort(data_id[:, 1])]
            vmax=np.max(data_id[:,6])
            vmaxs.append(vmax)
        bins = myBinsEdge(vmaxs)
        sns.histplot(vmaxs, bins=bins, ax=ax, stat='density',
                     facecolor='steelblue')
        sns.kdeplot(vmaxs, clip=(0, 3), ax=ax, color='indianred',
                    bw_method='scott', bw_adjust=1, linewidth=3,
                    label='kde_{}_{}'.format(motivations[motivation], width))
        # fit parameters
        x = np.linspace(0.5, 2.5, num=100)
        params, sigma, fvec = gaussFit(vmaxs,0.5 ,2.5)
        error = np.sum([i * i for i in fvec])
        ax.plot(x, gauss(x, *params), color='black', linestyle='--',
                        label='mu:{:.2f} sigma{:.2f} A{:.2f}'.format(*params), linewidth=2)

        ax.legend(loc='best')
        #ax.set_xlim(1, 4)
        ax.set_title("{} motivation $w$={:.1f} m".format(motivations[motivation], width),
                         fontsize=25)
        ax.set_xlabel('$\\bar{Vmax}$', fontsize=25)
        ax.set_ylabel('Density', fontsize=25)
    figname = "V0Distributions.png"
    plt.savefig(figname)
    plt.cla()
    plt.close()


if __name__=='__main__':
    cwd = os.getcwd()
    trajs = glob.glob("%s/TrajInfo/*.txt" % cwd)
    plotVmaxDistribution(trajs)
