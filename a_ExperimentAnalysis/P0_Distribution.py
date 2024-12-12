# 2024.03.26
# Qiancheng Xu
# Analyzing the distributions of pedestrians' free pushing intensity

import os
import glob
import numpy as np
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy import stats, integrate
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import pandas as pd
import statsmodels.api as smapi
from matplotlib.legend_handler import HandlerLine2D

def updateline(handle, orig):
    handle.update_from(orig)
    handle.set_markersize(15)

# some basic definition##############################################
target = (0, 0)  # the middle point of the exit
minFrame = 25  # ignore the unstable section
bodySize = 0.18  # the size of the pedestrian
sm = cm.ScalarMappable(cmap=cm.jet)  # color map for plotting
sm.set_clim(vmin=0, vmax=7)


####################################################################
# basic functions
def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def GenerateAllDataFile():
    print("INFO: Generate AllData.txt")
    Alldata = []
    trajs = glob.glob('TrajInfo/*.txt')
    for traj in trajs:
        print("Get data from {}".format(os.path.basename(traj)))
        # the index of each exp is unique
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
        data = np.loadtxt(traj)
        for d in data:
            Alldata.append([index, condition, width, motivation, *d])
    Alldata = np.array(Alldata)
    # write the data down
    f = open('AllData.txt', 'w')
    f.write('# framerate: 25\n')
    f.write('# 0: index\n')
    f.write('# 1: condition crowd(0) queue(1)\n')
    f.write('# 2: width(m)\n')
    f.write('# 3: motivation h0(0) h-(1)\n')
    f.write('# 4: id\n')
    f.write('# 5: frame\n')
    f.write('# 6: x\n')
    f.write('# 7: y\n')
    f.write('# 8: plevel (4:strong pushing 3:Mild pushing 2:Just walking 1:Falling behind)\n')
    f.write('# 9: density (-1: not in the measurement area)\n')
    f.write('# 10: speed\n')
    f.write('# 11: x_direction (0: speed is 0)\n')
    f.write('# 12: y_direction (0: speed is 0)\n')
    f.write('# 13: space (100: no peds in the direction of movement)\n')
    f.write('# 14: infid (-1: no peds in the direction of movement)\n')
    f.write(
        '#index\tcondition\twidth\tmotivation\tid\tframe\tx\ty\tplevel\tdensity\tspeed\tx_direction\ty_direction\tspace\tinfid\n')
    for d in Alldata:
        f.write(
            '{:.0f}\t{:.0f}\t{:.1f}\t{:.0f}\t{:.0f}\t{:.0f}\t{:.4f}\t{:.4f}\t{:.0f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.0f}\n'.format(
                *d))
    f.close()
    return Alldata


# remove the data that are not suitable
def CleanData(data):
    print("INFO: Remove the data that are not in the measurement area")
    # plevel should be bigger than 0 (0 means the pedsetrian is not in the consideration)
    data = data[data[:, 8] > 0]
    # The pedestrian should be in the measurement area（data[:,7] is the y position）
    data = data[data[:, 7] > 0]
    data = data[data[:, 7] < 6]
    # check the density just in case
    data = data[data[:, 9] > 0]
    # the first several seconds should be ignored (data[:,5] is the frame)
    # data=data[data[:,5]>minFrame]
    return data


# set bins based on scott Rule (the result is good)
def myBinsEdge(data):
    return np.histogram_bin_edges(data, bins='scott')


def gauss(x, mu, sigma, A):
    return A * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)


# Calculate the mean pushing level of the pedestrian and the relevant information
def GeneratePlevelInfo(data):
    indexs = np.unique(data[:, 0])
    plevelInfoS = []
    for index in indexs:
        data_traj = data[data[:, 0] == index]
        ids = np.unique(data_traj[:, 4])
        trajInfo = data_traj[0][:4]
        for id in ids:
            nums = []
            # the mean plevel of the pedestrian
            average = 0
            data_id = data_traj[data_traj[:, 4] == id]
            total = len(data_id)
            # the percent of each plevel and the mean plevel
            for i in range(1, 5):
                num = len(data_id[data_id[:, 8] == i])
                percent = num / total
                nums.append(percent)
                average += i * percent
            # the inipos(x,y) of the pedestrian
            iniPos = data_id[0, 6:8]
            # the start time and leave time of the pedestrian
            EvaTimes = [data_id[0, 5], data_id[-1, 5]]
            # the mean density and speed
            mean_density = np.mean(data_id[:, 9])
            mean_speed = np.mean(data_id[:, 10])
            plevelInfoS.append(
                [*trajInfo, id, *nums, average, *iniPos, *EvaTimes, mean_density, mean_speed])
    plevelInfoS = np.array(plevelInfoS)
    # write the pedPlevelInfos
    f = open('PushingIntensityInfos.txt', 'w')
    f.write('# the proportion of plevel for each pedestrian\n')
    f.write('# 0: index\n')
    f.write('# 1: condition crowd(0) queue(1)\n')
    f.write('# 2: width(m)\n')
    f.write('# 3: motivation h0(0) h-(1)\n')
    f.write('# 4: ID\n')
    f.write('# 5-8: plevels\n')
    f.write('# 9: Average plevel\n')
    f.write('# 10-11: IniPos(x,y)\n')
    f.write('# 12: start frame\n')
    f.write('# 13: leave frame\n')
    f.write('# 14: mean_density [1/m2]\n')
    f.write('# 15: mean_speed [m/s]\n')
    f.write('#index[0]\tcondition[1]\twidth[2]\tmotivation[3]\tID[4]\t'
            'p1\tp2\tp3\tp4\tAvePlevel\tiniX\tiniY\tstartF\tleaveF\tmDensity\tmSpeed\n')
    for plevelInfo in plevelInfoS:
        f.write('{}\t{}\t{}\t{}\t{}\t'
                '{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t'
                '{:.3f}\t{:.3f}\t{}\t{}\t{:.3f}\t{:.3f}\t\n'
                .format(*plevelInfo))
    f.close()
    return plevelInfoS


def bimodalFit(data, xmin, xmax):
    kde = gaussian_kde(data)
    x = np.linspace(xmin, xmax, num=100)
    y = kde(x)
    expected = (2, .2, 0.7, 3, .2, 0.7)
    params, cov, infodict, mesg, ier = curve_fit(bimodal, x, y, expected, full_output=True)
    sigma = np.sqrt(np.diag(cov))
    return params, sigma, infodict['fvec']


def gaussFit(data, xmin, xmax):
    kde = gaussian_kde(data)
    x = np.linspace(xmin, xmax, num=100)
    y = kde(x)
    expected = (2, .2, 0.7)
    params, cov, infodict, mesg, ier = curve_fit(gauss, x, y, expected, full_output=True)
    sigma = np.sqrt(np.diag(cov))
    return params, sigma, infodict['fvec']


# Return the p value of the T test for linear regression
def t_test(X, y):
    X = smapi.add_constant(X)
    model = smapi.OLS(y, X).fit()
    return model.pvalues[X.columns.values[1]]


# Analysis functions#######################################################
# Plot the histogram and kde curve fot distributions of mean pushing level in each experiment
def TrajDistribution(data):
    # Plot the distribution in each experiment
    ymax=[[1.2,1.2,1.2,1.2],[3,3,3,3]]
    textPos=[[(2.3,0.95),(2.3,0.95),(2.3,0.95),(2.3,0.95)],[(2.3,2.375),(2.3,2.375),(2.3,2.375),(2.3,2.375)]]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.3, hspace=0.37)
    motis = np.unique(data[:, 3])
    motivations = ['high', 'low']
    # record the bimodalfit parameters
    f = open('BimodalFitParameters.txt', 'w')
    f.write('# motivation h0(0) h-(1)\n')
    f.write(
        "#moti[0]\twidth[1]\tmu1[2]\tsigma1[3]\tA1[4]\tmu2[5]\tsigma2[6]\tA2[7]\terror[8]\tp1/(p1+p2)[9]\n")
    for i, moti in enumerate(motis):
        data_moti = data[data[:, 3] == moti]
        widths = np.unique(data_moti[:, 2])
        for j, width in enumerate(widths):
            data_width = data_moti[data_moti[:, 2] == width]
            bins = myBinsEdge(data_width[:, 9])
            sns.histplot(data_width[:, 9], bins=bins, ax=axes[i, j], stat='density',
                         facecolor='steelblue')
            sns.kdeplot(data_width[:, 9], clip=(1, 4), ax=axes[i, j], color='indianred',
                        bw_method='scott', bw_adjust=1, linewidth=6)
            # fit parameters
            x = np.linspace(1, 4, num=100)
            params, sigma, fvec = bimodalFit(data_width[:, 9], 1, 4)
            error = np.sum([i * i for i in fvec])
            prop1 = integrate.quad(gauss, 1, 4, args=(params[0], params[1], params[2]))[0]
            prop2 = integrate.quad(gauss, 1, 4, args=(params[3], params[4], params[5]))[0]
            f.write(
                '{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(
                    motis[i],
                    width,
                    *params,
                    error, prop1 / (prop1 + prop2)))
            axes[i][j].plot(x, bimodal(x, *params), color='black', linestyle='--', linewidth=4)
            axes[i][j].set_xlim(1, 4)
            axes[i][j].set_ylim(0,ymax[i][j])
            # axes[i][j].legend(loc='best', fontsize=20)
            fitParaText=('$A_1:{:.2f}~~~~A_2:{:.2f}$\n$\\mu_1:{:.2f}~~~~\\mu_2:{:.2f}$\n$\\sigma_1:{:.2f}~~~~\\sigma_2:{:.2f}$'
                         .format(params[2],params[5],params[0],params[3],params[1],params[4]))
            axes[i][j].text(textPos[i][j][0],textPos[i][j][1],fitParaText,bbox=dict(facecolor='grey', alpha=0.1),fontsize=12)
            axes[i][j].set_title("{}  {:.1f} m".format(motivations[i], width),
                                 fontsize=20)
            axes[i][j].set_xlabel('$P_i^0$', fontsize=20)
            axes[i][j].set_ylabel('Density', fontsize=20)
            axes[i][j].tick_params(axis='both', which='major', labelsize=15)
            axes[i][j].tick_params(axis='both', which='minor', labelsize=15)
    plt.savefig('P0DistHist.png', dpi=300)
    plt.close()
    plt.cla()
    plt.clf()
    f.close()
    # plot the kde curves of experiments together
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.1, top=0.95)
    motivations = ['low', 'high']
    ls = ['-', '--']
    for i, moti in enumerate([1, 0]):
        data_moti = data[data[:, 3] == moti]
        widths = np.unique(data_moti[:, 2])
        for j, width in enumerate(widths):
            data_width = data_moti[data_moti[:, 2] == width]
            sns.kdeplot(data_width[:, 9], label='{}  {} m'.format(motivations[i], width),
                        clip=(1, 4),
                        ax=ax,
                        color=sm.to_rgba(i * 4 + j), bw_method='scott', bw_adjust=1,
                        linestyle=ls[i], linewidth=5)
    ax.set_xlim(1, 4)
    ax.set_ylim(0,2.5)
    ax.set_xticks([1,2,3,4])
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel('$P_i^0$', fontsize=20)
    ax.set_ylabel('Density', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    plt.savefig('P0DistKde.png', dpi=300)
    plt.close()
    plt.cla()
    plt.clf()


# plot the relations between mean pushing level and relevant features
def PlotRelations(pInfos):
    # calculate information:
    plotInfos = []
    for info in pInfos:
        index = info[0]
        moti = 0
        if info[3] == 1:
            moti = 1
        plevel = info[9]
        iniDis = np.sqrt(info[10] * info[10] + info[11] * info[11])
        travelTime = (info[13] - info[12]) / 25
        mdensity = info[14]
        mspeed = info[15]
        plotInfos.append([index, moti, plevel, iniDis, travelTime, mdensity, mspeed])
    plotInfos = np.array(plotInfos)
    plotInfos = pd.DataFrame(plotInfos,
                             columns=['index', 'moti', 'plevel', 'iniDistance', 'travelTime',
                                      'mDensity', 'mSpeed'])


    # for all:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.85,wspace=0.4, hspace=0.9)
    ylims=[[0.9,4.1],[-5,70],[0,15],[0,0.6]]
    fNames = ['iniDistance', 'travelTime', 'mDensity', 'mSpeed']
    labelNames = ['$P_i^0$', '$d_i^{\\rm{initial}}$ [m]', '$t_i^{\\rm{travel}}$ [s]',
                  '$\\bar{\\rho}_i$ [$\\rm{m}^{-2}$]',
                  '$\\bar{v}_i$ [$\\rm{m}\\cdot \\rm{s}^{-1}$]']
    infoH = plotInfos[plotInfos['moti'] == 0]
    infoL = plotInfos[plotInfos['moti'] == 1]
    for i in range(4):
        if i == 0:
            t_high = t_test(infoH[fNames[i]], infoH['plevel'])
            t_low = t_test(infoL[fNames[i]], infoL['plevel'])
            sns.regplot(data=infoH, x=fNames[i], y='plevel', ax=axes[int(i / 2)][i % 2],
                        fit_reg=False,
                        color='indianred',
                        marker='o',
                        scatter_kws={'alpha': 0.15}, line_kws={'lw': 4})
            sns.regplot(data=infoL, x=fNames[i], y='plevel', ax=axes[int(i / 2)][i % 2],
                        fit_reg=False,
                        color='steelblue',
                        marker='x',
                        scatter_kws={'alpha': 0.15}, line_kws={'lw': 4, 'ls': '--'})
            sns.regplot(data=infoL, x=fNames[i], y='plevel', ax=axes[int(i/2)][i%2], scatter=False,
                        label='low  $p$-value:{:.2f}'.format(t_low),
                        color='steelblue',
                        marker='x',
                        scatter_kws={'alpha': 0.3}, line_kws={'lw': 4})
            sns.regplot(data=infoH, x=fNames[i], y='plevel', ax=axes[int(i / 2)][i % 2],
                        scatter=False,
                        label='high $p$-value:{:.2f}'.format(t_high),
                        color='indianred',
                        marker='o',
                        scatter_kws={'alpha': 0.3}, line_kws={'lw': 4, 'ls': '--'})
            axes[int(i/2)][i%2].set_xlabel("{}".format(labelNames[i + 1]), fontsize=20)
            axes[int(i/2)][i%2].set_ylabel('$P_i^0$', fontsize=20)

        else:
            t_high = t_test(infoH['plevel'], infoH[fNames[i]])
            t_low = t_test(infoL['plevel'], infoL[fNames[i]])
            sns.regplot(data=infoH, x='plevel', y=fNames[i], ax=axes[int(i / 2)][i % 2],
                        fit_reg=False,
                        color='indianred',
                        marker='o',
                        scatter_kws={'alpha': 0.15}, line_kws={'lw': 4})
            sns.regplot(data=infoL, x='plevel', y=fNames[i], ax=axes[int(i / 2)][i % 2],
                        fit_reg=False,
                        color='steelblue',
                        marker='x',
                        scatter_kws={'alpha': 0.15}, line_kws={'lw': 4})
            sns.regplot(data=infoL, x='plevel', y=fNames[i], ax=axes[int(i/2)][i%2], scatter=False,
                        label='low  $p$-value:{:.2f}'.format(t_low),
                        color='steelblue',
                        marker='x',
                        scatter_kws={'alpha': 0.3}, line_kws={'lw': 4})
            sns.regplot(data=infoH, x='plevel', y=fNames[i], ax=axes[int(i / 2)][i % 2],
                        scatter=False,
                        label='high $p$-value:{:.2f}'.format(t_high),
                        color='indianred',
                        marker='o',
                        scatter_kws={'alpha': 0.3}, line_kws={'lw': 4, 'ls': '--'})
            axes[int(i/2)][i%2].set_xlabel('$P_i^0$', fontsize=20)
            axes[int(i/2)][i%2].set_ylabel("{}".format(labelNames[i + 1]), fontsize=20)
        axes[int(i/2)][i%2].legend(bbox_to_anchor=(0.5,1.25),loc=10, fontsize=20,handler_map={plt.Line2D: HandlerLine2D(update_func=updateline)})
        axes[int(i/2)][i%2].tick_params(axis='both', which='major', labelsize=15)
        axes[int(i/2)][i%2].tick_params(axis='both', which='minor', labelsize=15)
        axes[int(i / 2)][i % 2].set_ylim(ylims[i])
    plt.savefig("FeatureRelations.png", dpi=300)
    plt.cla()
    plt.clf()
    plt.close()



    # Plot the distributions of the features
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.8,wspace=0.5)
    ylims=[[0,1.8],[0,0.21],[0,0.03],[0,0.32],[0,7]]
    xlims=[[1,4],[0,10],[0,100],[0,15],[0,0.6]]
    infoH = plotInfos[plotInfos['moti'] == 0]
    infoL = plotInfos[plotInfos['moti'] == 1]
    fNames = ['plevel', 'iniDistance', 'travelTime', 'mDensity', 'mSpeed']
    labelNames = ['$P_i^0$', '$d_i^{\\rm{initial}}$ [m]', '$t_i^{\\rm{travel}}$ [s]',
                  '$\\bar{\\rho}_i$ [$\\rm{m}^{-2}$]',
                  '$\\bar{v}_i$ [$\\rm{m}\\cdot \\rm{s}^{-1}$]']
    for i in range(5):
        feature = fNames[i]
        sns.kdeplot(infoL[feature], label='low', ax=axes[i], color='steelblue', bw_method='scott',
                    bw_adjust=1, fill=0.5,linewidth=5)
        sns.kdeplot(infoH[feature], label='high', ax=axes[i], color='indianred',
                    bw_method='scott', bw_adjust=1, fill=0.5, linewidth=5,linestyle='--')
        axes[i].set_xlabel("{}".format(labelNames[i]), fontsize=20)
        axes[i].set_ylabel('Density', fontsize=20)
        axes[i].tick_params(axis='both', which='major', labelsize=15)
        axes[i].tick_params(axis='both', which='minor', labelsize=15)
        axes[i].set_ylim(ylims[i])
        axes[i].set_xlim(xlims[i])
    axes[3].legend(bbox_to_anchor=(0.5,1.2),loc=10, fontsize=20,ncol=2,handleheight=1,handlelength=10)
    plt.savefig("FeatureDistributions.png", dpi=300)
    plt.cla()
    plt.clf()
    plt.close()

if __name__ == '__main__':
    cwd = os.getcwd()
    print("INFO: Current work directory is {}".format(cwd))
    # import the data
    if os.path.exists("AllData.txt"):
        print("INFO: AllData.txt is already exist")
        Alldata = np.loadtxt("AllData.txt")
    else:
        Alldata = GenerateAllDataFile()
    # clean the data (only consider the data in the measurement area)
    Alldata = CleanData(Alldata)
    # Calculate the mean pushing level
    plevelInfos = GeneratePlevelInfo(Alldata)
    # Plot the distribution of the mean pushing level in each experiment
    TrajDistribution(plevelInfos)
    # Plot the relationships between the mean pushing level and other features
    PlotRelations(plevelInfos)
