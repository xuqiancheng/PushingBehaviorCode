# Purpose: Analyzing the ML result of Plevel
# Author: Qiancheng Xu
# Date: 2024.04.19

import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import seaborn as sns
from math import log

def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def GetRFResult(Folder):
    RFResults = []
    for subfolder in os.listdir(Folder):
        tAnti = float(subfolder.split('_')[1])
        N = float(subfolder.split('_')[-1])
        path = Folder + '/' + subfolder
        fileAll = path + '/' + 'RFClassification_all.txt'
        AllData = np.loadtxt(fileAll)
        trainAcc = AllData[0, 1]
        NoPushF1 = AllData[1, 3]
        PushF1 = AllData[2, 3]
        MacroF1 = AllData[3, 3]
        WeightF1 = AllData[4, 3]
        Accuracies = [trainAcc, NoPushF1, PushF1, MacroF1, WeightF1]
        RFResults.append([tAnti, N, *Accuracies])
    RFResults = np.array(RFResults)
    RFResults = RFResults[RFResults[:, 1].argsort()]
    saveFileName = Folder.split('/')[-1] + 'Result.txt'
    f = open(saveFileName, 'w')
    f.write(
        "# tAnti[0]\tN[1]\ttrainAccuracy[2]\tNoPushF1[3]\tPushF1[4]\tMacroF1[5]\tWeightedF1[6]\n")
    for RFResult in RFResults:
        f.write("{:.0f}\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(*RFResult))
    f.close()
    return (RFResults)


def ShowNTedency(data, name):
    fig, ax = plt.subplots(1, 5, figsize=(30, 6))
    plt.title(name)
    # Plot the tedency of N
    AccNames = ['trainAccuracy', 'NoPushF1', 'PushF1', 'MacroF1', 'WeightedF1']
    for i in range(5):
        tAntis = np.unique(data[:, 0])
        for tAnti in tAntis:
            dataP = data[data[:, 0] == tAnti]
            ax[i].plot(dataP[:, 1], dataP[:, 2 + i], label=tAnti)
        ax[i].set_xlabel('N', fontsize=15)
        ax[i].set_ylabel(AccNames[i], fontsize=15)
        ax[i].legend(loc='best', fontsize=15)
        ymin = np.min(data[:, 2 + i]) - 0.05
        if i == 0:
            ymax = 1.05
        else:
            ymax = ymin + 0.20
        ax[i].set_ylim(ymin, ymax)
    figname = name + 'NTedency.png'
    plt.savefig(figname, dpi=300)


def ShowtAntiTedency(data, name):
    fig, ax = plt.subplots(1, 5, figsize=(30, 6))
    plt.title(name)
    # Plot the tedency of N
    AccNames = ['trainAccuracy', 'NoPushF1', 'PushF1', 'MacroF1', 'WeightedF1']
    for i in range(5):
        Ns = np.unique(data[:, 1])
        for N in Ns:
            dataP = data[data[:, 1] == N]
            dataP = dataP[dataP[:, 0].argsort()]
            ax[i].plot(dataP[:, 0], dataP[:, 2 + i], label=N)
        ax[i].set_xlabel('tAnti', fontsize=15)
        ax[i].set_ylabel(AccNames[i], fontsize=15)
        ax[i].legend(loc='best', fontsize=15)
        ymin = np.min(data[:, 2 + i]) - 0.05
        if i == 0:
            ymax = 1.05
        else:
            ymax = ymin + 0.20
        ax[i].set_ylim(ymin, ymax)
    figname = name + 'tAntiTedency.png'
    plt.savefig(figname, dpi=300)


def ShowNormEffect(data, dataNorm):
    N = 8
    tAnti = 0
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    data = data[np.logical_and(data[:, 0] == tAnti, data[:, 1] == N)]
    dataNorm = dataNorm[np.logical_and(dataNorm[:, 0] == tAnti, dataNorm[:, 1] == N)]
    xs = range(5)
    ax.plot(xs, data[0, 2:], color='red', label='NoNorm', ls='', marker='o')
    ax.plot(xs, dataNorm[0, 2:], color='blue', label='WithNorm', ls='', marker='*')
    ax.set_xticks([0, 1, 2, 3, 4], ['trainAccuracy', 'NoPushF1', 'PushF1', 'MacroF1', 'WeightedF1'])
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='best')
    figname = 'NormEffect.png'
    plt.savefig(figname, dpi=300)


def ShowWeights(Folder):
    RFWeights = []
    for subfolder in os.listdir(Folder):
        tAnti = float(subfolder.split('_')[1])
        N = int(subfolder.split('_')[-1])
        path = Folder + '/' + subfolder
        fileM = path + '/' + 'RFweights_all.txt'
        weights = np.loadtxt(fileM)
        RFweight = [np.sum(weights[0:N, 1]), np.sum(weights[N:2 * N, 1]),
                    np.sum(weights[2 * N:3 * N, 1]), np.sum(weights[3 * N:4 * N, 1])]
        RFWeights.append([tAnti, N, *RFweight])
    RFWeights = np.array(RFWeights)
    RFWeights = RFWeights[RFWeights[:, 1].argsort()]
    saveFileName = Folder.split('/')[-1] + 'Weights.txt'
    f = open(saveFileName, 'w')
    f.write("# tAnti[0]\tN[1]\tspace[2]\tspeed[3]\tdensity[4]\tplevel[5]\n")
    for RFWeight in RFWeights:
        f.write("{:.0f}\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(*RFWeight))
    f.close()
    labels = ['$W^d$', '$W^v$', '$W^\\rho$', '$W^p$']
    colors = ['indianred', 'steelblue', 'darkorange', 'seagreen']
    markers = ['o', '*', '^', 'v']
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(left=0.1, right=0.97,bottom=0.17,top=0.95,wspace=0.3)
    # plot the N tedency
    tAnti = 40
    data = RFWeights[RFWeights[:, 0] == tAnti]
    for i in range(4):
        ax[0].semilogx(data[:, 1], data[:, 2 + i], mec=colors[i], ls='--', marker=markers[i], ms=10,
                   mfc='none', mew='2', color=colors[i], lw=2,
                   label=labels[i])
        ax[0].legend(loc='best', ncol=2, fontsize=15, columnspacing=0.5)
        ax[0].set_xlabel('$n$', fontsize=20)
        ax[0].set_ylabel('Weights', fontsize=20)
        ax[0].set_ylim(0, 0.7)
        ax[0].set_xticks([2, 4, 8, 16, 32, 64],[2, 4, 8, 16, 32, 64])
        ax[0].tick_params(axis='both', which='major', labelsize=15)
        ax[0].tick_params(axis='both', which='minor', labelsize=15)
    # plot the tAnti tedency
    N = 2
    data = RFWeights[RFWeights[:, 1] == N]
    data = data[data[:, 0].argsort()]
    for i in range(4):
        ax[1].plot(data[:, 0], data[:, 2 + i], mec=colors[i], ls='--', marker=markers[i], ms=10,
                   mfc='none', mew='2', color=colors[i], lw=2,
                   label=labels[i])
        ax[1].legend(loc='best', ncol=2, fontsize=15, columnspacing=0.5)
        ax[1].set_xlabel('$t^{\\rm{anti}}$ [s]', fontsize=20)
        ax[1].set_ylabel('Weights', fontsize=20)
        ax[1].set_ylim(0, 0.7)
        ax[1].set_xticks([0,10,20,30,40,50],[0,0.4,0.8,1.2,1.6,2.0])
        ax[1].tick_params(axis='both', which='major', labelsize=15)
        ax[1].tick_params(axis='both', which='minor', labelsize=15)
    figname = 'RFGroupWeights1.png'
    plt.savefig(figname, dpi=300)
    # Plot all cases
    tAntis = np.unique(RFWeights[:, 0])
    fig, ax = plt.subplots(1, 11, figsize=(66, 6))
    for j, tAnti in enumerate(tAntis):
        data = RFWeights[RFWeights[:, 0] == tAnti]
        for i in range(4):
            ax[j].plot(data[:, 1], data[:, 2 + i], mec=colors[i], ls='--', marker=markers[i], ms=10,
                       mfc='none', mew='2', color=colors[i], lw=2,
                       label=labels[i])
            ax[j].legend(loc='best', ncol=2, fontsize=15, columnspacing=0.5)
            ax[j].set_xlabel('N', fontsize=20)
            ax[j].set_ylabel('Weights', fontsize=20)
            # ax[j].set_ylim(0, 0.7)
            ax[j].set_xticks([2, 4, 8, 16, 32, 64])
            ax[j].set_title("tAnti={:.0f}".format(tAnti))
    figname = 'tAnti_RFGroupWeights1.png'
    plt.savefig(figname, dpi=300)
    return


def ShowWeightsNorm(Folder):
    RFWeights = []
    for subfolder in os.listdir(Folder):
        tAnti = float(subfolder.split('_')[1])
        N = int(subfolder.split('_')[-1])
        path = Folder + '/' + subfolder
        fileM = path + '/' + 'RFweights_all.txt'
        weights = np.loadtxt(fileM)
        RFweight = [weights[0, 1], np.sum(weights[1:N + 1, 1]),
                    np.sum(weights[N + 1:2 * N + 1, 1]),
                    np.sum(weights[2 * N + 1:3 * N + 1, 1]),
                    np.sum(weights[3 * N + 1:4 * N + 1, 1])]
        RFWeights.append([tAnti, N, *RFweight])
    RFWeights = np.array(RFWeights)
    RFWeights = RFWeights[RFWeights[:, 1].argsort()]
    saveFileName = Folder.split('/')[-1] + 'Weights.txt'
    f = open(saveFileName, 'w')
    f.write(
        "# tAnti[0]\tN[1]\tMeanPlevel[2]\tspace[3]\tspeed[4]\tdensity[5]\tplevel[6]\n")
    for RFWeight in RFWeights:
        f.write(
            "{:.0f}\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(*RFWeight))
    f.close()
    # plot the tedency
    labels = ['$W(P_i^0)$','$W^d$', '$W^v$', '$W^\\rho$', '$W^p$']
    colors = ['black', 'indianred', 'steelblue', 'darkorange', 'seagreen']
    markers = ['+', 'o', '*', '^', 'v']
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(left=0.1, right=0.97,bottom=0.17,top=0.95,wspace=0.3)
    # plot the N tedency
    tAnti = 25
    data = RFWeights[RFWeights[:, 0] == tAnti]
    for i in range(5):
        if i==0:
            ax[0].semilogx(data[:, 1], data[:, 2 + i], mec=colors[i], ls='--', marker=markers[i], ms=10,
                   mfc='none', mew='2', color=colors[i], lw=2,
                   label=labels[i])
        else:
            ax[0].semilogx(data[:, 1], data[:, 2 + i], mec=colors[i], ls='--', marker=markers[i],
                           ms=10,
                           mfc='none', mew='2', color=colors[i], lw=2)
        ax[0].legend(loc='best', ncol=1, fontsize=15, columnspacing=0.5)
        ax[0].set_xlabel('$n$', fontsize=20)
        ax[0].set_ylabel('Weights', fontsize=20)
        ax[0].set_ylim(0, 0.7)
        ax[0].set_xticks([2, 4, 8, 16, 32, 64], [2, 4, 8, 16, 32, 64])
        ax[0].tick_params(axis='both', which='major', labelsize=15)
        ax[0].tick_params(axis='both', which='minor', labelsize=15)
    # plot the tAnti tedency
    N = 16
    data = RFWeights[RFWeights[:, 1] == N]
    data = data[data[:, 0].argsort()]
    for i in range(5):
        if i==0:
            ax[1].plot(data[:, 0], data[:, 2 + i], mec=colors[i], ls='--', marker=markers[i], ms=10,
                   mfc='none', mew='2', color=colors[i], lw=2,
                   label=labels[i])
        else:
            ax[1].plot(data[:, 0], data[:, 2 + i], mec=colors[i], ls='--', marker=markers[i], ms=10,
                       mfc='none', mew='2', color=colors[i], lw=2)
        ax[1].legend(loc='best', ncol=1, fontsize=15, columnspacing=0.5)
        ax[1].set_xlabel('$t^{\\rm{anti}}$ [s]', fontsize=20)
        ax[1].set_ylabel('Weights', fontsize=20)
        ax[1].set_ylim(0, 0.7)
        ax[1].set_xticks([0,10,20,30,40,50],[0,0.4,0.8,1.2,1.6,2.0])
        ax[1].tick_params(axis='both', which='major', labelsize=15)
        ax[1].tick_params(axis='both', which='minor', labelsize=15)
    figname = 'RFGroupWeights2.png'
    plt.savefig(figname, dpi=300)
    # Plot all cases
    tAntis = np.unique(RFWeights[:, 0])
    fig, ax = plt.subplots(1, 11, figsize=(66, 6))
    for j, tAnti in enumerate(tAntis):
        data = RFWeights[RFWeights[:, 0] == tAnti]
        for i in range(5):
            ax[j].plot(data[:, 1], data[:, 2 + i], mec=colors[i], ls='--', marker=markers[i], ms=10,
                       mfc='none', mew='2', color=colors[i], lw=2,
                       label=labels[i])
            ax[j].legend(loc='best', ncol=2, fontsize=15, columnspacing=0.5)
            ax[j].set_xlabel('N', fontsize=20)
            ax[j].set_ylabel('Weights', fontsize=20)
            # ax[j].set_ylim(0, 0.7)
            ax[j].set_xticks([2, 4, 8, 16, 32, 64])
            ax[j].set_title("tAnti={:.0f}".format(tAnti))
    figname = 'tAnti_RFGroupWeights2.png'
    plt.savefig(figname, dpi=300)
    return


def CollectWeightsPos(Folder):
    collects = 'Collection' + os.path.basename(Folder) + 'Weights'
    mkdir(collects)
    weightsInfos = []
    for subfolder in os.listdir(Folder):
        tAnti = float(subfolder.split('_')[1])
        N = float(subfolder.split('_')[-1])
        path = Folder + '/' + subfolder
        file = path + '/' + 'RFweights_all.png'
        newfile = '{:.0f}_{:.0f}_weights.png'.format(tAnti, N)
        shutil.copy(file, newfile)
        shutil.move(newfile, collects)
        # All weights Curve
        weightsFile = path + '/' + 'RFWeights_all.txt'
        weights = np.loadtxt(weightsFile)
        start = 0
        if weights[-1, 0] == 4 * N:
            start = 1
        weights = weights[start:, 1]
        times = int(64 / N)
        for i in range(4):
            info = weights[i * int(N):i * int(N) + int(N)]
            unsortedInfo = []
            for j in range(64):
                if j < 33:
                    pos = j
                else:
                    pos = j - 64
                index = int(j / times + 0.5)
                if index == N:
                    index = 0
                value = info[index] / times
                unsortedInfo.append([pos, value])
            unsortedInfo = np.array(unsortedInfo)
            infos = unsortedInfo[unsortedInfo[:, 0].argsort()]
            weightsInfos.append([tAnti, N, i, *infos[:, 1]])
    weightsInfos = np.array(weightsInfos)
    # write and plot
    FileName = Folder.split('/')[-1] + 'weightsCurve.txt'
    f = open(FileName, 'w')
    f.write('#i[0:space,1:speed,2:density,3:plevel]\n')
    f.write('#tAnti[0]\tN[1]\ti[2]\tFeatureWeights[3-66]\n')
    for weightInfo in weightsInfos:
        for i, w in enumerate(weightInfo):
            if i == 66:
                f.write('{:.3f}\n'.format(w))
            else:
                f.write('{:.3f}\t'.format(w))
    f.close()
    # Plot figures
    figName = Folder.split('/')[-1] + 'weightsCurve.png'
    titles = ['space', 'speed', 'density', 'plevel']
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    tAnti = 0
    dataP = weightsInfos[weightsInfos[:, 0] == tAnti]
    x = [num - 31 for num in range(64)]
    for i in range(4):
        dataPF = dataP[dataP[:, 2] == i]
        Ns = np.unique(dataPF[:, 1])
        for N in Ns:
            dataPFN = dataPF[dataPF[:, 1] == N][0, 3:]
            ax[i].plot(x, dataPFN, label=N)
        ax[i].set_title(titles[i])
        ax[i].set_xlabel('pos')
        ax[i].set_ylabel('weights')
        ax[i].legend(loc='best')
    plt.savefig(figName, dpi=300)


def AccuracyHeatMap(Result,figname):
    df=pd.DataFrame(Result,columns=['tAnti','N','train','nptest','ptest','macro','weighted'])
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    plt.subplots_adjust(left=0.06,right=0.96,bottom=0.15,top=0.9,wspace=0.45,hspace=0.2)
    features=['train','nptest','ptest','macro']
    titles=['$f_1^{~\\rm{train}}$','$f_1^{~\\rm{nptest}}$','$f_1^{~\\rm{ptest}}$','$f_1^{~\\rm{test}}$']
    for i in range(4):
        glue=df.pivot(index="tAnti",columns="N",values=features[i])
        sns.heatmap(glue,annot=True,linewidth=.1,cmap='crest',ax=axes[i],fmt=".2f").invert_yaxis()
        axes[i].set_xlabel('$n$',fontsize=20)
        axes[i].set_ylabel('$t^{\\rm{anti}}$ [s]',fontsize=20)
        axes[i].set_title(titles[i],fontsize=20)
        axes[i].tick_params(axis='both', which='major', labelsize=15)
        axes[i].tick_params(axis='both', which='minor', labelsize=15)
        axes[i].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5],[2,4,8,16,32,64])
        axes[i].set_yticks([tick+0.5 for tick in range(11)], [tick/5 for tick in range(11)])
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=15)
    plt.savefig('AccuracyHeatMap'+figname+'.png',dpi=300)





if __name__ == '__main__':
    # print("This paper will be published on Nature human behavior.")
    # 0. collect the data
    cwd = os.getcwd()
    RFFolder = cwd + '/RFApproach1'
    NormRFFolder = cwd + '/RFApproach2'
    RFResult = GetRFResult(RFFolder)
    NormRFResult = GetRFResult(NormRFFolder)
    # 0. collect all weights info
    CollectWeightsPos(RFFolder)
    CollectWeightsPos(NormRFFolder)
    # # 1. the influence of N
    # ShowNTedency(RFResult, 'RFGroup')
    # ShowNTedency(NormRFResult, 'RFGroupNorm')
    # # 2. the influence of tAnti
    # ShowtAntiTedency(RFResult, 'RFGroup')
    # ShowtAntiTedency(NormRFResult, 'RFGroupNorm')
    # # 3. compare RFGroup and RFGroupNorm
    # ShowNormEffect(RFResult, NormRFResult)
    # 4. show the weights
    ShowWeights(RFFolder)
    ShowWeightsNorm(NormRFFolder)
    # 5. Heatmap of tAnti and N
    AccuracyHeatMap(RFResult,'RFApproach1')
    AccuracyHeatMap(NormRFResult, 'RFApproach2')
