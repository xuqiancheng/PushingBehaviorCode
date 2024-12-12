# Analyze the different between the moving styles of pushing and nopushing
# Authot: Qiancheng
# Date: 2024.06.20
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import math
import copy
import seaborn as sns
from matplotlib.legend_handler import HandlerLine2D

def updateline(handle, orig):
    handle.update_from(orig)
    handle.set_markersize(10)

def myBinsEdge(data):
    return np.histogram_bin_edges(data, bins='scott')

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
            d[0] = index * 100 + d[0]
            if d[4] > 2.5:
                d[4] = 1
            else:
                d[4] = 0
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
    f.write('# 8: plevel (1:Pushing 0:noPushing)\n')
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
    # The pedestrian should be in the measurement area（data[:,7] is the y position）
    data = data[data[:, 7] > 0]
    data = data[data[:, 7] < 6]
    # check the density just in case
    data = data[data[:, 9] > 0]
    # the first several seconds should be ignored (data[:,5] is the frame)
    # data=data[data[:,5]>minFrame]
    return data


# the speed on the direction to the target
def generalize(data):
    newData = []
    target = (0, 0)
    for d in data:
        pos = (d[6], d[7])
        dis2Target = math.sqrt(pos[0] * pos[0] + pos[1] * pos[1])
        dir2Target = (-pos[0] / dis2Target, -pos[1] / dis2Target)
        moveDirection = (d[11], d[12])
        speed = d[10]
        geneSpeed = speed * (dir2Target[0] * moveDirection[0] + dir2Target[1] * moveDirection[1])
        d[10] = geneSpeed
        newData.append(d)
    newData = np.array(newData)
    return newData


# Obtain the mean data in the window
def IndividualMean(data):
    newData = []
    ids = np.unique(data[:, 4])
    for id in ids:
        dataid = data[data[:, 4] == id]
        index = dataid[0, 0]
        moti = dataid[0, 3]
        width = dataid[0, 2]
        frames = np.unique(dataid[:, 5])
        minFrame = np.min(frames)
        maxFrame = np.max(frames)
        frame1 = int(minFrame + (maxFrame - minFrame) / 3)
        frame2 = int(maxFrame - (maxFrame - minFrame) / 3)
        frame1 = minFrame
        frame2 = maxFrame
        dataFrame = dataid[dataid[:, 5] > frame1]
        dataFrame = dataFrame[dataFrame[:, 5] < frame2]
        if len(dataFrame) == 0:
            continue
        density = np.mean(dataFrame[:, 9])
        speed = np.mean(dataFrame[:, 10])
        newData.append([index, moti, width, id, density, speed])
    newData = np.array(newData)
    return newData


# plot the fd on the individual level (the mean value of agent)
def FDMean(data, figname):
    pushData = data[data[:, 8] == 1]
    noPushData = data[data[:, 8] == 0]
    meanPush = IndividualMean(pushData)
    meanNoPush = IndividualMean(noPushData)
    # plot figures
    widths = [1.2, 3.4, 4.5, 5.6]
    fig, ax = plt.subplots(2, 4, figsize=(24, 12))
    for i, width in enumerate(widths):
        widthPush = meanPush[meanPush[:, 2] == width]
        widthNoPush = meanNoPush[meanNoPush[:, 2] == width]
        for moti in [0, 1]:
            motiPush = widthPush[widthPush[:, 1] == moti]
            motiNoPush = widthNoPush[widthNoPush[:, 1] == moti]
            ax[moti][i].scatter(motiPush[:, 4], motiPush[:, 5], marker='o', color='steelblue',
                                label='push')
            ax[moti][i].scatter(motiNoPush[:, 4], motiNoPush[:, 5], marker='*', color='indianred',
                                label='noPush')
            ax[moti][i].set_xlim(0, 18)
            ax[moti][i].set_ylim(-0.2, 1)
            ax[moti][i].set_xlabel('density')
            ax[moti][i].set_ylabel('speed')
            ax[moti][i].set_title('moti={:.0f} width={:.1f}'.format(moti, width))
            ax[moti][i].legend(loc='best')
    plt.savefig(figname, dpi=300)


# Pick the stable frames in the experiemnt
def stableFrames(data):
    newData = []
    ids = np.unique(data[:, 4])
    for id in ids:
        dataid = data[data[:, 4] == id]
        minFrame = np.min(dataid[:, 5])
        maxFrame = np.max(dataid[:, 5])
        frame1 = int(minFrame + (maxFrame - minFrame) / 3)
        frame2 = int(maxFrame - (maxFrame - minFrame) / 3)
        dataid = dataid[dataid[:, 5] > frame1]
        dataid = dataid[dataid[:, 5] < frame2]
        for d in dataid:
            newData.append(d)
    newData = np.array(newData)
    return newData


def FDFrame(data, figname):
    dataStable = stableFrames(data)
    dataStable = data
    pushData = dataStable[dataStable[:, 8] == 1]
    noPushData = dataStable[dataStable[:, 8] == 0]
    # plot figures
    widths = [1.2, 3.4, 4.5, 5.6]
    fig, ax = plt.subplots(2, 4, figsize=(24, 12))
    for i, width in enumerate(widths):
        widthPush = pushData[pushData[:, 2] == width]
        widthNoPush = noPushData[noPushData[:, 2] == width]
        for moti in [0, 1]:
            motiPush = widthPush[widthPush[:, 3] == moti]
            motiNoPush = widthNoPush[widthNoPush[:, 3] == moti]
            ax[moti][i].scatter(motiNoPush[::40, 9], motiNoPush[::40, 10], marker='*',
                                color='indianred', fc='none',
                                label='noPush', alpha=0.7)
            ax[moti][i].scatter(motiPush[::20, 9], motiPush[::20, 10], marker='o', fc='none',
                                color='steelblue',
                                label='push', alpha=0.7)
            ax[moti][i].set_xlim(0, 20)
            ax[moti][i].set_ylim(-0.2, 1.5)
            ax[moti][i].set_xlabel('density')
            ax[moti][i].set_ylabel('speed')
            ax[moti][i].set_title('moti={:.0f} width={:.1f}'.format(moti, width))
            ax[moti][i].legend(loc='best')
    plt.savefig(figname, dpi=300)


def tedency(data):
    relations = []
    # densityMin=np.min(data[:,9])
    # densityMax=np.max(data[:,9])
    # N=int(np.ceil((densityMax-densityMin)/1))
    densityMin = 0
    densityMax = 20
    for i in range(20):
        dataSelected = data[data[:, 9] >= densityMin + i * 1]
        dataSelected = dataSelected[dataSelected[:, 9] < densityMin + i * 1 + 1]
        if len(dataSelected) < 100:
            continue
        densityMean = np.mean(dataSelected[:, 9])
        speedMean = np.mean(dataSelected[:, 10])
        speedStd = np.std(dataSelected[:, 10])
        relations.append([densityMean, speedMean, speedStd])
    relations = np.array(relations)
    return relations


def FDTendency(data, figname):
    dataStable = stableFrames(data)
    dataStable = data
    pushData = dataStable[dataStable[:, 8] == 1]
    noPushData = dataStable[dataStable[:, 8] == 0]
    # plot figures
    motivations=['high','low']
    widths = [1.2, 3.4, 4.5, 5.6]
    fig, ax = plt.subplots(2, 4, figsize=(16,8))
    plt.subplots_adjust(left=0.07,right=0.96,bottom=0.15,top=0.85,wspace=0.4,hspace=0.6)
    for i, width in enumerate(widths):
        widthPush = pushData[pushData[:, 2] == width]
        widthNoPush = noPushData[noPushData[:, 2] == width]
        for moti in [0, 1]:
            motiPush = widthPush[widthPush[:, 3] == moti]
            motiNoPush = widthNoPush[widthNoPush[:, 3] == moti]
            PushTedency = tedency(motiPush)
            NoPushTedency = tedency(motiNoPush)
            ax[moti][i].scatter(motiNoPush[::20, 9], motiNoPush[::20, 10], marker='*',
                                color='steelblue', fc='none', alpha=0.05)
            ax[moti][i].scatter(motiPush[::10, 9], motiPush[::10, 10], marker='o', fc='none',
                                color='indianred', alpha=0.05)
            ax[moti][i].plot(PushTedency[:, 0], PushTedency[:, 1], color='indianred', marker='o',
                             label='push', mec='k', ms=5)
            ax[moti][i].errorbar(PushTedency[:, 0], PushTedency[:, 1], PushTedency[:, 2], capsize=4,
                                 color='indianred')
            ax[moti][i].plot(NoPushTedency[:, 0], NoPushTedency[:, 1], color='steelblue',
                             marker='^', label='non push', mec='k', ms=5)
            ax[moti][i].errorbar(NoPushTedency[:, 0], NoPushTedency[:, 1], NoPushTedency[:, 2],
                                 capsize=4,
                                 color='steelblue')
            ax[moti][i].set_xlim(0, 20)
            ax[moti][i].set_ylim(-0.2, 1.2)
            ax[moti][i].set_xlabel('Density [$\\rm{m}^{-2}$]',fontsize=20)
            ax[moti][i].set_ylabel('Speed [$\\rm{m}\\cdot \\rm{s}^{-1}$]',fontsize=20)
            #ax[moti][i].set_title('moti={:.0f} width={:.1f}'.format(moti, width))
            #ax[moti][i].legend(loc='best')
            ax[moti][i].tick_params(axis='both', which='major', labelsize=15)
            ax[moti][i].tick_params(axis='both', which='minor', labelsize=15)
            ax[moti][i].text(10, 0.9,
                               '{}  {} m'.format(motivations[moti], width),
                               bbox=dict(facecolor='grey', alpha=0.1), fontsize=15)
    ax[0][3].legend(bbox_to_anchor=(0.3, 1.3), loc=10, fontsize=20, ncol=2)
    plt.savefig(figname, dpi=300)


# update the space based on the distance to walls
def geometry(data):
    newData = []
    for d in data:
        if d[9] > 0:
            width = d[2]
            space = dis2Walls(width, d)
            if space < d[13]:
                d[13] = space
            newData.append(d)
    newData = np.array(newData)
    return newData


# calculate the distance to the wall
def dis2Walls(width, d):
    walls = [[(-width / 2, 0), (-0.25, 0)],
             [(width / 2, 0), (0.25, 0)],
             [(-width / 2, 0), (-width / 2, 7)],
             [(width / 2, 0), (width / 2, 7)],
             [(-0.25, 0), (-0.25, -1)],
             [(0.25, 0), (0.25, -1)],
             ]
    pos = (d[6], d[7])
    dir = (d[11], d[12])
    space = 100
    for wall in walls:
        wp1 = wall[0]
        wp2 = wall[1]
        vec1 = (wp1[0] - pos[0], wp1[1] - pos[1])
        vec2 = (wp2[0] - pos[0], wp2[1] - pos[1])
        crossP1 = crossProduct(dir, vec1)
        crossP2 = crossProduct(dir, vec2)
        if crossP1 * crossP2 > 0:
            continue
        else:
            wVec = (wp2[0] - wp1[0], wp2[1] - wp1[1])
            pVec = (pos[0] - wp1[0], pos[1] - wp1[1])
            wallLength = math.sqrt(scalarProduct(wVec, wVec))
            length = scalarProduct(pVec, wVec) / wallLength
            ExtraVec = (wVec[0] * length / wallLength, wVec[1] * length / wallLength)
            pt = (wp1[0] + ExtraVec[0], wp1[1] + ExtraVec[1])
            vec2PT = (pt[0] - pos[0], pt[1] - pos[1])
            dis2Pt = math.sqrt(scalarProduct(vec2PT, vec2PT))
            if scalarProduct(vec2PT, dir) == 0:
                space2Wall = 100
            else:
                space2Wall = dis2Pt * dis2Pt / scalarProduct(vec2PT, dir)
            if space2Wall > 0:
                space2Wall = space2Wall - 0.18
                if space2Wall < space:
                    space = space2Wall
    return space


def scalarProduct(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def crossProduct(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def forward(data):
    newData = []
    # target=(0,0)
    for d in data:
        pos = (d[6], d[7])
        dis2Target = math.sqrt(pos[0] * pos[0] + pos[1] * pos[1])
        dir2Target = (-pos[0] / dis2Target, -pos[1] / dis2Target)
        moveDirection = (d[11], d[12])
        forward = (dir2Target[0] * moveDirection[0] + dir2Target[1] * moveDirection[1])
        if forward > 0:
            newData.append(d)
        else:
            d[10] = -1 * d[10]
            # newData.append(d)
    newData = np.array(newData)
    return newData


def TSpaceFrame(data, figname):
    dataStable = stableFrames(data)
    dataStable = data
    pushData = dataStable[dataStable[:, 8] == 1]
    noPushData = dataStable[dataStable[:, 8] == 0]
    # plot figures
    widths = [1.2, 3.4, 4.5, 5.6]
    fig, ax = plt.subplots(2, 4, figsize=(24, 12))
    for i, width in enumerate(widths):
        widthPush = pushData[pushData[:, 2] == width]
        widthNoPush = noPushData[noPushData[:, 2] == width]
        for moti in [0, 1]:
            motiPush = widthPush[widthPush[:, 3] == moti]
            motiNoPush = widthNoPush[widthNoPush[:, 3] == moti]
            ax[moti][i].scatter(motiNoPush[::20, 13], motiNoPush[::20, 10], marker='*',
                                color='indianred', fc='none',
                                label='noPush', alpha=0.7)
            ax[moti][i].scatter(motiPush[::10, 13], motiPush[::10, 10], marker='o', fc='none',
                                color='steelblue',
                                label='push', alpha=0.7)
            ax[moti][i].set_xlim(-0.5, 2)
            ax[moti][i].set_ylim(-0.2, 1.2)
            ax[moti][i].set_xlabel('space')
            ax[moti][i].set_ylabel('speed')
            ax[moti][i].set_title('moti={:.0f} width={:.1f}'.format(moti, width))
            ax[moti][i].legend(loc='best')
    plt.savefig(figname, dpi=300)


def reaction(data):
    minFrame = 25
    newData = []
    ids = np.unique(data[:, 4])
    for id in ids:
        dataid = data[data[:, 4] == id]
        dataid = dataid[dataid[:, 5] > minFrame]
        for d in dataid:
            newData.append(d)
    newData = np.array(newData)
    return newData


def TSpaceTendency(data, figname):
    dataStable = stableFrames(data)
    dataStable = data
    pushData = dataStable[dataStable[:, 8] == 1]
    noPushData = dataStable[dataStable[:, 8] == 0]
    # plot figures
    widths = [1.2, 3.4, 4.5, 5.6]
    ymax=[1.5,0.8]
    ytext=[1.2,0.64]
    motivations = ['high', 'low']
    fig, ax = plt.subplots(2, 4, figsize=(16, 6))
    plt.subplots_adjust(left=0.07, right=0.96, bottom=0.12, top=0.9, wspace=0.4, hspace=0.35)
    for i, width in enumerate(widths):
        widthPush = pushData[pushData[:, 2] == width]
        widthNoPush = noPushData[noPushData[:, 2] == width]
        for moti in [0, 1]:
            motiPush = widthPush[widthPush[:, 3] == moti]
            motiNoPush = widthNoPush[widthNoPush[:, 3] == moti]
            PushTedency = Ttedency(motiPush)
            NoPushTedency = Ttedency(motiNoPush)
            ax[moti][i].scatter(motiNoPush[::20, 13], motiNoPush[::20, 10], marker='*',
                                color='steelblue', fc='none', alpha=0.1)
            ax[moti][i].scatter(motiPush[::10, 13], motiPush[::10, 10], marker='o', fc='none',
                                color='indianred', alpha=0.1)
            ax[moti][i].errorbar(PushTedency[:, 0], PushTedency[:, 1], PushTedency[:, 2], capsize=4,
                                 color='indianred')
            ax[moti][i].errorbar(NoPushTedency[:, 0], NoPushTedency[:, 1], NoPushTedency[:, 2],
                                 capsize=4,
                                 color='steelblue')
            ax[moti][i].plot(PushTedency[:, 0], PushTedency[:, 1], color='k', marker='o',mfc='indianred',
                             lw=2,
                             label='pushing', mec='k', ms=5)
            ax[moti][i].plot(NoPushTedency[:, 0], NoPushTedency[:, 1], color='k', lw=2,mfc='steelblue',
                             marker='^', label='non-pushing', mec='k', ms=5)
            ax[moti][i].set_xlim(-0.5, 2)
            ax[moti][i].set_ylim(-0.05, ymax[moti])
            ax[moti][i].set_xlabel('$s$ [m]',fontsize=20)
            ax[moti][i].set_ylabel('$v$ [$\\rm{m}\\cdot \\rm{s}^{-1}$]', fontsize=20)
            ax[moti][i].tick_params(axis='both', which='major', labelsize=15)
            ax[moti][i].tick_params(axis='both', which='minor', labelsize=15)
            ax[moti][i].text(0.8, ytext[moti],
                             '{}  {} m'.format(motivations[moti], width),
                             bbox=dict(facecolor='grey', alpha=0.1), fontsize=15)
    ax[0][3].legend(bbox_to_anchor=(0.05, 1.17), loc=10, fontsize=20, ncol=2,handler_map={plt.Line2D: HandlerLine2D(update_func=updateline)})
    plt.savefig(figname, dpi=600)


def Ttedency(data):
    relations = []
    SpaceMin = -0.5
    SpaceMax = 2
    for i in range(25):
        dataSelected = data[data[:, 13] >= SpaceMin + i * 0.1]
        dataSelected = dataSelected[dataSelected[:, 13] < SpaceMin + i * 0.1 + 0.1]
        if len(dataSelected) < 100:
            continue
        SpaceMean = np.mean(dataSelected[:, 13])
        speedMean = np.mean(dataSelected[:, 10])
        speedStd = np.std(dataSelected[:, 10])
        relations.append([SpaceMean, speedMean, speedStd])
    relations = np.array(relations)
    return relations


# calculate mean values (density,speed,dangle1,dangle2) and plot the figure
def MeanValues(data):
    ids = np.unique(data[:, 4])
    results = []
    for id in ids:
        dataID = data[data[:, 4] == id]
        frames = np.unique(dataID[:, 5])
        for frame in frames:
            if frame == frames[-1]:
                continue
            dataFrame = dataID[dataID[:, 5] == frame]
            nextFrame = dataID[dataID[:, 5] == frame + 1]
            if len(nextFrame) == 0:
                continue
            # todo: check this function
            Angles = deviationAngles(dataFrame[0], nextFrame[0])
            if Angles[0]==-100:
                continue
            width = dataFrame[0][2]
            moti = dataFrame[0][3]
            plevel = dataFrame[0][8]
            density = dataFrame[0][9]
            speed = dataFrame[0][10]
            if speed==0:
                continue
            results.append([width, moti, plevel, density, speed, Angles[0]*math.pi/180, Angles[1]*math.pi/180])
    results = np.array(results)
    # plot the distributions
    features = ['Density', 'Speed', 'Turning angle', 'DeviationAngle']
    xlabels=['$\\rho$ [$\\rm{m}^{-2}$]','$v$ [$\\rm{m}\\cdot \\rm{s}^{-1}$]','$\\theta$ [rad]','$\\theta$ [rad]']
    pushData = results[results[:, 2] == 1]
    noPushData = results[results[:, 2] == 0]
    motivations = ['high', 'low']
    xlims=[[0,15],[0,1.5],[0,5],[0,2]]
    ylims=[[0,0.65],[0, 5],[0, 0.03],[0, 2]]
    textpos=[[7.5,0.52],[0.75,4],[50,0.024],[1,1.6]]
    for fea in [0,1,3]:
        # plot figures
        widths = [1.2, 3.4, 4.5, 5.6]
        fig, axes = plt.subplots(2, 4, figsize=(16, 5))
        plt.subplots_adjust(left=0.07,right=0.96,bottom=0.15,top=0.85,wspace=0.4,hspace=0.6)
        for i, width in enumerate(widths):
            widthPush = pushData[pushData[:, 0] == width]
            widthNoPush = noPushData[noPushData[:, 0] == width]
            for moti in [0, 1]:
                motiPush = widthPush[widthPush[:, 1] == moti]
                motiNoPush = widthNoPush[widthNoPush[:, 1] == moti]
                binsPush = myBinsEdge(motiPush[:, 3+fea])
                binsNoPush=myBinsEdge(motiNoPush[:, 3+fea])
                # sns.histplot(motiPush[:, 3+fea], bins=binsPush, ax=axes[moti, i], stat='density',
                #              facecolor='steelblue',alpha=0.5)
                sns.kdeplot(motiPush[:, 3+fea], ax=axes[moti, i], color='indianred',
                            bw_method='scott', bw_adjust=1, linewidth=3,
                            label='pushing',fill=0.5)

                # sns.histplot(motiNoPush[:, 3 + fea], bins=binsNoPush, ax=axes[moti, i], stat='density',
                #              facecolor='indianred', alpha=0.5)
                sns.kdeplot(motiNoPush[:, 3 + fea], ax=axes[moti, i], color='steelblue',
                            bw_method='scott', bw_adjust=1, linewidth=3,
                            label='non-pushing',fill=0.5,linestyle='--')
                axes[moti][i].set_xlim(xlims[fea][0],xlims[fea][1])
                axes[moti][i].set_ylim(ylims[fea][0],ylims[fea][1])
                axes[moti][i].set_xlabel(xlabels[fea],fontsize=20)
                axes[moti][i].set_ylabel('Density',fontsize=20)
                axes[moti][i].tick_params(axis='both', which='major', labelsize=15)
                axes[moti][i].tick_params(axis='both', which='minor', labelsize=15)
                axes[moti][i].text(textpos[fea][0],textpos[fea][1 ],'{}  {} m'.format(motivations[moti],width),bbox=dict(facecolor='grey', alpha=0.1), fontsize=15)
                #axes[moti][i].set_title('moti={:.0f} width={:.1f}'.format(moti, width))
                #axes[moti][i].legend(loc='best',fontsize=20)
        axes[0][3].legend(bbox_to_anchor=(-0.3, 1.3), loc=10, fontsize=20, ncol=2,handlelength=5)
        plt.savefig(features[fea]+'Distributions.png', dpi=300)

# calculate the turning angle and the deviationa angle with desired direction
def deviationAngles(dataCurrent, dataNext):
    des_direction = [-dataCurrent[6], -dataCurrent[7]]
    cur_direction = [dataCurrent[11], dataCurrent[12]]
    next_direction = [dataNext[11], dataNext[12]]
    des_Angle = math.atan2(-dataCurrent[7], -dataCurrent[6]) * (180 / math.pi)
    cur_Angle = math.atan2(dataCurrent[12], dataCurrent[11]) * (180 / math.pi)
    next_Angle = math.atan2(dataNext[12], dataNext[11]) * (180 / math.pi)
    # backward movements are not considered
    if scalarProduct(cur_direction,des_direction)<0 or scalarProduct(next_direction,des_direction)<0:
        return [-100,-100]
    angleTurn = np.abs(next_Angle - cur_Angle)
    if angleTurn > 180:
        angleTurn = 360 - angleTurn
    turn_direction = crossProduct(next_direction, cur_direction)
    if turn_direction < 0:
        angleTurn = -angleTurn
    angleDev = np.abs(cur_Angle - des_Angle)
    if angleDev > 180:
        angleDev = 360 - angleDev
    turn_direction = crossProduct(cur_direction, des_direction)
    if turn_direction < 0:
        angleDev = -angleDev
    # print("Info: Check this function.\n")
    # print("dataCurrent:",dataCurrent,'\n')
    # print("dataNext:",dataNext,'\n')
    # print("result:",[np.abs(angleTurn), np.abs(angleDev)],'\n')
    return [np.abs(angleTurn), np.abs(angleDev)]

# Generalized the speed based on the mean speed
def SpeedMean(data):
    dataMean=[]
    MeanSpeed={}
    ids=np.unique(data[:,4])
    for id in ids:
        dataID=data[data[:,4]==id]
        mean=np.max(dataID[:,10])
        MeanSpeed[id]=mean
    for d in data:
        id=d[4]
        mean=MeanSpeed[id]
        d[10]=d[10]/mean
        dataMean.append(d)
    dataMean=np.array(dataMean)
    return dataMean

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
    # FD individual level and frame level
    dataFD = copy.deepcopy(Alldata)
    FDTendency(dataFD, figname='FDTendency.png')
    # T-Space differencence
    dataTSpace = copy.deepcopy(Alldata)
    # dataTSpace=reaction(dataTSpace)
    dataTSpace = geometry(dataTSpace)  # calculate the space to the wall, which is also important
    dataTSpace = forward(
        dataTSpace)  # Giving speed the direction, backward movement corresponds to minues speed
    TSpaceFrame(dataTSpace, figname='HeadwayScatter.png')
    TSpaceTendency(dataTSpace, figname='HeadwayTendency.png')
    # Mean values and distribution
    dataMean = copy.deepcopy(Alldata)
    MeanValues(dataFD)
