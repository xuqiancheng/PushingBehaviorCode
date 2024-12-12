# Author Qiancheng
# Date 2024.09.19
# Purpose analyze the trajectories
import glob
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
import seaborn as sns


def pushProportion(data):
    pushData = data[data[:, 12] == 1]
    return len(pushData) / len(data)


def meanEvacuationGap(data):
    framerate = 25
    ids = np.unique(data[:, 0])
    PassTimes = []
    for id in ids:
        dataid = data[data[:, 0] == id]
        Time = np.max(dataid[:, 3])
        dataPass = dataid[dataid[:, 3] < 0]
        if len(dataPass) != 0:
            Time = np.min(dataPass[:, 1])
        PassTimes.append(Time)
    totalTime = (np.max(PassTimes) - np.min(PassTimes)) / framerate
    meanTime = totalTime / (len(ids)-1)
    return meanTime


def r2HeatMap(Result, para1, para2):
    df = pd.DataFrame(Result, columns=[para1, para2, 'r2PushProp', 'r2MeanTime'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(left=0.05, right=0.95)
    features = ['r2PushProp', 'r2MeanTime']
    for i in range(2):
        glue = df.pivot(index=para1, columns=para2, values=features[i])
        sns.heatmap(glue, annot=True, linewidth=.5, cmap='crest', ax=axes[i], fmt=".3f")
        # show the max values on the figure
        maxindex = np.argmax(Result[:, 2 + i])
        maxValue = np.max(Result[:, 2 + i])
        maxpara1 = Result[maxindex, 0]
        maxpara2 = Result[maxindex, 1]
        axes[i].set_title("maxr2={:.3f} para1={} para2={}".format( maxValue, maxpara1, maxpara2))
    plt.savefig('r2HeatMap' + para1 + para2 + '.png', dpi=300)


def MAPEHeatMap(Result, para1, para2):
    df = pd.DataFrame(Result, columns=[para1, para2, 'MAPEPushProp', 'MAPEMeanTime','MAPEAll'])
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(left=0.05, right=0.95)
    features = ['MAPEPushProp', 'MAPEMeanTime','MAPEAll']
    for i in range(3):
        glue = df.pivot(index=para1, columns=para2, values=features[i])
        sns.heatmap(glue, fmt=".3f",annot=True, linewidth=.5, cmap='crest', ax=axes[i])
        # show the min values on the figure
        minindex=np.argmin(Result[:,2+i])
        minValue=np.min(Result[:,2+i])
        minpara1=Result[minindex,0]
        minpara2=Result[minindex,1]
        axes[i].set_title("minMAPE={:.3f} para1={} para2={}".format(minValue,minpara1,minpara2))
    plt.savefig('MAPEHeatMap' + para1 + para2 + '.png', dpi=300)

def MAPE(expData,simData):
    res=[]
    for i,expd in enumerate(expData):
        simd=simData[i]
        temp=abs((simd-expd)/expd)
        res.append(temp)
    return res


if __name__ == '__main__':
    if os.path.exists('logAnalyze.log'):
        os.remove('logAnalyze.log')
    logging.basicConfig(filename='logAnalyze.log', level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    mainDir = os.getcwd()
    folders = os.listdir()
    paraNames = ['force_ped-a', 'force_ped-D', 'PVM-apush', 'PVM-Dpush', 'GCVM-Ts', 'PVM-Tpush',
                 'PVM-Spush', 'PVM-Snorm','PVM-aForce', 'PVM-DForce','force_wall-a','force_wall-D','GCVM-Td']
    # 0 Set studied parameters ###################################################################
    index1 = 4
    index2 = 5
    para1 = paraNames[index1]
    para2 = paraNames[index2]
    # 1 Calculate evacuation efficiency and push proportion for each folder
    EvacTandPushP = []
    for folder in folders:
        # Only check the influence of para1 and para2
        if folder.find(para1) == -1 or folder.find(para2) == -1:
            continue
        if folder.find('.txt') != -1 or folder.find('.png') != -1:
            continue
        os.chdir(folder)
        para1Value = float(folder.split(para1)[1].split('_')[1].split('_')[0])
        para2Value = float(folder.split(para2)[1].split('_')[1].split('.txt')[0])
        logging.info("Calculate evacuation efficiency and push proportion for {}.".format(folder))
        print("Calculate evacuation efficiency and push proportion for {}.".format(folder))
        files = glob.glob('*.txt')
        for file in files:
            seed = float(file.split('seed_')[1].split('_')[0])
            if seed>1000:
                continue
            moti = file.split('_')[3]
            if moti == 'high':
                moti = 0
            if moti == 'low':
                moti = 1
            width = float(file.split('_')[4].split('.txt')[0])
            trajData = np.loadtxt(file)
            # Calculate the push proportion and the mean evacuation gap
            pushProp = pushProportion(trajData)
            meanEGap = meanEvacuationGap(trajData)
            EvacTandPushP.append([para1Value, para2Value, seed, moti, width, pushProp, meanEGap])
        os.chdir(mainDir)
    # 2 write the trajectory information down
    EvacTandPushP = np.array(EvacTandPushP)
    f = open('EvacTPushP_{}_{}.txt'.format(para1, para2), 'w')
    f.write('# para1: {}\n'.format(para1))
    f.write('# para2: {}\n'.format(para2))
    f.write('# moti: high(0) low(1)\n')
    f.write('# para1\tpara2\tseed\tmoti\twidth[m]\tpushProp\tmeanEGap[s]\n')
    for ETPP in EvacTandPushP:
        f.write('{:.2f}\t{:.2f}\t{:.0f}\t{:.0f}\t{:.1f}\t{:.3f}\t{:.3f}\n'.format(*ETPP))
    f.close()
    # 3 calculate the mean values
    MeanValues = []
    para1values = np.unique(EvacTandPushP[:, 0])
    para2values = np.unique(EvacTandPushP[:, 1])
    motis = np.unique(EvacTandPushP[:, 3])
    widths = np.unique(EvacTandPushP[:, 4])
    for para1v in para1values:
        data1 = EvacTandPushP[EvacTandPushP[:, 0] == para1v]
        for para2v in para2values:
            data2 = data1[data1[:, 1] == para2v]
            for moti in motis:
                data3 = data2[data2[:, 3] == moti]
                for width in widths:
                    data4 = data3[data3[:, 4] == width]
                    meanPushP = np.mean(data4[:, 5])
                    stdPushP = np.std(data4[:, 5])
                    meanEvacT = np.mean(data4[:, 6])
                    stdEvacT = np.std(data4[:, 6])
                    MeanValues.append(
                        [para1v, para2v, moti, width, meanPushP, stdPushP, meanEvacT, stdEvacT])
    # 4 write the mean values down
    MeanValues = np.array(MeanValues)
    f = open('MeanValues_{}_{}.txt'.format(para1, para2), 'w')
    f.write('# para1: {}\n'.format(para1))
    f.write('# para2: {}\n'.format(para2))
    f.write('# moti: high(0) low(1)\n')
    f.write(
        '# para1\tpara2\tmoti\twidth[m]\tpushProp(mean)\tpushProp(std)\tmeanEGap(mean)[s]\tmeanEGap(std)[s]\n')
    for MeanValue in MeanValues:
        f.write(
            '{:.2f}\t{:.2f}\t{:.0f}\t{:.1f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(*MeanValue))
    f.close()
    # 5 plot figures
    ExpData = np.loadtxt('ExperimentalData.txt')
    para1Values = np.unique(MeanValues[:, 0])
    para2Values = np.unique(MeanValues[:, 1])
    motis = np.unique(MeanValues[:, 2])
    colors = ['b', 'g', 'r', 'c', 'm', 'y','peru']
    for para1Value in para1Values:
        logging.info("Plot figures for {}={}.".format(para1, para1Value))
        print("Plot figures for {}={}.".format(para1, para1Value))
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        dataPara1 = MeanValues[MeanValues[:, 0] == para1Value]
        for moti in motis:
            ExpMoti = ExpData[ExpData[:, 3] == moti]
            datamoti = dataPara1[dataPara1[:, 2] == moti]
            i = int(moti)
            axes[i][0].plot(ExpMoti[:, 2], ExpMoti[:, 6], color='k',
                            label='Exp_{}_pushP'.format(moti))
            axes[i][1].plot(ExpMoti[:, 2], ExpMoti[:, 7], color='k',
                            label='Exp_{}_meanT'.format(moti))
            for ci, para2Value in enumerate(para2Values):
                dataPara2 = datamoti[datamoti[:, 1] == para2Value]
                axes[i][0].plot(dataPara2[:, 3], dataPara2[:, 4], color=colors[ci],
                                label='p1_{}_p2_{}'.format(para1Value, para2Value))
                axes[i][0].errorbar(dataPara2[:, 3], dataPara2[:, 4], dataPara2[:, 5],
                                    color=colors[ci])
                axes[i][1].plot(dataPara2[:, 3], dataPara2[:, 6], color=colors[ci],
                                label='p1_{}_p2_{}'.format(para1Value, para2Value))
                axes[i][1].errorbar(dataPara2[:, 3], dataPara2[:, 6], dataPara2[:, 7],
                                    color=colors[ci])
            axes[i][0].set_xlabel('width [m]')
            axes[i][1].set_xlabel('width [m]')
            axes[i][0].set_ylabel('push proportion')
            axes[i][1].set_ylabel('mean evacuation time [s]')
            axes[i][1].set_ylim(0.5,2)
            axes[i][0].legend(loc='best')
            axes[i][1].legend(loc='best')
        plt.savefig('{}_{}.png'.format(para1, para1Value), dpi=300)
        plt.close()
        plt.cla()
        plt.clf()
    # 6 the value of r2
    r2Values = []
    ExpData = np.loadtxt('ExperimentalData.txt')
    # Sort in the same order: first motivation then width
    ExpData = ExpData[ExpData[:, 3].argsort()]
    ExpData = ExpData[ExpData[:, 2].argsort()]
    para1Values = np.unique(MeanValues[:, 0])
    para2Values = np.unique(MeanValues[:, 1])
    motis = np.unique(MeanValues[:, 2])
    for para1Value in para1Values:
        logging.info("Calculate r2 for {}={}.".format(para1, para1Value))
        print("Calculate r2 for {}={}.".format(para1, para1Value))
        dataPara1 = MeanValues[MeanValues[:, 0] == para1Value]
        for para2Value in para2Values:
            dataPara2 = dataPara1[dataPara1[:, 1] == para2Value]
            #  Sort in the same order: first motivation then width
            dataPara2 = dataPara2[dataPara2[:, 2].argsort()]
            dataPara2 = dataPara2[dataPara2[:, 3].argsort()]
            r2PushProp = r2_score(ExpData[:, 6], dataPara2[:, 4])
            r2MeanTime = r2_score(ExpData[:, 7], dataPara2[:, 6])
            r2Values.append([para1Value, para2Value, r2PushProp, r2MeanTime])
    # 7 write r2 result down
    r2Values = np.array(r2Values)
    f = open('r2Values_{}_{}.txt'.format(para1, para2), 'w')
    f.write('# para1: {}\n'.format(para1))
    f.write('# para2: {}\n'.format(para2))
    f.write(
        '# para1\tpara2\tr2PushProp\tr2MeanTime\n')
    for r2Value in r2Values:
        f.write(
            '{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}\n'.format(*r2Value))
    f.close()
    # 8 Plot heatmap
    logging.info("Plot r2 heatmap for {} {}.".format(para1, para2))
    print("Plot r2 heatmap for {} {}.".format(para1, para2))
    r2HeatMap(r2Values, para1, para2)

    # 9 the square of the relative error
    MAPEValues = []
    ExpData = np.loadtxt('ExperimentalData.txt')
    # Sort in the same order: first motivation then width
    ExpData = ExpData[ExpData[:, 3].argsort()]
    ExpData = ExpData[ExpData[:, 2].argsort()]
    para1Values = np.unique(MeanValues[:, 0])
    para2Values = np.unique(MeanValues[:, 1])
    motis = np.unique(MeanValues[:, 2])
    for para1Value in para1Values:
        logging.info("Calculate the Mean Absolute Percentage Error for {}={}.".format(para1, para1Value))
        print("Calculate the Mean Absolute Percentage Error for {}={}.".format(para1, para1Value))
        dataPara1 = MeanValues[MeanValues[:, 0] == para1Value]
        for para2Value in para2Values:
            dataPara2 = dataPara1[dataPara1[:, 1] == para2Value]
            #  Sort in the same order: first motivation then width
            dataPara2 = dataPara2[dataPara2[:, 2].argsort()]
            dataPara2 = dataPara2[dataPara2[:, 3].argsort()]
            MAPEPushProp = MAPE(ExpData[:, 6], dataPara2[:, 4])
            MAPEMeanTime = MAPE(ExpData[:, 7], dataPara2[:, 6])
            MAPEValues.append([para1Value, para2Value, np.mean(MAPEPushProp), np.mean(MAPEMeanTime),np.mean(MAPEPushProp+MAPEMeanTime)])
    # 10 write res result down
    MAPEValues = np.array(MAPEValues)
    f = open('MAPEValues_{}_{}.txt'.format(para1, para2), 'w')
    f.write('# para1: {}\n'.format(para1))
    f.write('# para2: {}\n'.format(para2))
    f.write(
        '# para1\tpara2\tMAPEPushProp\tMAPEMeanTime\tMAPEAll\n')
    for MAPEValue in MAPEValues:
        f.write(
            '{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(*MAPEValue))
    f.close()
    # 11 Plot heatmap
    logging.info("Plot MAPE heatmap for {} {}.".format(para1, para2))
    print("Plot MAPE heatmap for {} {}.".format(para1, para2))
    MAPEHeatMap(MAPEValues, para1, para2)