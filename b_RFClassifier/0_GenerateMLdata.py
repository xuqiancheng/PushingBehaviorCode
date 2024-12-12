# author: Qiancheng Xu
# date: 2024.01.18
# info: generate data for machine learning

import numpy as np
import glob
import os
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
import shutil

# somebasic definition
target = (0, 0)
minFrame = 25
bodySize = 0.18


############################################################################
def distance(pos1, pos2):
    temp = (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1])
    return math.sqrt(temp)


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


# the norm of the direction
def Norm(direction):
    temp = direction[0] * direction[0] + direction[1] * direction[1]
    dirNorm = math.sqrt(temp)
    # check the norm
    # print("\ndirection:", direction)
    # print("dirNorm:", dirNorm)
    return dirNorm


# return the normalization of vector
def UnitVector(direction):
    return [i / Norm(direction) for i in direction]


def scalarProduct(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def crossProduct(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


# anticlockwise
def Rotation(direction, angle):
    x = direction[0]
    y = direction[1]
    x1 = x * math.cos(angle) - y * math.sin(angle)
    y1 = y * math.cos(angle) + x * math.sin(angle)
    # Check the result
    # print('\ndirection:', direction)
    # print('angle:', angle)
    # print('newdirection:', [x1,y1])
    return [x1, y1]


# anticlockwise rotation is positive
def AngleBetween(d1, d2):
    if d1[0] == 0 and d1[1] == 0:
        return -100
    ang = np.arctan2(d1[1], d1[0]) - np.arctan2(d2[1], d2[0])
    if ang > np.pi:
        ang = ang - 2 * np.pi
    elif ang < -np.pi:
        ang = 2 * np.pi + ang
    return ang


def GenerateMLData_InfoDistribution(data, tAnti, width, N):
    print("INFO: Format the data for Machine Learning based on the surrounding infos")
    MLData = []
    ids = np.unique(data[:, 4])
    for id in ids:
        dataID = data[data[:, 4] == id]
        frames = np.unique(dataID[:, 5])
        # here may be normalization is important, every person's vmax and vmean
        vmax = np.max(dataID[:, 10])
        vmean = np.mean(dataID[:, 10])
        plevelmean = np.mean(dataID[:, 8])
        for frame in frames:
            dataIDFrame = dataID[dataID[:, 5] == frame][0]
            # the output of the classification
            Plevel = dataIDFrame[8]
            # Categorize pushing behaviors into two categories
            if Plevel > 2.5:
                Plevel = 1
            else:
                Plevel = 0
            ###################################################
            Speed = dataIDFrame[10]
            idDesiredDir = [-dataIDFrame[6], -dataIDFrame[7]]
            idVelocityDir = [dataIDFrame[11], dataIDFrame[12]]
            Direction = AngleBetween(idVelocityDir, idDesiredDir)
            Output = [Plevel, Speed, Direction]
            # the input of the classification
            InputData = data[data[:, 5] == (frame - tAnti)]
            if len(InputData[InputData[:, 4] == id]) == 0:
                continue
            FSpace, FSpeed, Fdensity, Fplevel = InfoSurrounding(id, InputData, width, N)
            MLData.append([id, frame, vmax, vmean, *Output, plevelmean, *FSpace, *FSpeed, *Fdensity,
                           *Fplevel])
    MLData = np.array(MLData)
    return MLData


# check the function spaceSurrounding
def checkSpace(id, InputData, Features):
    if id != 16:
        return
    idData = InputData[InputData[:, 4] == 16][0]
    if idData[6] != 0.3444:
        return
    print("\nCheck the space surrounding for id: {}".format(id))
    neighbors = np.unique(InputData[:, 4])
    for neighbor in neighbors:
        neData = InputData[InputData[:, 4] == neighbor][0]
        nePos = [neData[6], neData[7]]
        if neighbor == id:
            print("id position: ({},{})".format(*nePos))
        else:
            print("neighbor pos: ({},{})".format(*nePos))
    print("features: {}".format(Features))


def InfoSurrounding(id, InputData, width, N):
    idData = InputData[InputData[:, 4] == id][0]
    idPos = [idData[6], idData[7]]
    idDesiredDir = [-idData[6], -idData[7]]
    # idVelocityDir = [idData[11], idData[12]]
    neighbors = np.unique(InputData[:, 4])
    FeatureSpace = []
    FeatureSpeed = []
    FeatureDensity = []
    FeaturePlevel = []
    # space in the specefic region
    for index in range(0, N):
        angle = index * math.pi * 2 / N
        # The desired direction is tested firstly
        direction = Rotation(idDesiredDir, angle)
        # the space to neighbors
        space = 100000  # considering the existing of walls, the initial value is bigger enough
        vs = []
        ds = []
        ps = []
        for neighbor in neighbors:
            if neighbor == id:
                continue
            neData = InputData[InputData[:, 4] == neighbor][0]
            nePos = [neData[6], neData[7]]
            dist12 = distance(idPos, nePos)
            ep12 = [nePos[0] - idPos[0], nePos[1] - idPos[1]]
            # angleBetween=abs(math.arccos())
            sp = scalarProduct(UnitVector(ep12), UnitVector(direction))
            if sp > 1:
                sp = 1
            elif sp < -1:
                sp = -1
            temp = math.acos(sp)
            if temp <= math.pi / N:
                neVelocityDir = [neData[11], neData[12]]
                neDireciton = AngleBetween(neVelocityDir, idDesiredDir)
                if neData[9] > 0:  # only for the person in the measurement area
                    vs.append(neData[10] * math.cos(neDireciton))
                    ds.append(neData[9])
                    ps.append(neData[8])
                if dist12 < space:
                    space = dist12
        space = space - 2 * bodySize  # Since the space to wall is different
        space2Wall = space2Walls(width, idPos, UnitVector(direction)) - bodySize
        if space2Wall < space:
            space = space2Wall
        FeatureSpace.append(space)
        # how to deal with the situation that no neighbors in the region (should be 0 and 0)
        if len(vs) == 0:
            vs.append(0)
        if len(ds) == 0:
            ds.append(0)
        if len(ps) == 0:
            ps.append(0)
        FeatureSpeed.append(np.mean(vs))
        FeatureDensity.append(np.mean(ds))
        FeaturePlevel.append(np.mean(ps))
    return FeatureSpace, FeatureSpeed, FeatureDensity, FeaturePlevel


# the space to the wall on the direction
def space2Walls(width, pos, dir):
    walls = [[(-width / 2, 7), (width / 2, 7)],
             [(width / 2, 0), (width / 2, 7)],
             [(width / 2, 0), (0.25, 0)],
             [(0.25, 0), (0.25, -1)],
             [(0.25, -1), (-0.25, -1)],
             [(-0.25, 0), (-0.25, -1)],
             [(-width / 2, 0), (-0.25, 0)],
             [(-width / 2, 0), (-width / 2, 7)]
             ]
    space = 10000
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
            wallLength = Norm(wVec)
            length = scalarProduct(pVec, wVec) / wallLength
            ExtraVec = (wVec[0] * length / wallLength, wVec[1] * length / wallLength)
            pt = (wp1[0] + ExtraVec[0], wp1[1] + ExtraVec[1])
            vec2PT = (pt[0] - pos[0], pt[1] - pos[1])
            dis2Pt = Norm(vec2PT)
            space2Wall = dis2Pt * dis2Pt / scalarProduct(vec2PT, dir)
            if space2Wall > 0 and space2Wall < space:
                space = space2Wall
            # print('id={:.0f} frame={:.0f}'.format(d[0],d[1]))
            # print('pos={} dir={}'.format(pos,dir))
            # print('wp1={} wp2={}'.format(wp1, wp2))
            # print('vec1={} vec2={}'.format(vec1,vec2))
            # print('crossP1={} crossP2={}'.format(crossP1,crossP2))
            # print('wall={}'.format(wall))
            # print('space={}'.format(space2Wall))
    return space


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
    # The pedestrian should be in the measurement area
    data = data[data[:, 7] > 0]
    data = data[data[:, 7] < 6]
    # check the density just in case
    data = data[data[:, 9] > 0]
    # the first several seconds should be ignored
    data = data[data[:, 5] > minFrame]
    return data


def writeMLfile(MLData, MLfile, N):
    f = open(MLfile, 'w')
    f.write(
        "#Note: when no neighbors in the region, speed density plevel are all set as 0.\n")
    f.write("#id[0]\tframe[1]\tvmax[2]\tvmean[3]\t"
            "pLevel[4]\tspeed[5]\tdirection[6]\tplevelmean[7]\tspace[8,{}]\tnspeed[{},{}]\tndensity[{},{}]\tnplevel[{},{}]\n".format(
        7 + N, 8 + N, 7 + 2 * N, 8 + 2 * N, 7 + 3 * N, 8 + 3 * N, 7 + 4 * N))
    for data in MLData:
        for index, num in enumerate(data):
            num = float(num)
            if index < len(data) - 1:
                f.write("{:.3f}\t".format(num))
            else:
                f.write("{:.3f}\n".format(num))
    f.close()


if __name__ == '__main__':
    cwd = os.getcwd()
    print("INFO: Current work directory is {}".format(cwd))
    # import the data
    if os.path.exists("AllData.txt"):
        print("INFO: AllData.txt is already exist")
        Alldata = np.loadtxt("AllData.txt")
    else:
        Alldata = GenerateAllDataFile()
    # clean the data
    Alldata = CleanData(Alldata)
    # Create the folder for saving result
    resultFolder = "MLdata"
    mkdir(resultFolder)
    os.chdir(resultFolder)
    # anticipation time, because I think the decision is based on prediction
    tAntis = [0,5,10,15,20,25,30,35,40,45,50]
    # the number of features
    Ns = [2,4,8,16,32,64]
    # the index of trajs
    indexs = np.unique(Alldata[:, 0])
    for tAnti in tAntis:
        for N in Ns:
            print("INFO: Start generating data (tAnti={}, N={})".format(tAnti, N))
            dirName = "tAnti_{}_N_{}".format(tAnti, N)
            mkdir(dirName)
            for index in indexs:
                data = Alldata[Alldata[:, 0] == index]
                width = data[0][2]
                conditions = ['crowd', 'queue']
                motivations = ['h0', 'h-']
                print("INFO: index={} width={} condition={} motivation={}".format(index, width,
                                                                                  conditions[int(
                                                                                      data[0][1])],
                                                                                  motivations[int(
                                                                                      data[0][3])]))
                MLData = GenerateMLData_InfoDistribution(data, tAnti, width, N)
                MLfile = "MLData_{}_{}.txt".format(index, motivations[int(data[0][3])])
                writeMLfile(MLData, MLfile, N)
                shutil.move(MLfile, dirName)
