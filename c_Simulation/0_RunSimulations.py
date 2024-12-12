# Author Qiancheng
# Date 2024.09.18
# Purpose run simulations with different parameters
import os
import glob
import shutil
import numpy as np
import subprocess
import logging
from os import path
from shutil import rmtree, move
import errno

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def make_dir(path):
    if os.path.exists(path):
        rmtree(path)
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def update_tag_value(root, tag, value):
    for rank in root.iter(tag):
        rank.text = str(value)


def update_attrib_value(root, attr_tag, value):
    # location
    # print ("update_attrib_value: ", attr_tag, value)
    # raw_input()
    if attr_tag == "location":  # e.g. location
        for r in root.iter():
            if attr_tag in r.attrib:
                r.attrib[attr_tag] = str(value)
        return

    attr = attr_tag.split("-")[1]
    cor_tag = attr_tag.split("-")[0]

    for r in root.iter(cor_tag):
        if attr in r.attrib:
            r.attrib[attr] = str(value)


if __name__ == '__main__':
    if os.path.exists('log.log'):
        os.remove('log.log')
    logging.basicConfig(filename='log.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
    # 0 set parameters
    repeatNumber = 2
    jpsV=0
    seeds = [100 * i for i in range(repeatNumber)]
    #############################################
    aNorms = [3,3.2,3.4,3.6,3.8,4]
    DNorms = [0.01, 0.02, 0.05, 0.1]
    #############################################
    aPushs = [2,2.2,2.4,2.6,2.8,3]
    DPushs = [0.01, 0.02, 0.05, 0.1]
    #############################################
    TNorms = [0.1,0.2,0.3,0.4,0.5]
    TPushs = [0.05,0.1,0.15,0.20,0.25]
    #############################################
    SPushs = [0.15,0.16,0.17,0.18,0.19,0.2]
    SNorms = [0.05,0.06,0.07,0.08,0.09,0.1]
    #############################################
    aForces = [0.5, 1, 2, 3, 4, 5]
    DForces = [0.05, 0.1, 0.2, 0.3, 0.4]
    #############################################
    aWalls=[1, 2, 3, 4, 5, 6]
    DWalls=[0.01, 0.02, 0.05]
    #############################################
    Tds=[0.1, 0.2, 0.3, 0.4, 0.5]
    #############################################
    paraNames = ['force_ped-a', 'force_ped-D', 'PVM-apush', 'PVM-Dpush', 'GCVM-Ts', 'PVM-Tpush',
                 'PVM-Spush', 'PVM-Snorm','PVM-aForce', 'PVM-DForce', 'force_wall-a','force_wall-D','GCVM-Td']
    paras = [aNorms, DNorms, aPushs, DPushs, TNorms, TPushs, SPushs,SNorms,aForces,DForces, aWalls,DWalls,Tds]
    jpscores=['jpscore.exe','jpscore_random.exe','jpscore_nopush.exe','jpscore_push.exe']
    #############################################
    DefaultParas = {'force_ped-a': 3.2, 'force_ped-D': 0.05,
                    'PVM-apush': 2.8, 'PVM-Dpush': 0.01,
                    'GCVM-Ts': 0.3, 'PVM-Tpush': 0.2,
                    'PVM-Spush': 0.15, 'PVM-Snorm': 0.08,
                    'PVM-aForce': 2, 'PVM-DForce': 0.4,
                    'force_wall-a':2,'force_wall-D':0.02,
                    'GCVM-Td':0.1}
    #############################################
    # 1 Generate Folders with different parameters
    logging.info("Generate Folders with different parameters.")
    print("Generate Folders with different parameters.")
    iniFolders = []
    index1 = 4
    index2 = 5
    for para1 in paras[index1]:
        for para2 in paras[index2]:
            para1Name = paraNames[index1]
            para2Name = paraNames[index2]
            folderName = para1Name + '_{}_'.format(para1) + para2Name + '_{}'.format(para2)
            logging.info("Create folder {}.".format(folderName))
            print("Create folder {}.".format(folderName))
            make_dir(folderName)
            iniFolders.append(folderName)
            iniFiles = glob.glob('DefaultFiles/*.xml')
            for inifile in iniFiles:
                fileName = os.path.basename(inifile)
                if fileName.split('_')[0] == 'geometry':
                    shutil.copy(inifile, folderName)
                else:
                    tree = ET.parse(inifile)
                    root = tree.getroot()
                    # updata the values of parameters based on the values in the DefaultParas
                    for DefaultPara in DefaultParas.keys():
                        update_attrib_value(root,DefaultPara,DefaultParas[DefaultPara])
                    update_attrib_value(root, para1Name, para1)
                    update_attrib_value(root, para2Name, para2)
                    for seed in seeds:
                        update_tag_value(root, 'seed', seed)
                        newInfile = 'seed_{}_'.format(seed) + fileName
                        trajName = 'traj_seed_{}_'.format(seed) + \
                                   fileName.split('inifile_')[1].split('.xml')[0] + '.txt'
                        update_attrib_value(root, 'location', trajName)
                        update_tag_value(root,'exit_crossing_strategy',1)
                        # logging.info("    Generate inifile {}.".format(newInfile))
                        shutil.copy(inifile, newInfile)
                        tree.write(newInfile)
                        shutil.move(newInfile, folderName)
    #############################################
    # 2_Run Simulations
    # close the output of jpscore
    logging.info("Run simulations.")
    print("Run simulations")
    for iniFolder in iniFolders:
        logging.info("Genetate trajectories in {}.".format(iniFolder))
        print("Genetate trajectories in {}.".format(iniFolder))
        inifiles = glob.glob('{}/*.xml'.format(iniFolder))
        for inifile in inifiles:
            fileName = os.path.basename(inifile)
            if fileName.split('_')[0] == 'geometry':
                continue
            else:
                subprocess.call([jpscores[jpsV], inifile])
