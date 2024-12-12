import glob
import os
import shutil
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.cm as cm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report
import pandas

sm = cm.ScalarMappable(cmap=cm.jet)
sm.set_clim(vmin=0, vmax=8)
n_jobs = -1


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def Heatmap(ax, data, title):
    n = len(data)
    m = 1
    # The area without data will be set as blank
    rad = np.linspace(0, 10, m + 1)
    a_fill = np.linspace(0, 2 * np.pi, 64 + 1) - (np.pi / n)
    r_fill, th_fill = np.meshgrid(rad, a_fill)
    # Here the purpose of times is making the shape of heatmap as circle
    times = int(64 / n)
    Hdata = []
    for d in data:
        for i in range(times):
            Hdata.append([d])
    Hdata = np.array(Hdata)
    cm = ax.pcolormesh(th_fill, r_fill, Hdata, cmap='binary', vmin=0)
    ax.plot(a_fill, r_fill, ls='--', color='k')
    ax.set_title(title, pad=15, fontsize=15)
    ax.set_yticks([])
    plt.colorbar(cm, ax=ax, pad=0.1)


def CombineInfos(files):
    print("INFO: Combine MLdata")
    f = open('AllMldata.txt', 'w')
    f.write(
        "#Note: when no neighbors in the region, speed (projection on the desired direction of ped), density and plevel are all set as 0.\n")
    f.write('# motivation h0(0) h-(1)\n')
    f.write("#index[0]\tmotivation[1]\tid[2]\tframe[3]\tvmax[4]\tvmean[5]\t"
            "pLevel[6]\tspeed[7]\tdirection[8]\tplevelmean[9]\tspace[10,{}]\tnspeed[{},{}]\tndensity[{},{}]\tnplevel[{},{}]\n"
            .format(9 + N, 10 + N, 9 + 2 * N, 10 + 2 * N, 9 + 3 * N, 10 + 3 * N, 9 + 4 * N))
    for file in files:
        traj_index = float(file.split('_')[-2])
        motivation = file.split('_')[-1]
        moti_index = 0
        if motivation == 'h-.txt':
            moti_index = 1
        data = np.loadtxt(file)
        for d in data:
            # change the id to unique id
            uniqueID = int(traj_index) * 100 + int(d[0])
            d[0] = uniqueID
            f.write('{:.0f}\t{:.0f}\t'.format(traj_index, moti_index))
            for i, num in enumerate(d):
                num = float(num)
                if i < len(d) - 1:
                    f.write("{:.3f}\t".format(num))
                else:
                    f.write("{:.3f}\n".format(num))
    f.close()
    alldata = np.loadtxt('AllMLdata.txt')
    return alldata


def myRF_groupIDs(data, features, figname, foldername):
    # choose features for ML
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    X = data[:, features]
    y = data[:, 6]
    groups = data[:, 2]
    # tuning parameters
    # split based on the ids
    GSS = GroupShuffleSplit(test_size=.2, n_splits=1, random_state=5)
    splitIndex = next(GSS.split(X, y, groups=groups))
    trainIndex = list(splitIndex[0])
    testIndex = list(splitIndex[1])
    X_train = X[trainIndex, :]
    y_train = y[trainIndex]
    X_test = X[testIndex, :]
    y_test = y[testIndex]
    groups_train = groups[trainIndex]
    # 0_the number of decision trees range(1,202,10)
    GSS_search = GroupShuffleSplit(test_size=.2, n_splits=5, random_state=7)
    print("INFO: search n_estimators")
    param_test = {'n_estimators': range(1, 202, 10)}
    gsearch = GridSearchCV(
        estimator=RandomForestClassifier(max_features='sqrt', random_state=10, n_jobs=n_jobs),
        param_grid=param_test, scoring='f1_micro',
        cv=GSS_search.split(X_train, y_train, groups=groups_train))
    gsearch.fit(X_train, y_train)
    para1 = gsearch.best_params_['n_estimators']
    scores = gsearch.cv_results_['mean_test_score']
    # save result
    writeGSresult('n_estimators', param_test['n_estimators'], scores, gsearch, figname, foldername)
    ax[0].plot(param_test['n_estimators'], scores, 'o-', color='steelblue')
    ax[0].set_title('best={} score={:.4f}'.format(para1, gsearch.best_score_), fontsize=15)
    ax[0].set_xlabel('n_estimators', fontsize=15)
    ax[0].set_ylabel('f1_score', fontsize=15)
    # 1_the number of max_depth range(1,52,2)
    print("INFO: search max_depth")
    param_test = {'max_depth': range(1, 52, 2)}
    gsearch = GridSearchCV(
        estimator=RandomForestClassifier(n_estimators=para1, max_features='sqrt', random_state=10,
                                         n_jobs=n_jobs),
        param_grid=param_test, scoring='f1_micro',
        cv=GSS_search.split(X_train, y_train, groups=groups_train))
    gsearch.fit(X_train, y_train)
    para2 = gsearch.best_params_['max_depth']
    scores = gsearch.cv_results_['mean_test_score']
    # save result
    writeGSresult('max_depth', param_test['max_depth'], scores, gsearch, figname, foldername)
    ax[1].plot(param_test['max_depth'], scores, 'o-', color='steelblue')
    ax[1].set_title('best={} score={:.3f}'.format(para2, gsearch.best_score_), fontsize=15)
    ax[1].set_xlabel('max_depth', fontsize=15)
    ax[1].set_ylabel('f1_score', fontsize=15)
    ####################################################################################
    # # 2_the number of min_samples_split range(2,102,10)
    # print("INFO: search min_samples_split")
    # param_test = {'min_samples_split': range(2, 103, 10)}
    # gsearch = GridSearchCV(
    #     estimator=RandomForestClassifier(n_estimators=para1, max_depth=para2, max_features='sqrt',
    #                                      random_state=10, n_jobs=n_jobs),
    #     param_grid=param_test, scoring='f1_micro',
    #     cv=GSS_search.split(X_train, y_train, groups=groups_train))
    # gsearch.fit(X_train, y_train)
    # para3 = gsearch.best_params_['min_samples_split']
    # scores = gsearch.cv_results_['mean_test_score']
    # # save result
    # writeGSresult('min_samples_split', param_test['min_samples_split'], scores, gsearch, figname,
    #               foldername)
    # ax[1][0].plot(param_test['min_samples_split'], scores, 'o-', color='steelblue')
    # ax[1][0].set_title('best={} score={:.3f}'.format(para3, gsearch.best_score_),fontsize=15)
    # ax[1][0].set_xlabel('min_samples_split',fontsize=15)
    # ax[1][0].set_ylabel('f1_micro_test',fontsize=15)
    # # 3_the number of min_samples_leaf range(1,101,10)
    # print("INFO: search min_samples_leaf")
    # param_test = {'min_samples_leaf': range(1, 102, 10)}
    # gsearch = GridSearchCV(
    #     estimator=RandomForestClassifier(n_estimators=para1, max_depth=para2,
    #                                      min_samples_split=para3, max_features='sqrt',
    #                                      random_state=10, n_jobs=n_jobs),
    #     param_grid=param_test, scoring='f1_micro',
    #     cv=GSS_search.split(X_train, y_train, groups=groups_train))
    # gsearch.fit(X_train, y_train)
    # para4 = gsearch.best_params_['min_samples_leaf']
    # scores = gsearch.cv_results_['mean_test_score']
    # # save result
    # writeGSresult('min_samples_leaf', param_test['min_samples_leaf'], scores, gsearch, figname,
    #               foldername)
    # ax[1][1].plot(param_test['min_samples_leaf'], scores, 'o-', color='steelblue')
    # ax[1][1].set_title('best={} score={:.3f}'.format(para4, gsearch.best_score_),fontsize=15)
    # ax[1][1].set_xlabel('min_samples_leaf',fontsize=15)
    # ax[1][1].set_ylabel('f1_micro_test',fontsize=15)
    #########################################################################
    plt.savefig(figname)
    plt.close()
    plt.cla()
    plt.clf()
    shutil.move(figname, foldername)
    # calculate the accuracy and f1 score
    classifier = gsearch.best_estimator_
    trainAccuracy = classifier.score(X_train, y_train)
    report = classification_report(y_test, classifier.predict(X_test),
                                   target_names=["noPush", "Push"], output_dict=True)
    return classifier, trainAccuracy, report


def RFtraj(data, N):
    print("INFO: start Normalized random forest classification on trajectory level")
    GSRFolder = 'GridSearch_traj'
    mkdir(GSRFolder)
    # FeatureIndex is used to adjust the number of features used for classification
    FeatureIndexs = [list(range(9, 10 + 4 * N))]
    TrajIndexs = np.unique(data[:, 0])
    RFResults = []
    for trajIndex in TrajIndexs:
        print("INFO: train classifier based on traj {}".format(trajIndex))
        data_train = data[data[:, 0] == trajIndex]
        weights = {}
        # Here we only consider the case with all features
        for featureIndex in range(len(FeatureIndexs)):
            print("INFO: FeatureIndex {}".format(featureIndex))
            # training classifier
            figname = 'gridSearch_{}_{}.png'.format(trajIndex, featureIndex)
            features = FeatureIndexs[featureIndex]
            # Obtain the optimal classifier
            gsResults = myRF_groupIDs(data_train, features, figname, GSRFolder)
            classifier = gsResults[0]
            weights[featureIndex] = classifier.feature_importances_
            for testIndex in TrajIndexs:
                if testIndex == trajIndex:
                    continue
                else:
                    data_test = data[data[:, 0] == testIndex]
                    X = data_test[:, FeatureIndexs[featureIndex]]
                    y = data_test[:, 6]
                    accuracy = classifier.score(X, y)
                result = [trajIndex, featureIndex, *gsResults[1:], testIndex, accuracy]
                RFResults.append(result)
        plotWeights(weights, 'RFweights_traj_{}.png'.format(trajIndex))
    RFResults = np.array(RFResults)
    # write the result down
    f = open('RFClassification_traj.txt', 'w')
    f.write('#train_index[0]\tfeature_index[1]\ttrain_accuracy[2]\ttest_accuracy[3]\t'
            'test_index[4]\taccuracy[5]\n')
    for RFResult in RFResults:
        f.write('{:.0f}\t{:.0f}\t{:.3f}\t{:.3f}\t{:.0f}\t{:.3f}\n'.format(*RFResult))
    f.close()
    # plot figure
    figIndexs = [[110, 270, 50, 30], [120, 280, 60, 40]]
    motis = ['h0', 'h-']
    widths = [1.2, 3.4, 4.5, 5.6]
    fig, ax = plt.subplots(2, 4, figsize=(24, 12))
    for i in range(0, 2):
        for j in range(0, 4):
            figIndex = figIndexs[i][j]
            data_index = RFResults[RFResults[:, 0] == figIndex]
            scores = [[0, 0, 0, 0], [0, 0, 0, 0]]
            for k, testIndex in enumerate(figIndexs):
                for l, Index in enumerate(testIndex):
                    if Index == figIndex:
                        scores[k][l] = data_index[0, 3]
                    else:
                        scores[k][l] = data_index[data_index[:, 4] == Index][0, 5]
            trainScore = data_index[0, 2]
            testScore = data_index[0, 3]
            title = '{}_{} train: {:.3f} test: {:.3f}'.format(motis[i], widths[j],
                                                              trainScore, testScore)
            ax[i][j].plot(widths, scores[0], 'o', markersize=15, label='h0', color='steelblue')
            ax[i][j].plot(widths, scores[1], 'x', markersize=15, label='h-', color='indianred')
            ax[i][j].set_xlabel("Widths [m]", fontsize=15)
            ax[i][j].set_ylabel("Accuracy", fontsize=15)
            ax[i][j].set_title(title, fontsize=15)
            ax[i][j].legend(loc='best', fontsize=15)
            ax[i][j].set_ylim(0.3, 1)
    plt.savefig('RFClassification_traj.png')
    plt.close()
    plt.cla()
    plt.clf()


def RFmoti(data, N):
    print("INFO: start Normalized random forest classification on motivation level")
    GSRFolder = 'GridSearch_moti'
    mkdir(GSRFolder)
    FeatureIndexs = [list(range(9, 10 + 4 * N))]
    motis = np.unique(data[:, 1])
    RFResults = []
    motivations = ['h0', 'h-']
    for moti in motis:
        print("INFO: train classifier based on moti {}".format(motivations[int(moti)]))
        data_train = data[data[:, 1] == moti]
        weights = {}
        for featureIndex in range(len(FeatureIndexs)):
            print("INFO: FeatureIndex {}".format(featureIndex))
            # training classifier
            figname = 'gridSearch_{}_{}.png'.format(moti, featureIndex)
            features = FeatureIndexs[featureIndex]
            gsResults = myRF_groupIDs(data_train, features, figname, GSRFolder)
            classifier = gsResults[0]
            weights[featureIndex] = classifier.feature_importances_
            for moti2 in motis:
                if moti2 == moti:
                    continue
                else:
                    data_test = data[data[:, 1] == moti2]
                    X = data_test[:, FeatureIndexs[featureIndex]]
                    y = data_test[:, 6]
                    accuracy = classifier.score(X, y)
                result = [moti, featureIndex, *gsResults[1:], moti2, accuracy]
                RFResults.append(result)
        plotWeights(weights, 'RFweights_moti_{}.png'.format(motivations[int(moti)]))
    RFResults = np.array(RFResults)
    # write the result down
    f = open('RFClassification_moti.txt', 'w')
    f.write("# moti: 0(h0) 1(h-)\n")
    f.write('#train_moti[0]\tfeature_index[1]\ttrain_accuracy[2]\ttest_accuracy[3]\t'
            'test_moti[4]\taccuracy[5]\n')
    for RFResult in RFResults:
        f.write('{:.0f}\t{:.0f}\t{:.3f}\t{:.3f}\t{:.0f}\t{:.3f}\n'.format(*RFResult))
    f.close()
    # plot figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for i in [0, 1]:
        moti = motis[i]
        data_moti = RFResults[RFResults[:, 0] == moti]
        trainScore = data_moti[0, 2]
        testScore = data_moti[0, 3]
        if i == 0:
            scores = [data_moti[0, 3], data_moti[0, 5]]
        else:
            scores = [data_moti[0, 5], data_moti[0, 3]]
        title = '{} train: {:.3f} test: {:.3f}'.format(motivations[i], trainScore, testScore)
        ax[i].plot(motis, scores, 'o', markersize=15,
                   color='steelblue', label='testAccuracy')
        ax[i].set_xticks([0, 1], ['h0', 'h-'])
        ax[i].set_xlabel("Motivation", fontsize=15)
        ax[i].set_ylabel("Accuracy", fontsize=15)
        ax[i].set_title(title, fontsize=15)
        # ax[i].legend(loc='best', fontsize=15)
        ax[i].set_ylim(0.3, 1)
    plt.savefig('RFClassification_moti.png')
    plt.close()
    plt.cla()
    plt.clf()


def RFall(data, N):
    print("INFO: start random forest classification on all data")
    GSRFolder = 'GridSearch_all'
    mkdir(GSRFolder)
    FeatureIndexs = [list(range(9, 10 + 4 * N))]
    RFResults = []
    weights = {}
    for featureIndex in range(len(FeatureIndexs)):
        print("INFO: Feature {}".format(featureIndex))
        # training classifier
        figname = 'gridSearch_all_{}.png'.format(featureIndex)
        features = FeatureIndexs[featureIndex]
        # Training process
        gsResults = myRF_groupIDs(data, features, figname, GSRFolder)
        classifier = gsResults[0]
        weights[featureIndex] = classifier.feature_importances_
        result = [featureIndex, *gsResults[1:]]
        RFResults.append(result)
    plotWeights(weights, 'RFweights_all.png')
    # write the result down
    f = open('RFClassification_all.txt', 'w')
    f.write('#feature_index[1]\ttrain_accuracy[2]\ttest_accuracy[3]\n')
    for RFResult in RFResults:
        report = pandas.DataFrame(RFResult[2]).transpose()
        f.write('{:.0f}\t{:.3f}\t{:.3f}\t{:.0f}\t{:.0f}\n'.format(RFResult[0], RFResult[1],
                                                                  report.loc[
                                                                      'accuracy', 'precision'], 0,
                                                                  0))
        f.write('#Type: noPush(0)\tPush(1)\tmacro_avg(3)\tweighted_avg(4)\n')
        f.write('#Type\tprecision\trecall\tf1-score\tsupport\n')
        report = report.drop('accuracy')
        for i in range(4):
            f.write('{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.0f}\n'.format(i, *report.iloc[i]))
    f.close()
    # plot figure
    dataPlot = np.loadtxt('RFClassification_all.txt')
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    title = 'train: {:.3f} test: {:.3f}'.format(dataPlot[0, 1], dataPlot[0, 2])
    ax.plot([0, 1, 2], dataPlot[1, 1:4], 'o', color='steelblue', markersize=15, label='noPush')
    ax.plot([0, 1, 2], dataPlot[2, 1:4], '*', color='indianred', markersize=15, label='Push')
    ax.plot([0, 1, 2], dataPlot[3, 1:4], 's', color='green', markersize=15, label='macroAve')
    ax.plot([0, 1, 2], dataPlot[4, 1:4], '^', color='black', markersize=15, label='weightedAve')
    ax.set_xticks([0, 1, 2], ['precision', 'recall', 'f1-score'])
    ax.set_xlabel("Metrics", fontsize=15)
    ax.set_ylabel("Value", fontsize=15)
    ax.legend(loc='best')
    ax.set_title(title, fontsize=15)
    ax.set_ylim(0.3, 1)
    plt.savefig('RFClassification_all.png')
    plt.close()
    plt.cla()
    plt.clf()


def plotWeights(weights, figname):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), subplot_kw={'projection': 'polar'})
    featureType = ['space', 'speed', 'density', 'plevel']
    N = int((len(weights[0]) - 1) / 4)
    TotalWeight = [weights[0][0]]
    for i in range(4):
        Heatmap(axes[i], weights[0][(1 + i * N):(1 + (i + 1) * N)], featureType[i])
        TotalWeight.append(np.sum(weights[0][(1 + i * N):(1 + (i + 1) * N)]))
    plt.savefig(figname)
    plt.close()
    plt.cla()
    plt.clf()
    # write the data down
    filename = figname.split('png')[0] + 'txt'
    f = open(filename, 'w')
    f.write('# MeanPlevel [0] {:.3f}\n'.format(TotalWeight[0]))
    f.write('# Space [{:d},{:d}] {:.3f}\n'.format(1, N, TotalWeight[1]))
    f.write('# Speed [{:d},{:d}] {:.3f}\n'.format(1 + N, 2 * N, TotalWeight[2]))
    f.write('# Density [{:d},{:d}] {:.3f}\n'.format(1 + 2 * N, 3 * N, TotalWeight[3]))
    f.write('# Plevel [{:d},{:d}] {:.3f}\n'.format(1 + 3 * N, 4 * N, TotalWeight[4]))
    f.write("# featureNumber\tfeatureImportances\n")
    for i in range(4 * N+1):
        f.write('{}\t{:f}\n'.format(i, weights[0][i]))
    f.close()
    # Plot the hist figure
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    axes.bar(range(5), TotalWeight, tick_label=['meanPlevel'] + featureType, color='steelblue',
             edgecolor='k')
    axes.set_ylabel('importance', fontsize=15)
    axes.set_xlabel('features', fontsize=15)
    plt.savefig('TotalWeights_' + figname)
    plt.close()
    plt.cla()
    plt.clf()


def writeGSresult(paraName, params, scores, gsearch, figname, foldername):
    filename = figname.split('png')[0] + '_' + paraName + '.txt'
    f = open(filename, 'w')
    classifier = gsearch.best_estimator_
    classifier.get_params()
    f.write('# n_estimators {}\n'.format(classifier.get_params()['n_estimators']))
    f.write('# max_depth {}\n'.format(classifier.get_params()['max_depth']))
    f.write('# min_samples_split {}\n'.format(classifier.get_params()['min_samples_split']))
    f.write('# min_samples_leaf {}\n'.format(classifier.get_params()['min_samples_leaf']))
    f.write('# best score {}\n'.format(gsearch.best_score_))
    f.write('#{}\tscores\n'.format(paraName))
    for i in range(len(scores)):
        f.write('{}\t{:.5f}\n'.format(params[i], scores[i]))
    f.close()
    shutil.move(filename, foldername)


if __name__ == '__main__':
    dir = os.getcwd()
    # move in to the result data (the train and test data are separated by the id of pedestrian)
    mkdir('RFApproach2')
    os.chdir('RFApproach2')
    folders = os.listdir('{}/MLdata'.format(dir))
    for folder in folders:
        print("INFO: analyze {}".format(folder))
        mkdir(folder)
        os.chdir(folder)
        files = glob.glob('{}/MLdata/{}/*.txt'.format(dir, folder))
        N = int(folder.split('_')[-1])
        # Combine all data together (here the index should be unique)
        alldata = CombineInfos(files)
        # Random forest
        # using one experiment to train the classifier, then test others
        # RFtraj(alldata, N)
        # using one motivation to train the classifier, then test others
        # RFmoti(alldata, N)
        # using all data to train the classifier, then test others
        RFall(alldata, N)
        os.chdir('{}/RFApproach2'.format(dir))
