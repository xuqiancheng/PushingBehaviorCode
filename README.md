The scripts used to analyze pedestrian pushing behavior.
Please get in touch with xuqianchengdut@gmail.com if you have any problems.

# a_ExperimentAnalysis
The original experimental data are collected in the folder **TrajInfo**, which contains both trajectory and behavioral data from experiments conducted under varying corridor widths and motivation levels. 

### P0_Distribution.py
Analyze the distributions of pedestrian's free pushing intensity.
Generated figures include:
- **P0DistHist.png**: The distributions of pedestrians' free pushing intensity for each experiment are illustrated using histograms and KDE curves. These distributions are fitted using bimodal functions, with the corresponding parameters provided within the figures. 
- **P0DistHist.png**: The comparision of kernel density estimates across experiments with varying corridor widths and motivation levels.
- **FeatureRelations**: The relationships between movement variables, including the distance from the initial position to the midpoint of the entrance, the time taken to travel from the initial position to the entrance, the mean density and the mean speed over the period from leaving the initial position until reaching the entrance.
- **FeatureDistributions**: The distributions of each movement variable.

### V0_Distribution.py
Analyze the distributions of pedestrian's free speed.
Generated figures include:
- **V0Distributions.png**: The distribution of pedestrians' max speed for each expeiments. These distributions are fitted using Gaussian distribution function, with the corresponding parameters provided within the figures. 

### MS_Difference.py
Show the differences between the movement strategies of pushing and non-pushing behaviors.
Generated figures include:
- **DeviationAngleDistributions.png**: The probability density function (PDF) of pedestrians' deviation angles during pushing and non-pushing behaviors.
- **SpeedDistributions.png**: The PDFs of pedestrians' speeds during pushing and non-pushing behaviors. 
- **DensityDistribution.png**: The PDFs of pedestrians' densities during pushing and non-pushing behaviors. 
- **HeadwayScatter.png**: The relationship between pedestrians' available distance in the direction of movement and their corresponding speed during pushing and non-pushing behaviors, is shown with the scatter plot. 
- **HeadwayTendency.png**: The relationship between pedestrians' available distance in the direction of movement and their corresponding speed during pushing and non-pushing behaviors, is shown with the mean value tendency.
- **FDTendency.png**: The relationship between pedestrians' speed and density during pushing and non-pushing behaviors, is shown with the mean value tendency.

# b_RFClassifier
Train and evaluate the random forest classifiers for predicting pedestrians' behaviors.
The original experimental data are collected in the folder **TrajInfo**.
### 0_GeneraetMLdata.py
Process the original data to a suitable format for machine learning.
Adjustable parameters include:
- ***tAntis=[]***: list of the adopted anticipation time 
- ***Ns=[]***: list of the adopted discrete regions number

The script will generate the machine learning datasets for all combinations of elements in ***tAntis*** and ***Ns***.
The generated datasets are collected in the folder **MLdata**.

### 1_RFApproach1.py
Train the random forest classifier with approach 1, which uses only the feature vector representing the state of neighbors as the input.
Adjustable parameters include:
- ***param_test = {'n_estimators': range()}***: The search range of the parameter 'n_estimators' in the random forest classifier.
- ***param_test = {'max_depth': range()}***: The search range of the parameter 'max_depth' in the random forest classifier.

The perfomence of the trained random forest classifer and the weights of input features are presented in the folder **RFApproach1**.

### 2_RFApproach2.py
Train the random forest classifier with approach 2, which combines the feature vector representing the state of neighbors and the pedestrian's free pushing intensity as the input.
Adjustable parameters include:
- ***param_test = {'n_estimators': range()}***: The search range of the parameter 'n_estimators' in the random forest classifier.
- ***param_test = {'max_depth': range()}***: The search range of the parameter 'max_depth' in the random forest classifier.

The perfomence of the trained random forest classifer and the weights of input features are presented in the folder **RFApproach2**.

### 3_MLAnalysis.py
Evaluate the performance of classifiers trained with different approaches and parameters.
Generated figures include:
- **AccuracyHeatMapRFApproach1.png**: The performence of classifiers trained with approach 1.
- **AccuracyHeatMapRFApproach2.png**: The performence of classifiers trained with approach 2.
- **RFGroupWeights1.png**: The weights of input features in the classifiers trained trained with approach 1. 
- **RFGroupWeights2.png**: The weights of input features in the classifiers trained trained with approach 2. 
- **RFApproach1weightsCurve.png**: The curves of features weight in the classifiers trained trained with approach 1. 
- **RFApproach2weightsCurve.png**: The curves of features weight in the classifiers trained trained with approach 1. 

# c_Simulation
The default input files for simulations are collected in the folder **DefaultFiles**.
Experimental data for comparison is saved in **ExperimentalData.txt**.
Simulations are implemented with the software **jpscore**. 
The source code can be found in  https://github.com/xuqiancheng/jpscore.git (Branch 'PushModel'). 

Four versions of **jpscore** are provided, which are **jpscore.exe** (using random forest classifer to predict pushing behavior), **jpscore_push.exe** (all agents push), **jpscore_nopush.exe** (no agents push), and **jpscore_random.exe** (agents push randomly).

### 0_RunSimulations.py
Run simulations with different parameters and versions of **jpscore.exe**.
Adjustable parameters include:
- ***repeatNumber***: The number of simulation repetitions.
- ***jpsV***: The version of **jpscore**, **jpscore.exe** (0), **jpscore_random.exe** (1), **jpscore_nopush.exe** (2), and **jpscore_push.exe** (3).
- ***index1***: The first investigated parameter.
- ***index2***: The second investigated parameter.
- ***aNorms,DNorms***: Adjusting the the range of the impact from neighbors on the direction of movement for agents engaging in non-pushing behavior.
- ***aPushs,DPushs***: Adjusting the the range of the impact from neighbors on the direction of movement for agents engaging in pushing behavior.
- ***aForces,DForces***: Adjusting the the range of the acceleration caused by contact with neighbors.
- ***TNorms,DNorms***: Adjusting the speed-headway relationships for agents engaging in non-pushing behavior.
- ***TPushs,DPushs***: Adjusting the speed-headway relationships for agents engaging in pushing behavior.
- ***aWalls,DWalls***: Adjusting the the range of the impact from walss on the direction of movement.
- ***Tds***: Adjusting the rate of agents' turning process.

Simulations will be implemented for all combinations of the two investigated parameters.
The trajectory and behavioral data will be saved in the corresponding folders.

### 1_AnalyzeTrajs.py
Evaluate simulation results.
Adjustable parameters include:
- ***index1***: The first investigated parameter should be the same as in **0_RunSimulations.py**.
- ***index2***: The second investigated parameter should be the same as in **0_RunSimulations.py**.

The generated figures will show the influence of the two investigated parameters on the proportion of pushing behaviors and the mean time lapse between two consecutive agents entering the entrance.




