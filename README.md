# a_ExperimentAnalysis
The original experimental data are collected in the folder **TrajInfo**.
## P0_Distribution.py
Analyze the distributions of pedestrian's free pushing intensity.
## V0_Distribution.py
Analyze the distributions of pedestrian's free speed.
## MS_Difference.py
Show the differences between the movement strategies of pushing and non-pushing behaviors.

# b_RFClassifier
Train and evaluate the random forest classifiers for predicting pedestrians' behaviors.
The original experimental data are collected in the folder **TrajInfo**.
## 0_GeneraetMLdata.py
Process the original data to a suitable format for machine learning.
## 1_RFApproach1.py
Train the random forest classifier with approach 1.
## 2_RFApproach1.py
Train the random forest classifier with approach 2.
## 3_MLAnalysis.py
Evaluate the performance of classifiers trained with different approaches and parameters.

# c_Simulation
The default input files for simulations are collected in the folder **DefaultFiles**.
Simulations are implemented with the software **jpscore.exe**.
Four version of **jpscore.exe** are provided, which are **jpscore.exe** (using random forest classifer), **jpscore_push.exe** (all agents push), **jpscore_nopush.exe** (no agents push), and **jpscore_random.exe** (agents push randomly).
## 0_RunSimulations.py
Run simulations with different parameters and jpscore version.
## 1_AnalyzeTrajs.py
Analyze simulation results.





