#SciKit DSC540 HW1
'''created by Casey Bennett 2018, www.CaseyBennett.com'''

import sys
import csv
import math
import numpy as np
from operator import itemgetter
import time
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, chi2
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split, cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import KBinsDiscretizer, scale

#Handle annoying warnings
import warnings, sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


#############################################################################
#
# Global parameters
#
#####################

target_idx=0                                        #Index of Target variable
cross_val=1                                         #Control Switch for CV                                                                                                                                                      
norm_target=0                                       #Normalize target switch
norm_features=0                                     #Normalize target switch
binning=1                                           #Control Switch for Bin Target
bin_cnt=2                                           #If bin target, this sets number of classes
feat_select=0                                       #Control Switch for Feature Selection                                                                                   
fs_type=2                                           #Feature Selection type (1=Stepwise Backwards Removal, 2=Wrapper Select, 3=Univariate Selection)                        
lv_filter=0                                         #Control switch for low variance filter on features
feat_start=1                                        #Start column of features
k_cnt=5                                             #Number of 'Top k' best ranked features to select, only applies for fs_types 1 and 3

#Set global model parameters
rand_st=1                                           #Set Random State variable for randomizing splits on runs


#############################################################################
#
# Load Data
#
#####################

file_paths = ['C:/Users/wasay/Desktop/DSC 540/Final/data.csv', 
               'C:/Users/wasay/Desktop/DSC 540/Final/dataNoCurricular.csv', 
               'C:/Users/wasay/Desktop/DSC 540/Final/featureEngineering.csv']

sns.set(style="whitegrid")
custom_colors = ['blue', 'green', 'red']
custom_labels = ['D1: original','D2: 12 features removed', 'D3: feature engineering']
plt.figure(figsize=(8, 6)) 
for idx, file_path in enumerate(file_paths):


    file1= csv.reader(open(file_path), delimiter=',', quotechar='"')

    #Read Header Line
    header=next(file1)            

    #Read data
    data=[]
    target=[]
    for row in file1:
        #Load Target
        if row[target_idx]=='':                         #If target is blank, skip row                       
            continue
        else:
            target.append(float(row[target_idx]))       #If pre-binned class, change float to int

        #Load row into temp array, cast columns  
        temp=[]
                    
        for j in range(feat_start,len(header)):
            if row[j]=='':
                temp.append(float())
            else:
                temp.append(float(row[j]))

        #Load temp into Data array
        data.append(temp)
    
    #Test Print
    print(header)
    print(len(target),len(data))
    print('\n')

    data_np=np.asarray(data)
    target_np=np.asarray(target)
    #############################################################################
    #
    # Preprocess data
    #
    ##########################################
    #############################################################################
    #
    # Feature Selection
    #
    ##########################################
    #Low Variance Filter
    #Feature Selection
    #############################################################################
    #
    # Train SciKit Models
    #
    ##########################################
    
    print('--ML Model Output--', '\n')

    #Test/Train split
    data_train, data_test, target_train, target_test = train_test_split(data_np, target_np, test_size=0.35)

    ####Classifiers####
    if binning==1 and cross_val==0:
        #SciKit
        '''Test/Train split unused in this homework, skip down to CV section'''
    

                                                                                                                            
    
    ####Cross-Val Classifiers####
    if binning==1 and cross_val==1:
        #Setup Crossval classifier scorers
        scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}                                                                                                                
        
        #SciKit Gradient Boosting - Cross Val
        start_ts=time.time()
        clf= GradientBoostingClassifier(n_estimators = 200, loss = 'log_loss', learning_rate = 0.1, max_depth = 3, min_samples_split = 3, random_state = rand_st)
        scores= cross_validate(estimator = clf, X = data_np, y = target_np, scoring = scorers, cv = 5)

        scores_Acc = scores['test_Accuracy']                                                                                                                                    
        print("Gradient Boosting Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
        scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
        print("Gradient Boosting AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
        print("CV Runtime:", time.time()-start_ts)

        # Generate predicted probabilities for the positive class using cross_val_predict
        y_pred_proba = cross_val_predict(clf, data_np, target_np, cv=5, method="predict_proba")[:, 1]

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(target_np, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve using Seaborn
        
        sns.lineplot(x=fpr, y=tpr, color=custom_colors[idx], lw=2, label=f'{custom_labels[idx]} (AUC = {roc_auc:.2f})')
sns.lineplot(x=[0, 1], y=[0, 1], color='black', linestyle='--', label="0.5")

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Gradient Boosting ROC Curve No FS', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig('plot1.png')
plt.close()
     
