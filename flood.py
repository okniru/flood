#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:43:28 2022

@author: K
"""

import glob
import pandas as pd
import statistics as stats
import math
import numpy as np
import csv
from collections import Counter
from sklearn.metrics import accuracy_score,balanced_accuracy_score, precision_score,recall_score,roc_auc_score,f1_score,average_precision_score

rdm = 24

# get data temp file names

temp_files = glob.glob("Temp/*.csv",  recursive = True) # get file name
temp_list = []

for file in temp_files:
    df = pd.read_csv(file, index_col=False,skiprows = 4)
    df[len(df.columns)] = file
    temp_list.append(df)

# get data precip file names

precip_files = glob.glob("Precip/*.csv") # get file name
precip_list = []

for file in precip_files:
    df = pd.read_csv(file, index_col=False,skiprows = 4)
    df[len(df.columns)] = file
    precip_list.append(df)

# get data flood file names

flood_files = glob.glob("Floods/*.csv") # get file name
flood_list = []

for file in flood_files:
    df = pd.read_csv(file, index_col=False, header=2)
    df[len(df.columns)] = file
    flood_list.append(df)

#%%

# clean up datasets

all_temp = pd.concat(temp_list) # convert lists to a dataframe
all_temp = all_temp.reset_index(drop=True) # reset row index
all_temp.rename(columns={3: 'State'}, inplace = True) #rename the file column
all_temp.rename(columns={'Value': 'Temp'}, inplace = True) #rename the file column
all_temp["State"] = all_temp["State"].str.replace('Temp/', '', regex=True) #remove extra text
all_temp["State"] = all_temp["State"].str.replace('-tavg-12-12-1980-2021.csv', '', regex=True)
all_temp['Date'] = pd.to_datetime(all_temp.Date, format="%Y%m", yearfirst=True, errors='ignore') #change to date type
all_temp['Date'] = pd.DatetimeIndex(all_temp['Date']).to_period('Y') #change to year

all_precip = pd.concat(precip_list) # convert lists to a dataframe
all_precip = all_precip.reset_index(drop=True) # reset row index
all_precip.rename(columns={3: 'State'}, inplace = True) #rename the file column
all_precip.rename(columns={'Value': 'Precipitation'}, inplace = True) #rename the file column
all_precip["State"] = all_precip["State"].str.replace('Precip/', '', regex=True) #remove extra text
all_precip["State"] = all_precip["State"].str.replace('-pcp-12-12-1980-2021.csv', '', regex=True) #remove extra text
all_precip['Date'] = pd.to_datetime(all_precip.Date, format="%Y%m", yearfirst=True, errors='ignore') #change to date type
all_precip['Date'] = pd.DatetimeIndex(all_precip['Date']).to_period('Y') #change to year

all_flood = pd.concat(flood_list) # convert lists to a dataframe
all_flood = all_flood.reset_index(drop=True) # reset row index
all_flood.rename(columns={5: 'State'}, inplace = True) #rename the file column
all_flood.rename(columns={'Year': 'Date'}, inplace = True) #rename the file column

all_flood["State"] = all_flood["State"].str.replace('Floods/', '', regex=True) #remove extra text
all_flood["State"] = all_flood["State"].str.replace('time-series-', '', regex=True) #remove extra text
all_flood["State"] = all_flood["State"].str.replace('-1980-2021.csv', '', regex=True) #remove extra text

all_flood["Flooding Cost Range"] = all_flood["Flooding Cost Range"].str.replace('0-0', '0', regex=True) #correct and standardize amounts
all_flood["Flooding Cost Range"] = all_flood["Flooding Cost Range"].str.replace('0-5', '5', regex=True)
all_flood["Flooding Cost Range"] = all_flood["Flooding Cost Range"].str.replace('5-100', '50', regex=True)
all_flood["Flooding Cost Range"] = all_flood["Flooding Cost Range"].str.replace('100-250', '150', regex=True)
all_flood["Flooding Cost Range"] = all_flood["Flooding Cost Range"].str.replace('250-500', '350', regex=True)
all_flood["Flooding Cost Range"] = all_flood["Flooding Cost Range"].str.replace('25500', '350', regex=True)
all_flood["Flooding Cost Range"] = all_flood["Flooding Cost Range"].str.replace('500-1000', '750', regex=True)
all_flood["Flooding Cost Range"] = all_flood["Flooding Cost Range"].str.replace('1000-2000', '1500', regex=True)
all_flood["Flooding Cost Range"] = all_flood["Flooding Cost Range"].str.replace('2000-5000', '3500', regex=True)
all_flood["Flooding Cost Range"] = all_flood["Flooding Cost Range"].str.replace('2005000', '3500', regex=True)
all_flood["Flooding Cost Range"] = all_flood["Flooding Cost Range"].str.replace('5000-10000', '3500', regex=True)
all_flood["Flooding Cost Range"] = all_flood["Flooding Cost Range"].str.replace('10000-20000', '15000', regex=True)

all_flood["Flooding Cost Range"] = all_flood["Flooding Cost Range"].astype('int64') #convert to integer

all_flood['Date'] = pd.to_datetime(all_flood.Date, format="%Y", yearfirst=True, errors='ignore') #change to date type
all_flood['Date'] = pd.DatetimeIndex(all_flood['Date']).to_period('Y') #change to period

#%%
# join all files on state and date

flood = all_flood[all_flood.Date != '2021'] #remove 2021
flood = flood[all_flood.State != 'IA'] #remove Iowa

all_weather = all_temp.merge(all_precip, how='inner', on = ["State", "Date"], left_index=True, suffixes =('_T', '_P') )

# join weather values and flood labels

all_weather = pd.merge(all_weather, flood,  how='left', left_on=["State", "Date"], right_on = ["State", "Date"])
all_weather['Date'] = all_weather['Date'].astype(str)

import matplotlib.pyplot as plt

Measures = all_weather[['Temp','Anomaly_T','Precipitation','Anomaly_P']]
Measures.hist()
plt.show()

#%%

# normalize labels and ranges

from sklearn import preprocessing

X = all_weather.iloc[:,0:6]
y = all_weather.iloc[:,6:8]

lab = preprocessing.LabelEncoder()
X['Date'] = lab.fit_transform(X['Date'])
X['State'] = lab.fit_transform(X['State'])
y['Flooding Count'] = lab.fit_transform(y['Flooding Count'])
y['Flooding Cost Range'] = lab.fit_transform(y['Flooding Cost Range'])

minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
X[['Temp','Anomaly_T','Precipitation','Anomaly_P']] = minmax.fit(X[['Temp','Anomaly_T','Precipitation','Anomaly_P']]).transform(X[['Temp','Anomaly_T','Precipitation','Anomaly_P']])

Measures = X[['Temp','Anomaly_T','Precipitation','Anomaly_P']]
Measures.hist()
plt.show()

y_fl = y['Flooding Count']
y_flct = y['Flooding Cost Range']

# binarize multiclass labels

brz = preprocessing.LabelBinarizer()
y_fl_bin = pd.DataFrame(brz.fit_transform(y['Flooding Count']))
y_fl_bin.set_axis(["Fl 0", "Fl 1", "Fl 2"], axis=1, inplace=True)
y_fl_bin = y_fl_bin.iloc[:,1:3] #add all flood columns except for zeros
y_flct_bin = pd.DataFrame(brz.fit_transform(y['Flooding Cost Range']))
y_flct_bin.set_axis(["Flct 0", "Flct 1", "Flct 2", "Flct 3", "Flct 4", "Flct 5", "Flct 6", "Flct 7", "Flct 8"], axis=1, inplace=True)
y_flct_bin = y_flct_bin.iloc[:,1:9] #add all flood columns except for zeros

# dataset split per label type

from sklearn.model_selection import train_test_split

x_train, x_test, y_train_fl, y_test_fl = train_test_split(X,y['Flooding Count'],test_size=0.2, random_state=rdm)
x_train, x_test, y_train_flct, y_test_flct = train_test_split(X,y['Flooding Cost Range'],test_size=0.2, random_state=rdm)
x_train, x_test, y_train_fl_bin, y_test_fl_bin = train_test_split(X,y_fl_bin,test_size=0.2, random_state=rdm)
x_train, x_test, y_train_flct_bin, y_test_flct_bin = train_test_split(X,y_flct_bin,test_size=0.2, random_state=rdm)

# convert to array

y_fl = y_fl.to_numpy()
y_flct = y_flct.to_numpy()
y_fl_bin = y_fl_bin.to_numpy()
y_flct_bin = y_flct_bin.to_numpy()
y_train_fl = y_train_fl.to_numpy()
y_test_fl = y_test_fl.to_numpy()
y_train_flct = y_train_flct.to_numpy()
y_test_flct = y_test_flct.to_numpy()
y_train_fl_bin = y_train_fl_bin.to_numpy()
y_test_fl_bin = y_test_fl_bin.to_numpy()
y_train_flct_bin = y_train_flct_bin.to_numpy()
y_test_flct_bin = y_test_flct_bin.to_numpy()

Measures = x_train[['Temp','Anomaly_T','Precipitation','Anomaly_P']]
Measures.hist()
plt.show()

Measures = x_test[['Temp','Anomaly_T','Precipitation','Anomaly_P']]
Measures.hist()
plt.show()

#%%

# Flood count prediction

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

knn_cl = KNeighborsClassifier()
log_cl = LogisticRegression(random_state=rdm, multi_class='multinomial')
dt_cl = DecisionTreeClassifier(splitter='best', random_state=rdm)
rf_cl = RandomForestClassifier(n_estimators=10,random_state=rdm)
svc_cl = SVC(random_state=rdm, decision_function_shape='ovo')
vt_cl = VotingClassifier(estimators=[('knn', knn_cl), ('lr', log_cl), ('dt', dt_cl), ('rf', rf_cl), ('svc', svc_cl)],voting='hard')

column_names = ['Label', 'Model', 'Accuracy', 'Precision','Recall','F1','PR AUC']
scores = pd.DataFrame(columns=column_names)

for model in (knn_cl, log_cl, dt_cl, rf_cl, svc_cl, vt_cl):
    model.fit(x_train, y_train_fl)
    y_pred = model.predict(x_test)
    row = ['Fl Multiclass', str(model)[0:10], accuracy_score(y_test_fl, y_pred), 
           precision_score(y_test_fl, y_pred, average='weighted'), 
           recall_score(y_test_fl, y_pred, average='weighted'), 
           f1_score(y_test_fl, y_pred, average='weighted'), 
           float("NaN")]
    scores.loc[len(scores)] = row

for model in (knn_cl, dt_cl, rf_cl):
    model.fit(x_train, y_train_fl_bin)
    y_pred = model.predict(x_test)
    row = ['Fl Binary', str(model)[0:10], accuracy_score(y_test_fl_bin, y_pred), 
           precision_score(y_test_fl_bin, y_pred, average='weighted'), 
           recall_score(y_test_fl_bin, y_pred, average='weighted'), 
           f1_score(y_test_fl_bin, y_pred, average='weighted'), 
           average_precision_score(y_test_fl_bin, y_pred, average='weighted')]
    scores.loc[len(scores)] = row

# cross validate
    
for model in (knn_cl, log_cl, dt_cl, rf_cl, svc_cl, vt_cl):
    row = ['Fl Multi CV', str(model)[0:10], 
           cross_val_score(model,X,y_fl,cv=5,scoring='accuracy').mean(), 
           cross_val_score(model,X,y_fl,cv=5,scoring='precision_weighted').mean(), 
           cross_val_score(model,X,y_fl,cv=5,scoring='recall_weighted').mean(),  
           cross_val_score(model,X,y_fl,cv=5,scoring='f1_weighted').mean(), 
           float("NaN")]
    scores.loc[len(scores)] = row
    
for model in (knn_cl, dt_cl, rf_cl):
    row = ['Fl Bin CV', str(model)[0:10], 
           cross_val_score(model,X,y_fl_bin,cv=5,scoring='accuracy').mean(), 
           cross_val_score(model,X,y_fl_bin,cv=5,scoring='precision_weighted').mean(), 
           cross_val_score(model,X,y_fl_bin,cv=5,scoring='recall_weighted').mean(),  
           cross_val_score(model,X,y_fl_bin,cv=5,scoring='f1_weighted').mean(), 
           cross_val_score(model,X,y_fl_bin,cv=5,scoring='average_precision').mean()]
    scores.loc[len(scores)] = row
    
#%%

# Flood cost range prediction

for model in (knn_cl, log_cl, dt_cl, rf_cl, svc_cl, vt_cl):
    model.fit(x_train, y_train_flct)
    y_pred = model.predict(x_test)
    row = ['FlCt Multiclass', str(model)[0:10], accuracy_score(y_test_flct, y_pred), 
           precision_score(y_test_flct, y_pred, average='weighted'), 
           recall_score(y_test_flct, y_pred, average='weighted'), 
           f1_score(y_test_flct, y_pred, average='weighted'), 
           float("NaN")]
    scores.loc[len(scores)] = row

for model in (knn_cl, dt_cl, rf_cl):
    model.fit(x_train, y_train_flct_bin)
    y_pred = model.predict(x_test)
    row = ['FlCt Binary', str(model)[0:10], accuracy_score(y_test_flct_bin, y_pred), 
           precision_score(y_test_flct_bin, y_pred, average='weighted'), 
           recall_score(y_test_flct_bin, y_pred, average='weighted'), 
           f1_score(y_test_flct_bin, y_pred, average='weighted'), 
           average_precision_score(y_test_flct_bin, y_pred, average='weighted')]
    scores.loc[len(scores)] = row

# cross validate
    
for model in (knn_cl, log_cl, dt_cl, rf_cl, svc_cl, vt_cl):
    row = ['FlCt Multi CV', str(model)[0:10], 
           cross_val_score(model,X,y_flct,cv=5,scoring='accuracy').mean(), 
           cross_val_score(model,X,y_flct,cv=5,scoring='precision_weighted').mean(), 
           cross_val_score(model,X,y_flct,cv=5,scoring='recall_weighted').mean(),  
           cross_val_score(model,X,y_flct,cv=5,scoring='f1_weighted').mean(), 
           float("NaN")]
    scores.loc[len(scores)] = row
    
for model in (knn_cl, dt_cl, rf_cl):
    row = ['FlCt Bin CV', str(model)[0:10], 
           cross_val_score(model,X,y_flct_bin,cv=5,scoring='accuracy').mean(), 
           cross_val_score(model,X,y_flct_bin,cv=5,scoring='precision_weighted').mean(), 
           cross_val_score(model,X,y_flct_bin,cv=5,scoring='recall_weighted').mean(),  
           cross_val_score(model,X,y_flct_bin,cv=5,scoring='f1_weighted').mean(),  
           cross_val_score(model,X,y_flct_bin,cv=5,scoring='average_precision').mean()]
    scores.loc[len(scores)] = row
    
#%%

# update model names in scores table

scores['Model'] = scores['Model'].replace("KNeighbors", "KNN")
scores['Model'] = scores['Model'].replace("LogisticRe", "LR")
scores['Model'] = scores['Model'].replace("DecisionTr", "DT")
scores['Model'] = scores['Model'].replace("RandomFore", "RF")
scores['Model'] = scores['Model'].replace("SVC(decisi", "SVC")
scores['Model'] = scores['Model'].replace("VotingClas", "VT")

#%%

# flood count barplot

# accuracy

fl = scores[scores['Label'].str.contains(r'Fl Multiclass')]
fl_bin = scores[scores['Label'].str.contains(r'Fl Binary')]
fl_cv = scores[scores['Label'].str.contains(r'Fl Multi CV')]
fl_bin_cv = scores[scores['Label'].str.contains(r'Fl Bin CV')]

fig, ((ax_fl, ax_fl_bin), (ax_fl_cv, ax_fl_cv_bin)) = plt.subplots(2, 2)
fig.suptitle('Flooding Count Accuracy')

ax_fl_bar = ax_fl.bar(fl['Model'], fl['Accuracy'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl.set(ylabel='Accuracy')
ax_fl.set_title('Flood Count Multiclass')
ax_fl.bar_label(ax_fl.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl.margins(y=0.2)

ax_fl_bin.bar(fl_bin['Model'], fl_bin['Accuracy'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl_bin.set_title('Flood Count Binary')
ax_fl_bin.bar_label(ax_fl_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl_bin.margins(y=0.2)

ax_fl_cv.bar(fl_cv['Model'], fl_cv['Accuracy'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl_cv.set(xlabel='Classifier', ylabel='Accuracy')
ax_fl_cv.set_title('Flood Count Multiclass CV')
ax_fl_cv.bar_label(ax_fl_cv.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl_cv.margins(y=0.2)

ax_fl_cv_bin.bar(fl_bin_cv['Model'], fl_bin_cv['Accuracy'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl_cv_bin.set(xlabel='Classifier')
ax_fl_cv_bin.set_title('Flood Count Binary CV')
ax_fl_cv_bin.bar_label(ax_fl_cv_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl_cv_bin.margins(y=0.2)

for ax in fig.get_axes():
    ax.label_outer()

plt.tight_layout()
plt.show()

# precision

fig, ((ax_fl, ax_fl_bin), (ax_fl_cv, ax_fl_cv_bin)) = plt.subplots(2, 2)
fig.suptitle('Flooding Count Precision')

ax_fl_bar = ax_fl.bar(fl['Model'], fl['Precision'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl.set(ylabel='Precision')
ax_fl.set_title('Flood Count Multiclass')
ax_fl.bar_label(ax_fl.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl.margins(y=0.2)

ax_fl_bin.bar(fl_bin['Model'], fl_bin['Precision'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl_bin.set_title('Flood Count Binary')
ax_fl_bin.bar_label(ax_fl_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl_bin.margins(y=0.2)

ax_fl_cv.bar(fl_cv['Model'], fl_cv['Precision'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl_cv.set(xlabel='Classifier', ylabel='Precision')
ax_fl_cv.set_title('Flood Count Multiclass CV')
ax_fl_cv.bar_label(ax_fl_cv.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl_cv.margins(y=0.2)

ax_fl_cv_bin.bar(fl_bin_cv['Model'], fl_bin_cv['Precision'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl_cv_bin.set(xlabel='Classifier')
ax_fl_cv_bin.set_title('Flood Count Binary CV')
ax_fl_cv_bin.bar_label(ax_fl_cv_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl_cv_bin.margins(y=0.2)

for ax in fig.get_axes():
    ax.label_outer()

plt.tight_layout()
plt.show()

# recall

fig, ((ax_fl, ax_fl_bin), (ax_fl_cv, ax_fl_cv_bin)) = plt.subplots(2, 2)
fig.suptitle('Flooding Count Recall')

ax_fl_bar = ax_fl.bar(fl['Model'], fl['Recall'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl.set(ylabel='Recall')
ax_fl.set_title('Flood Count Multiclass')
ax_fl.bar_label(ax_fl.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl.margins(y=0.2)

ax_fl_bin.bar(fl_bin['Model'], fl_bin['Recall'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl_bin.set_title('Flood Count Binary')
ax_fl_bin.bar_label(ax_fl_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl_bin.margins(y=0.2)

ax_fl_cv.bar(fl_cv['Model'], fl_cv['Recall'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl_cv.set(xlabel='Classifier', ylabel='Recall')
ax_fl_cv.set_title('Flood Count Multiclass CV')
ax_fl_cv.bar_label(ax_fl_cv.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl_cv.margins(y=0.2)

ax_fl_cv_bin.bar(fl_bin_cv['Model'], fl_bin_cv['Recall'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl_cv_bin.set(xlabel='Classifier')
ax_fl_cv_bin.set_title('Flood Count Binary CV')
ax_fl_cv_bin.bar_label(ax_fl_cv_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl_cv_bin.margins(y=0.2)

for ax in fig.get_axes():
    ax.label_outer()

plt.tight_layout()
plt.show()

# f1

fig, ((ax_fl, ax_fl_bin), (ax_fl_cv, ax_fl_cv_bin)) = plt.subplots(2, 2)
fig.suptitle('Flooding Count F1')

ax_fl_bar = ax_fl.bar(fl['Model'], fl['F1'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl.set(ylabel='F1')
ax_fl.set_title('Flood Count Multiclass')
ax_fl.bar_label(ax_fl.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl.margins(y=0.2)

ax_fl_bin.bar(fl_bin['Model'], fl_bin['F1'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl_bin.set_title('Flood Count Binary')
ax_fl_bin.bar_label(ax_fl_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl_bin.margins(y=0.2)

ax_fl_cv.bar(fl_cv['Model'], fl_cv['F1'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl_cv.set(xlabel='Classifier', ylabel='F1')
ax_fl_cv.set_title('Flood Count Multiclass CV')
ax_fl_cv.bar_label(ax_fl_cv.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl_cv.margins(y=0.2)

ax_fl_cv_bin.bar(fl_bin_cv['Model'], fl_bin_cv['F1'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl_cv_bin.set(xlabel='Classifier')
ax_fl_cv_bin.set_title('Flood Count Binary CV')
ax_fl_cv_bin.bar_label(ax_fl_cv_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl_cv_bin.margins(y=0.2)

for ax in fig.get_axes():
    ax.label_outer()

plt.tight_layout()
plt.show()

# pr auc

fig, ((ax_fl, ax_fl_bin), (ax_fl_cv, ax_fl_cv_bin)) = plt.subplots(2, 2)
fig.suptitle('Flooding Count PR AUC')

ax_fl_bar = ax_fl.bar(fl['Model'], fl['PR AUC'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl.set(ylabel='PR AUC')
ax_fl.set_title('Flood Count Multiclass')
ax_fl.bar_label(ax_fl.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl.margins(y=0.2)

ax_fl_bin.bar(fl_bin['Model'], fl_bin['PR AUC'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl_bin.set_title('Flood Count Binary')
ax_fl_bin.bar_label(ax_fl_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl_bin.margins(y=0.2)

ax_fl_cv.bar(fl_cv['Model'], fl_cv['PR AUC'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl_cv.set(xlabel='Classifier', ylabel='PR AUC')
ax_fl_cv.set_title('Flood Count Multiclass CV')
ax_fl_cv.bar_label(ax_fl_cv.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl_cv.margins(y=0.2)

ax_fl_cv_bin.bar(fl_bin_cv['Model'], fl_bin_cv['PR AUC'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_fl_cv_bin.set(xlabel='Classifier')
ax_fl_cv_bin.set_title('Flood Count Binary CV')
ax_fl_cv_bin.bar_label(ax_fl_cv_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_fl_cv_bin.margins(y=0.2)

for ax in fig.get_axes():
    ax.label_outer()

plt.tight_layout()
plt.show()

#%%

# flood cost range barplot

flct = scores[scores['Label'].str.contains(r'FlCt Multiclass')]
flct_bin = scores[scores['Label'].str.contains(r'FlCt Binary')]
flct_cv = scores[scores['Label'].str.contains(r'FlCt Multi CV')]
flct_bin_cv = scores[scores['Label'].str.contains(r'FlCt Bin CV')]

# accuracy

fig, ((ax_flct, ax_flct_bin), (ax_flct_cv, ax_flct_cv_bin)) = plt.subplots(2, 2)
fig.suptitle('Flooding Cost Range Accuracy')

ax_flct_bar = ax_flct.bar(flct['Model'], flct['Accuracy'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct.set(ylabel='Accuracy')
ax_flct.set_title('Flood Cost Range Multiclass')
ax_flct.bar_label(ax_flct.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct.margins(y=0.2)

ax_flct_bin.bar(flct_bin['Model'], flct_bin['Accuracy'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct_bin.set_title('Flood Cost Range Binary')
ax_flct_bin.bar_label(ax_flct_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct_bin.margins(y=0.2)

ax_flct_cv.bar(flct_cv['Model'], flct_cv['Accuracy'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct_cv.set(xlabel='Classifier', ylabel='Accuracy')
ax_flct_cv.set_title('Flood Cost Range Multiclass CV')
ax_flct_cv.bar_label(ax_flct_cv.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct_cv.margins(y=0.2)

ax_flct_cv_bin.bar(flct_bin_cv['Model'], flct_bin_cv['Accuracy'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct_cv_bin.set(xlabel='Classifier')
ax_flct_cv_bin.set_title('Flood Cost Range Binary CV')
ax_flct_cv_bin.bar_label(ax_flct_cv_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct_cv_bin.margins(y=0.2)

for ax in fig.get_axes():
    ax.label_outer()

plt.tight_layout()
plt.show()

# precision

fig, ((ax_flct, ax_flct_bin), (ax_flct_cv, ax_flct_cv_bin)) = plt.subplots(2, 2)
fig.suptitle('Flooding Cost Range Precision')

ax_flct_bar = ax_flct.bar(flct['Model'], flct['Precision'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct.set(ylabel='Precision')
ax_flct.set_title('Flood Cost Range Multiclass')
ax_flct.bar_label(ax_flct.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct.margins(y=0.2)

ax_flct_bin.bar(flct_bin['Model'], flct_bin['Precision'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct_bin.set_title('Flood Cost Range Binary')
ax_flct_bin.bar_label(ax_flct_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct_bin.margins(y=0.2)

ax_flct_cv.bar(flct_cv['Model'], flct_cv['Precision'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct_cv.set(xlabel='Classifier', ylabel='Precision')
ax_flct_cv.set_title('Flood Cost Range Multiclass CV')
ax_flct_cv.bar_label(ax_flct_cv.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct_cv.margins(y=0.2)

ax_flct_cv_bin.bar(flct_bin_cv['Model'], flct_bin_cv['Precision'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct_cv_bin.set(xlabel='Classifier')
ax_flct_cv_bin.set_title('Flood Cost Range Binary CV')
ax_flct_cv_bin.bar_label(ax_flct_cv_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct_cv_bin.margins(y=0.2)

for ax in fig.get_axes():
    ax.label_outer()

plt.tight_layout()
plt.show()

# recall

fig, ((ax_flct, ax_flct_bin), (ax_flct_cv, ax_flct_cv_bin)) = plt.subplots(2, 2)
fig.suptitle('Flooding Cost Range Recall')

ax_flct_bar = ax_flct.bar(flct['Model'], flct['Recall'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct.set(ylabel='Recall')
ax_flct.set_title('Flood Cost Range Multiclass')
ax_flct.bar_label(ax_flct.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct.margins(y=0.2)

ax_flct_bin.bar(flct_bin['Model'], flct_bin['Recall'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct_bin.set_title('Flood Cost Range Binary')
ax_flct_bin.bar_label(ax_flct_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct_bin.margins(y=0.2)

ax_flct_cv.bar(flct_cv['Model'], flct_cv['Recall'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct_cv.set(xlabel='Classifier', ylabel='Recall')
ax_flct_cv.set_title('Flood Cost Range Multiclass CV')
ax_flct_cv.bar_label(ax_flct_cv.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct_cv.margins(y=0.2)

ax_flct_cv_bin.bar(flct_bin_cv['Model'], flct_bin_cv['Recall'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct_cv_bin.set(xlabel='Classifier')
ax_flct_cv_bin.set_title('Flood Cost Range Binary CV')
ax_flct_cv_bin.bar_label(ax_flct_cv_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct_cv_bin.margins(y=0.2)

for ax in fig.get_axes():
    ax.label_outer()

plt.tight_layout()
plt.show()

# f1

fig, ((ax_flct, ax_flct_bin), (ax_flct_cv, ax_flct_cv_bin)) = plt.subplots(2, 2)
fig.suptitle('Flooding Cost Range F1')

ax_flct_bar = ax_flct.bar(flct['Model'], flct['F1'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct.set(ylabel='F1')
ax_flct.set_title('Flood Cost Range Multiclass')
ax_flct.bar_label(ax_flct.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct.margins(y=0.2)

ax_flct_bin.bar(flct_bin['Model'], flct_bin['F1'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct_bin.set_title('Flood Cost Range Binary')
ax_flct_bin.bar_label(ax_flct_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct_bin.margins(y=0.2)

ax_flct_cv.bar(flct_cv['Model'], flct_cv['F1'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct_cv.set(xlabel='Classifier', ylabel='F1')
ax_flct_cv.set_title('Flood Cost Range Multiclass CV')
ax_flct_cv.bar_label(ax_flct_cv.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct_cv.margins(y=0.2)

ax_flct_cv_bin.bar(flct_bin_cv['Model'], flct_bin_cv['F1'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct_cv_bin.set(xlabel='Classifier')
ax_flct_cv_bin.set_title('Flood Cost Range Binary CV')
ax_flct_cv_bin.bar_label(ax_flct_cv_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct_cv_bin.margins(y=0.2)

for ax in fig.get_axes():
    ax.label_outer()

plt.tight_layout()
plt.show()

# PR AUC

fig, ((ax_flct, ax_flct_bin), (ax_flct_cv, ax_flct_cv_bin)) = plt.subplots(2, 2)
fig.suptitle('Flooding Cost Range PR AUC')

ax_flct_bar = ax_flct.bar(flct['Model'], flct['PR AUC'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct.set(ylabel='PR AUC')
ax_flct.set_title('Flood Cost Range Multiclass')
ax_flct.bar_label(ax_flct.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct.margins(y=0.2)

ax_flct_bin.bar(flct_bin['Model'], flct_bin['PR AUC'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct_bin.set_title('Flood Cost Range Binary')
ax_flct_bin.bar_label(ax_flct_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct_bin.margins(y=0.2)

ax_flct_cv.bar(flct_cv['Model'], flct_cv['PR AUC'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct_cv.set(xlabel='Classifier', ylabel='PR AUC')
ax_flct_cv.set_title('Flood Cost Range Multiclass CV')
ax_flct_cv.bar_label(ax_flct_cv.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct_cv.margins(y=0.2)

ax_flct_cv_bin.bar(flct_bin_cv['Model'], flct_bin_cv['PR AUC'], color=['blue', 'royalblue', 'lightblue', 'dodgerblue', 'lightsteelblue', 'skyblue'])
ax_flct_cv_bin.set(xlabel='Classifier')
ax_flct_cv_bin.set_title('Flood Cost Range Binary CV')
ax_flct_cv_bin.bar_label(ax_flct_cv_bin.containers[0], fmt='%.2f', fontsize=9, label_type = 'edge')
ax_flct_cv_bin.margins(y=0.2)

for ax in fig.get_axes():
    ax.label_outer()

plt.tight_layout()
plt.show()

#%%

# GridSearchCV

from sklearn.model_selection import GridSearchCV

knn_param = {'n_neighbors':[3,5,11,19], 'weights':['uniform', 'distance']}
log_param = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
dt_param = {"criterion":['gini', 'entropy'], "max_depth":range(1,10)}
rf_param = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [5,10,50],
    'criterion' :['gini', 'entropy']
}
svc_param = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

# flood count

knn_gs_fl = GridSearchCV(knn_cl, knn_param, scoring='f1_weighted')
knn_gs_fl.fit(X, y_fl)

log_gs_fl = GridSearchCV(log_cl, log_param, scoring='f1_weighted')
log_gs_fl.fit(X, y_fl)

dt_gs_fl = GridSearchCV(dt_cl, dt_param, scoring='f1_weighted')
dt_gs_fl.fit(X, y_fl)

rf_gs_fl = GridSearchCV(rf_cl, rf_param, scoring='f1_weighted')
rf_gs_fl.fit(X, y_fl)

svc_gs_fl = GridSearchCV(svc_cl, svc_param, scoring='f1_weighted')
svc_gs_fl.fit(X, y_fl)

# flood damage cost

knn_gs_flct = GridSearchCV(knn_cl, knn_param, scoring='f1_weighted')
knn_gs_flct.fit(X, y_flct)

log_gs_flct = GridSearchCV(log_cl, log_param, scoring='f1_weighted')
log_gs_flct.fit(X, y_flct)

dt_gs_flct = GridSearchCV(dt_cl, dt_param, scoring='f1_weighted')
dt_gs_flct.fit(X, y_flct)

rf_gs_flct = GridSearchCV(rf_cl, rf_param, scoring='f1_weighted')
rf_gs_flct.fit(X, y_flct)

svc_gs_flct = GridSearchCV(svc_cl, svc_param, scoring='f1_weighted')
svc_gs_flct.fit(X, y_flct)
#%%
# flood count binary

knn_gs_fl_bin = GridSearchCV(knn_cl, knn_param, scoring='f1_weighted')
knn_gs_fl_bin.fit(X, y_fl_bin)

dt_gs_fl_bin = GridSearchCV(dt_cl, dt_param, scoring='f1_weighted')
dt_gs_fl_bin.fit(X, y_fl_bin)

rf_gs_fl_bin = GridSearchCV(rf_cl, rf_param, scoring='f1_weighted')
rf_gs_fl_bin.fit(X, y_fl_bin)

# flood damage cost binary

knn_gs_flct_bin = GridSearchCV(knn_cl, knn_param, scoring='f1_weighted')
knn_gs_flct_bin.fit(X, y_flct_bin)

dt_gs_flct_bin = GridSearchCV(dt_cl, dt_param, scoring='f1_weighted')
dt_gs_flct_bin.fit(X, y_flct_bin)

rf_gs_flct_bin = GridSearchCV(rf_cl, rf_param, scoring='f1_weighted')
rf_gs_flct_bin.fit(X, y_flct)


#%%

# 20-fold cross validate

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=20, random_state=rdm, shuffle=True)

y_fl = pd.DataFrame(y_fl)
X_train_base=[]
X_test_base=[]
y_train_base=[]
y_test_base=[]

# flood count

for train_index, test_index in skf.split(X, y_fl):
     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
     y_train, y_test = y_fl.iloc[train_index], y_fl.iloc[test_index]
     X_train_base.append(X_train)
     X_test_base.append(X_test)
     y_train_base.append(y_train)
     y_test_base.append(y_test)

knn_cl = KNeighborsClassifier(**knn_gs_fl.best_params_)
knn_scores = pd.DataFrame(columns=['F', 'KNN F1', 'KNN Acc'])

for fold in range(len(X_train_base)):
    y_train = y_train_base[fold]
    y_train = y_train.to_numpy()
    y_test = y_test_base[fold]
    y_test = y_test.to_numpy()   
    knn_cl.fit(X_train_base[fold], y_train)
    y_pred = knn_cl.predict(X_test_base[fold])
    row = [str(fold), f1_score(y_test, y_pred, average='weighted'), balanced_accuracy_score(y_test, y_pred)]
    knn_scores.loc[len(knn_scores)] = row
    
log_cl = LogisticRegression(**log_gs_fl.best_params_)
log_scores = pd.DataFrame(columns=['F', 'LR F1', 'LR Acc'])

for fold in range(len(X_train_base)):
    y_train = y_train_base[fold]
    y_train = y_train.to_numpy()
    y_test = y_test_base[fold]
    y_test = y_test.to_numpy()   
    log_cl.fit(X_train_base[fold], y_train)
    y_pred = log_cl.predict(X_test_base[fold])
    row = [str(fold), f1_score(y_test, y_pred, average='weighted'), balanced_accuracy_score(y_test, y_pred)]
    log_scores.loc[len(log_scores)] = row
    
dt_cl = DecisionTreeClassifier(**dt_gs_fl.best_params_)
dt_scores = pd.DataFrame(columns=['F', 'DT F1', 'DT Acc'])

for fold in range(len(X_train_base)):
    y_train = y_train_base[fold]
    y_train = y_train.to_numpy()
    y_test = y_test_base[fold]
    y_test = y_test.to_numpy()   
    dt_cl.fit(X_train_base[fold], y_train)
    y_pred = dt_cl.predict(X_test_base[fold])
    row = [str(fold), f1_score(y_test, y_pred, average='weighted'), balanced_accuracy_score(y_test, y_pred)]
    dt_scores.loc[len(dt_scores)] = row
    
rf_cl = RandomForestClassifier(**rf_gs_fl.best_params_)
rf_scores = pd.DataFrame(columns=['F', 'RF F1', 'RF Acc'])

for fold in range(len(X_train_base)):
    y_train = y_train_base[fold]
    y_train = y_train.to_numpy()
    y_test = y_test_base[fold]
    y_test = y_test.to_numpy()   
    rf_cl.fit(X_train_base[fold], y_train)
    y_pred = rf_cl.predict(X_test_base[fold])
    row = [str(fold), f1_score(y_test, y_pred, average='weighted'), balanced_accuracy_score(y_test, y_pred)]
    rf_scores.loc[len(rf_scores)] = row
    
svc_cl = SVC(**svc_gs_fl.best_params_)
svc_scores = pd.DataFrame(columns=['F', 'SVC F1', 'SVC Acc'])

for fold in range(len(X_train_base)):
    y_train = y_train_base[fold]
    y_train = y_train.to_numpy()
    y_test = y_test_base[fold]
    y_test = y_test.to_numpy()   
    svc_cl.fit(X_train_base[fold], y_train)
    y_pred = svc_cl.predict(X_test_base[fold])
    row = [str(fold), f1_score(y_test, y_pred, average='weighted'), balanced_accuracy_score(y_test, y_pred)]
    svc_scores.loc[len(svc_scores)] = row

fl_gs_scores = pd.concat([pd.DataFrame(knn_scores),pd.DataFrame(log_scores[['LR F1', 'LR Acc']],), pd.DataFrame(dt_scores[['DT F1', 'DT Acc']]), pd.DataFrame(rf_scores[['RF F1', 'RF Acc']],), pd.DataFrame(svc_scores[['SVC F1', 'SVC Acc']])],axis=1)

# flood damage cost

y_flct = pd.DataFrame(y_flct)
X_train_base=[]
X_test_base=[]
y_train_base=[]
y_test_base=[]

for train_index, test_index in skf.split(X, y_flct):
     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
     y_train, y_test = y_flct.iloc[train_index], y_flct.iloc[test_index]
     X_train_base.append(X_train)
     X_test_base.append(X_test)
     y_train_base.append(y_train)
     y_test_base.append(y_test)

knn_cl = KNeighborsClassifier(**knn_gs_flct.best_params_)
knn_scores = pd.DataFrame(columns=['F', 'KNN F1', 'KNN Acc'])

for fold in range(len(X_train_base)):
    y_train = y_train_base[fold]
    y_train = y_train.to_numpy()
    y_test = y_test_base[fold]
    y_test = y_test.to_numpy()   
    knn_cl.fit(X_train_base[fold], y_train)
    y_pred = knn_cl.predict(X_test_base[fold])
    row = [str(fold), f1_score(y_test, y_pred, average='weighted'), balanced_accuracy_score(y_test, y_pred)]
    knn_scores.loc[len(knn_scores)] = row
    
log_cl = LogisticRegression(**log_gs_flct.best_params_)
log_scores = pd.DataFrame(columns=['F', 'LR F1', 'LR Acc'])

for fold in range(len(X_train_base)):
    y_train = y_train_base[fold]
    y_train = y_train.to_numpy()
    y_test = y_test_base[fold]
    y_test = y_test.to_numpy()   
    log_cl.fit(X_train_base[fold], y_train)
    y_pred = log_cl.predict(X_test_base[fold])
    row = [str(fold), f1_score(y_test, y_pred, average='weighted'), balanced_accuracy_score(y_test, y_pred)]
    log_scores.loc[len(log_scores)] = row
    
dt_cl = DecisionTreeClassifier(**dt_gs_flct.best_params_)
dt_scores = pd.DataFrame(columns=['F', 'DT F1', 'DT Acc'])

for fold in range(len(X_train_base)):
    y_train = y_train_base[fold]
    y_train = y_train.to_numpy()
    y_test = y_test_base[fold]
    y_test = y_test.to_numpy()   
    dt_cl.fit(X_train_base[fold], y_train)
    y_pred = dt_cl.predict(X_test_base[fold])
    row = [str(fold), f1_score(y_test, y_pred, average='weighted'), balanced_accuracy_score(y_test, y_pred)]
    dt_scores.loc[len(dt_scores)] = row
    
rf_cl = RandomForestClassifier(**rf_gs_flct.best_params_)
rf_scores = pd.DataFrame(columns=['F', 'RF F1', 'RF Acc'])

for fold in range(len(X_train_base)):
    y_train = y_train_base[fold]
    y_train = y_train.to_numpy()
    y_test = y_test_base[fold]
    y_test = y_test.to_numpy()   
    rf_cl.fit(X_train_base[fold], y_train)
    y_pred = rf_cl.predict(X_test_base[fold])
    row = [str(fold), f1_score(y_test, y_pred, average='weighted'), balanced_accuracy_score(y_test, y_pred)]
    rf_scores.loc[len(rf_scores)] = row
    
svc_cl = SVC(**svc_gs_flct.best_params_)
svc_scores = pd.DataFrame(columns=['F', 'SVC F1', 'SVC Acc'])

for fold in range(len(X_train_base)):
    y_train = y_train_base[fold]
    y_train = y_train.to_numpy()
    y_test = y_test_base[fold]
    y_test = y_test.to_numpy()   
    svc_cl.fit(X_train_base[fold], y_train)
    y_pred = svc_cl.predict(X_test_base[fold])
    row = [str(fold), f1_score(y_test, y_pred, average='weighted'), balanced_accuracy_score(y_test, y_pred)]
    svc_scores.loc[len(svc_scores)] = row

flct_gs_scores = pd.concat([pd.DataFrame(knn_scores),pd.DataFrame(log_scores[['LR F1', 'LR Acc']],), pd.DataFrame(dt_scores[['DT F1', 'DT Acc']]), pd.DataFrame(rf_scores[['RF F1', 'RF Acc']],), pd.DataFrame(svc_scores[['SVC F1', 'SVC Acc']])],axis=1)

#%%

# multiple line plots

plt.plot( 'F', 'KNN F1', data=fl_gs_scores, marker='', color='skyblue', linewidth=2, label="KNN")
plt.plot( 'F', 'LR F1', data=fl_gs_scores, marker='', color='royalblue', linewidth=2, label='LR')
plt.plot( 'F', 'DT F1', data=fl_gs_scores, marker='', color='lightblue', linewidth=2, label="DT")
plt.plot( 'F', 'RF F1', data=fl_gs_scores, marker='', color='dodgerblue', linewidth=2, label="RF")
plt.plot( 'F', 'SVC F1', data=fl_gs_scores, marker='', color='lightsteelblue', linewidth=2, label="SVC")
plt.legend()
plt.show()
#%%
plt.plot( 'F', 'KNN F1', data=flct_gs_scores, marker='', color='skyblue', linewidth=2, label="KNN")
plt.plot( 'F', 'LR F1', data=flct_gs_scores, marker='', color='royalblue', linewidth=2, label='LR')
plt.plot( 'F', 'DT F1', data=flct_gs_scores, marker='', color='lightblue', linewidth=2, label="DT")
plt.plot( 'F', 'RF F1', data=flct_gs_scores, marker='', color='dodgerblue', linewidth=2, label="RF")
plt.plot( 'F', 'SVC F1', data=flct_gs_scores, marker='', color='lightsteelblue', linewidth=2, label="SVC")

plt.legend()
plt.show()