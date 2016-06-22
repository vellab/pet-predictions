# -*- coding: utf-8 -*-
"""
Created on Sat May 28 23:05:32 2016

@author: bharath
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.feature_selection import RFE
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from dateutil.parser import parse
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import re

forcolor=preprocessing.LabelEncoder()
forday=preprocessing.LabelEncoder()
def mungeData(dataframe,training):
    def agetoweeks(agestr):
        [num,numtype]=agestr.split()
        if(numtype=='year' or numtype=='years'):
            return int(num)*365
        elif(numtype=='month' or numtype=='months'):
            return int(num)*30
        elif(numtype=='week' or numtype=='weeks'):
            return int(num)*7
        else:
            return int(num)
    def isintact(sex):
        if(re.search('Intact',sex)):
            return 1
        else:
            return 0
    def morf(sex):
        if(re.search('Female',sex)):
            return 1
        elif(re.search('Unknown',sex)):
            return 0
        else:
            return 2
    def firstel(l):
        return l[0]
    def baby(age):
        if(age<365):
            return 1
        else:
            return 0
    def weekday(date):
        temp=parse(date)
        weekday=temp.weekday()
        return weekday
    def timeoftheday(date):
        temp=parse(date)
        hour=temp.timetuple().tm_hour
        if(hour<8 or hour>18):
            return 0
        else:
            return 1
    #Fill missing data for SexuponOutcome,AgeuponOutcome
    dataframe['SexuponOutcome']=dataframe['SexuponOutcome'].fillna('Neutered Male')
    dataframe['AgeuponOutcome']=dataframe['AgeuponOutcome'].fillna('1 year')
    dataframe['Name']=dataframe['Name'].fillna('Nameless')
    #Convert string values into numerical parameters
    dataframe['AnimalType']=dataframe['AnimalType'].map({'Cat':0,'Dog':1})
    dataframe['weekage']=dataframe['AgeuponOutcome'].apply(agetoweeks)
    dataframe['gender']=dataframe['SexuponOutcome'].apply(morf)
    dataframe['isintact']=dataframe['SexuponOutcome'].apply(isintact)
    dataframe['simplecolor']=dataframe['Color'].str.split('/').apply(firstel)
    dataframe['simplecolor']=forcolor.fit_transform(dataframe['simplecolor'])
    dataframe['isamix']=dataframe['Breed'].str.contains('mix',case=False).astype(int)
    dataframe['simplebreed']=dataframe['Breed'].str.split('/').apply(firstel)
    dataframe['simplebreed']=forcolor.fit_transform(dataframe['simplebreed'])
    dataframe['baby']=dataframe['weekage'].apply(baby)
    dataframe['hasname']=dataframe['Name'].str.contains('Nameless').astype(int)
    dataframe['weekday']=dataframe['DateTime'].apply(weekday)
    dataframe['weekday']=forday.fit_transform(dataframe['weekday'])
    dataframe['hour']=dataframe['DateTime'].apply(timeoftheday)
    dataframe['deathold']=dataframe['hour']+dataframe['baby']
    
    if(training):
        #dataframe['outcome']=preprocessing.LabelEncoder().fit_transform(dataframe['OutcomeType'])
        dataframe['outcome']=dataframe['OutcomeType'].map({'Return_to_owner':3, 'Euthanasia':2, 'Adoption':0, 'Transfer':4, 'Died':1})
    
    return dataframe

def param_importance(data):
    alg=RandomForestClassifier()
    param_grid={
        'n_estimators':[50,400],
        'max_features':['auto','sqrt','log2']    
    }
    CV_alg=GridSearchCV(estimator=alg,param_grid=param_grid,cv=5)
    CV_alg.fit(data[predictors],data['outcome'])
    return CV_alg.best_params_
    
predictors=['AnimalType','weekage','gender','isintact','hasname','baby','hour','weekday']
#Obtain Dataframe object from csv file
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
munged_train=mungeData(train_data,1)
munged_test=mungeData(test_data,0)

#print(param_importance(munged_train))
dt=DecisionTreeClassifier()
#alg=RandomForestClassifier(n_estimators=400,max_features='auto')
#alg=GradientBoostingClassifier(min_samples_leaf=3,n_estimators=400,min_samples_split=3)
#alg=KNeighborsClassifier(n_neighbors=8)
#alg=AdaBoostClassifier(n_estimators=400,base_estimator=dt,learning_rate=0.5)
#results = smf.ols('outcome ~ weekage + isintact +gender + AnimalType+baby+ hasname +isamix+simplebreed+simplecolor+hour+deathold+weekday', data=munged_train).fit()
#print(results.summary())
#rfe=RFE(alg,5)
#rfe=rfe.fit(munged_train[predictors],munged_train['outcome'])
#print(rfe.support_)
#print(rfe.ranking_)
print(cross_validation.cross_val_score(alg,munged_train[predictors],munged_train['outcome'],cv=8))
#alg.fit(munged_train[predictors],munged_train['outcome'])
##
#predictions=alg.predict_proba(munged_test[predictors])
#output = pd.DataFrame(predictions,columns=['Adoption','Died','Euthanasia','Return_to_owner','Transfer'])
#output.columns.names=['ID']
#output.index.names=['ID']
#output.index+=1
#output.index.names=['ID']
#output.to_csv('predictions.csv')

#data_num=dataframe['OutcomeType'].shape[0]
#train_predictors=munged_train[predictors]
#train_target=list(munged_train['OutcomeType'].values)
#test_predictors=(munged_test[predictors])
#test_predictions=alg.predict(dataframe[predictors].iloc[int(math.floor(0.95*data_num)):,:])
#
#print(test_predictions)
