import os
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\user\\Documents\\Python\\Kaggle Competition\\Deployment\\NTT practise")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
Submission = pd.read_csv("gender_submission.csv")

train.isnull().sum()

df_train = pd.DataFrame(train.isnull().sum())
df_train.columns = ['Nas']
df_train['Allowed_limit'] = round(train.shape[0]*0.4,2)
df_train['Status'] = np.where(df_train['Nas'] > df_train['Allowed_limit'],'Remove','Retain')

train.drop(['Cabin'],axis =1, inplace =True)
train.drop(['Name','Ticket'],axis =1, inplace =True)
train['Fare'] = round(train['Fare'],2)

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x = 'Pclass',hue= 'Survived',data = train)

sns.boxplot(x= 'Pclass', y = 'Age',data = train)

train['Age'].fillna(train.groupby('Pclass')['Age'].transform('mean'),inplace =True)
train['Age'] = round(train['Age'],2)

train.isnull().sum()
train['Embarked'].dtypes == 'object'

mode = train['Embarked'].mode()
train['Embarked'].fillna('S',inplace =True)

#--------------------------------------------------------------------------------------------------

test.isnull().sum()

test['Age'] = test['Age'].fillna(test.groupby('Pclass')['Age'].transform('mean'))
mean = test['Fare'].mean()
test['Fare'].fillna(mean, inplace =True)
test.drop(['Cabin','Name','Ticket'],axis =1,inplace =True)

train.dtypes


train_cat = (train.dtypes == 'object')
train_dummy = pd.get_dummies(train.loc[:,train_cat],drop_first =True)

train2 = pd.concat([train.loc[:,~train_cat],train_dummy],axis =1)

test_cat = (test.dtypes == 'object')
test_dummy = pd.get_dummies(test.loc[:,test_cat],drop_first =True)

test2 = pd.concat([test.loc[:,~test_cat],test_dummy],axis =1)
 
Train_X = train2.drop(['Survived'],axis =1)
Train_Y = train2['Survived']
Test_X = test2.copy()
Test_Y = Submission['Survived'].copy()

from sklearn.linear_model import LogisticRegression

M1 = LogisticRegression(random_state=123).fit(Train_X,Train_Y)

Test_Pred = M1.predict(Test_X)

from sklearn.metrics import confusion_matrix

Con_Mat= confusion_matrix(Test_Pred,Test_Y)
sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

import pickle

pickle.dump(M1,open('model.pkl','wb'))

Submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': Test_Pred})
filename = 'Titanic_prediction.csv'
Submission.to_csv(filename,index = False)


