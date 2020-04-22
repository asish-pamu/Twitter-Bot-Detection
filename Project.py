# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:10:07 2019

@author: Ashish
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm

genuine_accounts = pd.read_csv(r'C:\Users\Ashish\Desktop\datasets_full.csv\genuine_accounts.csv\users.csv')
traditional_spambots_1 = pd.read_csv(r'C:\Users\Ashish\Desktop\datasets_full.csv\traditional_spambots_1.csv\users.csv')
traditional_spambots_2 = pd.read_csv(r'C:\Users\Ashish\Desktop\datasets_full.csv\traditional_spambots_2.csv\users.csv')
traditional_spambots_3 = pd.read_csv(r'C:\Users\Ashish\Desktop\datasets_full.csv\traditional_spambots_3.csv\users.csv')
traditional_spambots_4 = pd.read_csv(r'C:\Users\Ashish\Desktop\datasets_full.csv\traditional_spambots_4.csv\users.csv')
social_spambots_1 = pd.read_csv(r'C:\Users\Ashish\Desktop\datasets_full.csv\social_spambots_1.csv\users.csv')
social_spambots_2 = pd.read_csv(r'C:\Users\Ashish\Desktop\datasets_full.csv\social_spambots_2.csv\users.csv')
social_spambots_3 = pd.read_csv(r'C:\Users\Ashish\Desktop\datasets_full.csv\social_spambots_3.csv\users.csv')


genuine_accounts.drop(['test_set_1','test_set_2'], axis = 1 , inplace = True)

traditional_spambots = pd.concat([traditional_spambots_1,traditional_spambots_2,
                  traditional_spambots_3,traditional_spambots_4], axis = 0,ignore_index = True,sort = False)

social_spambots = pd.concat([social_spambots_1,social_spambots_2,
                             social_spambots_3],axis = 0, ignore_index = True,sort = False)

genuine_accounts['bot'] = 0
traditional_spambots['bot'] = 1
social_spambots['bot'] = 1

columns_to_extract = ['id','screen_name','location','url','followers_count','friends_count','listed_count',
                    'favourites_count','created_at','verified','statuses_count','lang','default_profile',
                    'name', 'description','bot']

users = genuine_accounts[columns_to_extract]
bots = pd.concat([social_spambots,traditional_spambots], axis = 0 , ignore_index = True, sort = False)
bots = bots[columns_to_extract]

plt.figure(figsize=(10,4))
plt.subplot(2, 1, 1)
plt.scatter(bots.followers_count, bots.friends_count, color='red')
plt.xlabel("Bots Followers")
plt.ylabel("Bots Following")
plt.xlim(0, 500)
plt.ylim(0, 500)
plt.title("Bots Followers vs Following count")

plt.subplot(2, 1, 2)
plt.scatter(users.followers_count, users.friends_count, color='green')
plt.xlabel("Users Followers")
plt.ylabel("Users Following")
plt.xlim(0, 500)
plt.ylim(0, 500)
plt.title("Users Followers vs Following count")

plt.tight_layout()
plt.show()
plt.savefig(r'C:\Users\Ashish\Desktop\datasets_full.csv\Followers vs Following count')

plt.figure(figsize=(10,4))
plt.subplot(2,1,1)
plt.title('Bots Friends vs Followers')
plt.scatter(bots.friends_count, bots.followers_count, color='red', label='Bots')
plt.xlim(0, 500)
plt.ylim(0, 500)
plt.tight_layout()

plt.subplot(2,1,2)
plt.title('users Friends vs Followers')
plt.scatter(users.friends_count, users.followers_count, color='blue', label='Users')
plt.xlim(0, 500)
plt.ylim(0, 500)

plt.tight_layout()
plt.show()
plt.savefig(r'C:\Users\Ashish\Desktop\datasets_full.csv\Friends vs Followers')


bots['friends_by_followers'] = bots.friends_count/bots.followers_count
users['friends_by_followers'] = users.friends_count/users.followers_count

plt.figure(figsize=(10,4))
plt.plot(bots.listed_count, color='red', label='Bots')
plt.plot(users.listed_count, color='blue', label='Users')
plt.legend(loc='upper left')
plt.ylim(0,200)

bots['location_binary'] = bots.location.isnull()==True
bots['verified_binary'] = bots.verified==False
bots['listed_count_binary'] = (bots.listed_count>3000)==False


users['location_binary'] = users.location.isnull()==False
users['verified_binary'] = users.verified==True
users['listed_count_binary'] = (users.listed_count>3000)==False


features = ['verified_binary','location_binary','followers_count', 'friends_count', 'statuses_count', 'listed_count_binary', 'bot']

final_dataset = pd.concat([bots,users], axis = 0 , ignore_index = True,sort = False)
heat_map = final_dataset.drop(['verified','default_profile'],axis = 1)

X = final_dataset[features].iloc[:,:-1]
y = final_dataset[features].iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

## Decision Tree


dt = DecisionTreeClassifier(criterion='gini', min_samples_leaf=100, min_samples_split=20)
dt = dt.fit(X_train, y_train)
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)

print("Trainig Accuracy: %.5f" %accuracy_score(y_train, y_pred_train))
print("Test Accuracy: %.5f" %accuracy_score(y_test, y_pred_test))

scores_train = dt.predict_proba(X_train)
scores_test = dt.predict_proba(X_test)

y_scores_train = []
y_scores_test = []
for i in range(len(scores_train)):
    y_scores_train.append(scores_train[i][1])

for i in range(len(scores_test)):
    y_scores_test.append(scores_test[i][1])
    
fpr_train, tpr_train, _ = roc_curve(y_train, y_scores_train, pos_label=1)
fpr_test, tpr_test, _ = roc_curve(y_test, y_scores_test, pos_label=1)

plt.figure(figsize=(10,4))
plt.plot(fpr_train, tpr_train, color='darkblue', label='Train AUC: %5f' %auc(fpr_train, tpr_train))
plt.plot(fpr_test, tpr_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_test, tpr_test))
plt.title("Decision Tree ROC Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc='lower right')
plt.show()
plt.savefig(r'C:\Users\Ashish\Desktop\datasets_full.csv\decision_tree_3.png')


## Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_test = logreg.predict(X_test)
y_pred_train = logreg.predict(X_train)

print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(logreg.score(X_test, y_test)))

fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train, pos_label=1)
fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test, pos_label=1)
plt.figure(figsize=(10,4))
plt.plot(fpr_train, tpr_train, color='darkblue', label='Train AUC: %5f' %auc(fpr_train, tpr_train))
plt.plot(fpr_test, tpr_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_test, tpr_test))
plt.title("Logistic Regression ROC Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc='lower right')
plt.show()
plt.savefig(r'C:\Users\Ashish\Desktop\datasets_full.csv\logistic.png')


## SVM

#clf = svm.SVC()
clf = svm.SVC(gamma=0.0001, C=1)
clf.fit(X_train,y_train)
y_pred_test = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

print('Accuracy of logistic regression classifier on train set: {:.5f}'.format(clf.score(X_train, y_train)))
print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(clf.score(X_test, y_test)))

fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train, pos_label=1)
fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test, pos_label=1)
plt.figure(figsize=(10,4))
plt.plot(fpr_train, tpr_train, color='darkblue', label='Train AUC: %5f' %auc(fpr_train, tpr_train))
plt.plot(fpr_test, tpr_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_test, tpr_test))
plt.title("SVM ROC Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc='lower right')
plt.show()
plt.savefig(r'C:\Users\Ashish\Desktop\datasets_full.csv\svm_5.png') 


import seaborn as sns
plt.figure(figsize=(10,4))
sns.heatmap(heat_map.corr(method='spearman'), cmap='coolwarm', annot=True)
plt.tight_layout()
plt.show()
plt.savefig(r'C:\Users\Ashish\Desktop\datasets_full.csv\heatmap.png')
