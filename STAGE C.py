#!/usr/bin/env python
# coding: utf-8

# ## STAGE C QUIZ

# In[ ]:


#importing necessary libraries
import numpy as np
import pandas as pd


# ### Importing the dataset

# In[2]:


df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv')


# ### Analyzing the dataset

# In[3]:


df.head()


# In[4]:


df.isna().sum()


# In[5]:


df=df.drop('stab',axis=1)


# ### Splitting the dataset into features and response datasets

# In[6]:


x=df.drop('stabf',axis=1)
y=df['stabf']


# ### Splitting the datasets into train and test dataset for the model

# In[7]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# ### Scaling the train and test datasets (x)

# In[8]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scale=scaler.fit_transform(x_train)
x_test_scale=scaler.transform(x_test)


# In[9]:


#Converting the scaled datasets back to Dataframes
x_train_df=pd.DataFrame(x_train_scale,columns=x_train.columns)
x_test_df=pd.DataFrame(x_test_scale,columns=x_test.columns)


# ### Training a Random Forest Classifier

# In[10]:


from sklearn.ensemble import RandomForestClassifier
RF_Classifier=RandomForestClassifier(random_state=1)
RF_Classifier.fit(x_train_df,y_train)


# In[11]:


#predicting values using RandomForest
RF_Classifier_Prediction=RF_Classifier.predict(x_test_df)


# ### Classification Report 

# In[12]:


from sklearn.metrics import classification_report
print(classification_report(y_test,RF_Classifier_Prediction,digits=4))


# In[13]:


#importing necessary libraries to check model performance
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, confusion_matrix


# In[19]:


accuracy = accuracy_score(y_test,RF_Classifier_Prediction)
print('Accuracy: {}'.format(round(accuracy*100), 4))


# In[20]:


precision=precision_score(y_test,RF_Classifier_Prediction,pos_label='stable')
print('Precision: {}'.format(round(precision*100),4))


# In[21]:


recall=recall_score(y_test,RF_Classifier_Prediction,pos_label='stable')
print('Recall: {}'.format(round(recall*100),4))


# In[22]:


f1=f1_score(y_test,RF_Classifier_Prediction,pos_label='stable')
print('f1_score: {}'.format(round(f1*100),2))


# In[23]:


#confusion matrix
RF_cnf_mat=confusion_matrix(y_test,RF_Classifier_Prediction,labels=['unstable','stable'])
print('Confusion Matrix\n',RF_cnf_mat)


# ### Training an Extra Tree Classifier

# In[24]:


#importing the necessary libraries
from sklearn.ensemble import ExtraTreesClassifier
ET_Classifier=ExtraTreesClassifier(random_state=1)
ET_Classifier.fit(x_train_df,y_train)


# In[25]:


#Predicting values using Extra Trees
ET_Classifier_Prediction=ET_Classifier.predict(x_test_df)


# ### Classification Report

# In[26]:


from sklearn.metrics import classification_report
print(classification_report(y_test,ET_Classifier_Prediction,digits=4))


# In[27]:


#importing necessary libraries to check model performance
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, confusion_matrix


# In[28]:


accuracy = accuracy_score(y_test,ET_Classifier_Prediction)
print('Accuracy: {}'.format(round(accuracy*100), 4))

precision=precision_score(y_test,ET_Classifier_Prediction,pos_label='stable')
print('Precision: {}'.format(round(precision*100),4))

recall=recall_score(y_test,ET_Classifier_Prediction,pos_label='stable')
print('Recall: {}'.format(round(recall*100),4))

f1=f1_score(y_test,ET_Classifier_Prediction,pos_label='stable')
print('f1_score: {}'.format(round(f1*100),2))


# In[29]:


#confusion matrix
ET_cnf_mat=confusion_matrix(y_test,ET_Classifier_Prediction,labels=['unstable','stable'])
print('Confusion Matrix\n',ET_cnf_mat)


# ### Training an Extreme Boosting Model

# In[30]:


from xgboost import XGBClassifier
XGB= XGBClassifier(random_state=1)
XGB.fit(x_train_df, y_train)


# In[31]:


#predicting values using xgboost
XGB_Prediction=XGB.predict(x_test_df)


# ### Classification Report

# In[32]:


from sklearn.metrics import classification_report
print(classification_report(y_test,XGB_Prediction,digits=4))


# In[33]:


#importing necessary libraries to check model performance
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, confusion_matrix


# In[34]:


accuracy = accuracy_score(y_test,XGB_Prediction)
print('Accuracy: {}'.format(round(accuracy*100), 4))

precision=precision_score(y_test,XGB_Prediction,pos_label='stable')
print('Precision: {}'.format(round(precision*100),4))

recall=recall_score(y_test,XGB_Prediction,pos_label='stable')
print('Recall: {}'.format(round(recall*100),4))

f1=f1_score(y_test,XGB_Prediction,pos_label='stable')
print('f1_score: {}'.format(round(f1*100),2))


# In[35]:


#confusion matrix
XGB_cnf_mat=confusion_matrix(y_test,XGB_Prediction,labels=['unstable','stable'])
print('Confusion Matrix\n',XGB_cnf_mat)


# ### Training a Light Gradient Boosting Model

# In[36]:


from lightgbm import LGBMClassifier
lgbm_classifier=LGBMClassifier(random_state=1)
lgbm_classifier.fit(x_train_df,y_train)


# In[37]:


lgbm_classifier_prediction=lgbm_classifier.predict(x_test_df)


# ### Classification Report

# In[38]:


from sklearn.metrics import classification_report
print(classification_report(y_test,lgbm_classifier_prediction,digits=4))


# In[39]:


#Model Performance
accuracy = accuracy_score(y_test,lgbm_classifier_prediction)
print('Accuracy: {}'.format(round(accuracy*100), 4))

precision=precision_score(y_test,lgbm_classifier_prediction,pos_label='stable')
print('Precision: {}'.format(round(precision*100),4))

recall=recall_score(y_test,lgbm_classifier_prediction,pos_label='stable')
print('Recall: {}'.format(round(recall*100),4))

f1=f1_score(y_test,lgbm_classifier_prediction,pos_label='stable')
print('f1_score: {}'.format(round(f1*100),2))


# In[40]:


#confusion matrix
lgbm_cnf_mat=confusion_matrix(y_test,lgbm_classifier_prediction,labels=['unstable','stable'])
print('Confusion Matrix\n',lgbm_cnf_mat)


# ### Improving Extra Trees Classifier

# In[41]:


n_estimators = [50, 100, 300, 500, 1000]

min_samples_split = [2, 3, 5, 7, 9]

min_samples_leaf = [1, 2, 4, 6, 8]

max_features = ['auto', 'sqrt', 'log2', None]

hyperparameter_grid = {'n_estimators': n_estimators,
        
                      'min_samples_leaf': min_samples_leaf,

                      'min_samples_split': min_samples_split,

                       'max_features': max_features}


# In[42]:


from sklearn.model_selection import RandomizedSearchCV
random_cv = RandomizedSearchCV(estimator = ET_Classifier, 
                              param_distributions = hyperparameter_grid, cv=5, n_iter=10, scoring = 'accuracy', n_jobs = -1, verbose = 1,
                              random_state = 1)


# In[43]:


search_ = random_cv.fit(x_train_df, y_train)


# In[44]:


#getting best parameters
search_.best_params_


# In[45]:


importance = ET_Classifier.feature_importances_


# In[47]:


#feature importance
for x,y in enumerate(importance):
     print('Feature: %0d, Score: %.4f' % (x,y))


# In[48]:


search_.best_score_


# In[49]:


#using the best parameters
ET_Classifier_1 = ExtraTreesClassifier(max_features = None, 
                            min_samples_leaf= 8,
                            min_samples_split= 2,
                            n_estimators= 1000, 
                            random_state = 1)


ET_Classifier_1.fit(x_train_df, y_train)


# In[50]:


ET_Classifier_1_Prediction=ET_Classifier_1.predict(x_test_df)


# In[51]:


print(classification_report(y_test,ET_Classifier_1_Prediction, digits =4))

