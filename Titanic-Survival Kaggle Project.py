#!/usr/bin/env python
# coding: utf-8

# # Titanic - Machine Learning from Disaster
# 

# # Step1:Import all libraries

# In[78]:


import os as o 
import pandas as p
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
# machine learning
from sklearn.linear_model import LogisticRegression


# # Step2:Importing dataset
#  2.1 Since data is in form of csv file we have to use pandas read_csv to load the data.
#  
# 2.2 After loading it is important to check the complete information of data as it can indication many of the hidden infomation such as null values in a column or a row.
# 

# In[2]:


trainData=p.read_csv('C:/Users/52038585/Desktop/Kaggle/titanic/train.csv')
testData=p.read_csv('C:/Users/52038585/Desktop/Kaggle/titanic/test.csv')


# In[3]:


trainData.shape


# In[4]:


trainData.head()


# # Feature Engineering 
# 
# # Step 3: Check whether any null values exists or not in dataset

# In[47]:


trainData.isnull().sum()


# In[ ]:


#We can see that Age is a integer data type and have null values Therefore we have to fill null values via Median


# In[6]:


trainData['Age_Median']=trainData['Age'].fillna(trainData['Age'].median())


# In[7]:


#Counting the Null percentage for Cabin

percentage= (trainData['Cabin'].isnull().sum() / len(trainData['Cabin']))*100

percentage


# In[8]:


#As more than 70% data is null for Cabin attribute so we can drop the column.
trainData.drop(columns=['Cabin'], axis=1,  inplace=True)


# In[9]:


#One Hot Encoding method  is used to convert categorical data into numerical data.

trainData= p.get_dummies(trainData, columns=['Sex', 'Embarked'], drop_first= True)


# In[46]:


#Now we have to drop original columns 

#trainData.drop(columns=['Age'], axis=1,  inplace=True)
trainData.drop(columns=['Name'], axis=1,  inplace=True)
trainData.drop(columns=['Ticket'], axis=1,  inplace=True)


# # EDA -Visualization 

# In[11]:


#Below graph shows the survival count 
plt.figure(figsize=(8,6))
sb.countplot(x='Survived', data= trainData)


# In[12]:


#Below graph shows how many male and female survived.

plt.figure(figsize=(8,6))
sb.countplot(x='Survived', hue='Sex_male', data= trainData)


# In[13]:


#Below graph shows how many male and female survived as per their age.
plt.figure(figsize=(8,6))
sb.boxplot(x='Survived', y= 'Age_Median', hue='Sex_male', data= trainData)


# In[14]:


#Below graph shows survival rate as per Pclass.
plt.figure(figsize=(8,6))
sb.countplot(x='Survived', hue='Pclass', data= trainData)


# # Check the outliers in training data

# In[87]:


plt.figure(figsize=(20,25))
sb.boxplot( data= trainData)


# In[15]:


#As a result There are outliers in Age_Median, SibSp, Parch and Fare variables.
#I am using clip() funtion to remove the outliers. 


# In[16]:


#Outlier Treatment 
cols= ['Age_Median', 'SibSp', 'Parch', 'Fare']

trainData[cols]= trainData[cols].clip(lower= trainData[cols].quantile(0.15), upper= trainData[cols].quantile(0.85), axis=1)


# In[102]:


trainData.plot(kind='box', figsize= (25,20)) 


# In[17]:


#Dropping Parch variable is because more than 75% of the values are 0.

trainData.drop(columns=['Parch'], axis=1, inplace=True)

#Dropping Embarked_Q because most of the values are zero

trainData.drop(columns=['Embarked_Q'], axis=1, inplace=True)


# In[18]:


#There are no Outliers in  Training dataset now.


# In[19]:


trainData.head(20)


# In[183]:


#Feature Scaling to fetch the important independent variables which are highly correlated with dependent variable 


# In[20]:


corrmat = trainData.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(12,12))
#plot heat map
g=sb.heatmap(trainData[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[21]:


#Remove the correlated 
threshold=0.3


# In[22]:



# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[23]:


correlation(trainData.iloc[:,:-1],threshold)


# # TEST Data -EDA
# 
# # Feature Engineering 
# 

# In[40]:


#Check whether any null values exists or not in dataset
testData.isnull().sum()


# In[25]:


testData.shape


# In[34]:


#Fill Null values -Age with median 
testData['Age_Median']=testData['Age'].fillna(testData['Age'].median())


# In[39]:


#Fill Null values -Fare with median 
testData['Fare_Median']=testData['Fare'].fillna(testData['Fare'].median())


# In[37]:


testData.info()


# In[28]:


#One Hot Encoding method  is used to convert categorical data into numerical data.

testData= p.get_dummies(testData, columns=['Sex', 'Embarked'], drop_first= True)


# In[30]:


#Counting the Null percentage for Cabin

percentageTest= (testData['Cabin'].isnull().sum() / len(testData['Cabin']))*100

percentageTest


# In[32]:


#As 78% data is null for Cabin column - so as a result we can drop it .
testData.drop(columns=['Cabin'], axis=1,  inplace=True)


# In[49]:


#Drop Original columns
testData.drop(columns=['Fare'], axis=1,  inplace=True)
testData.drop(columns=['Age'], axis=1,  inplace=True)
testData.drop(columns=['Name'], axis=1,  inplace=True)
testData.drop(columns=['Ticket'], axis=1,  inplace=True)


# # Train Test Split 

# In[58]:


X_train = trainData.drop("Survived", axis=1)
y_train = trainData["Survived"]
print("X_train.shape" ,X_train.shape)
print("y_train.shape" ,y_train.shape)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=111)


# In[59]:


print("X_train.shape" ,X_train.shape)
print("y_train.shape" ,y_train.shape)
print("X_test.shape" ,X_test.shape)
print("y_test.shape" ,y_test.shape)


# # Building Model 

# In[70]:


# Logistic Regression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred_lr = logreg.predict(X_test)
Score_lr = accuracy_score(y_test,Y_pred_lr)
print(Score_lr)


# In[71]:


y_test.shape


# In[72]:


Y_pred_lr.shape


# In[82]:


my_submission = p.DataFrame({'PassengerId': X_test.PassengerId, 'Survived': Y_pred_lr})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[81]:





# In[ ]:




