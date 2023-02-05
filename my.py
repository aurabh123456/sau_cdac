#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
#PREPROCESSING
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

#MODELLING
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve,auc


# In[2]:


data=pd.read_csv("Bank1.csv")


# In[3]:


data


# In[4]:


#preprocessing techniques
#1)standardization(replace by zscore)
#2)normalization(min max scalar)
#3)principle component analysis


# In[5]:


#logically neglect accound_id ,obs
data.drop(['Obs','account_id'],axis=1, inplace=True)


# In[6]:


data.columns


# In[7]:


data['second']


# In[8]:


#split features and target variables first.
X=data.iloc[:,0:37]
y=data.iloc[:,-1]


# In[9]:


X.columns


# In[10]:


y


# In[11]:


#splitting categorical from X dataframe
categorical=X[['sex', 'card', 'second', 'frequency', 'region']]


# In[12]:


categorical.columns


# In[13]:


categorical


# In[14]:


#dropping categorical from X
X.drop(['sex', 'card', 'second', 'frequency', 'region'],axis=1,inplace=True)


# In[15]:


X.columns


# In[16]:


#normalize(zscore ) for numeric and coding number for catogorical
##update columns with there normalized values
sc=StandardScaler()
X[X.columns]= sc.fit_transform(X)
X


# In[17]:


#labelling for categorical
#convert string variable to encoding
labeled=pd.get_dummies(categorical)


# In[18]:


labeled


# In[19]:


#Encode the y(TARGET/DEPENDENT) variable as well

labelencoder_y=LabelEncoder()
y_final=labelencoder_y.fit_transform(y)
y_final


# In[20]:


#concat the dataframe by clubbing catogorical + numeric data of independent variables
X_conc=pd.concat([X,labeled],axis=1)


# In[21]:


X_conc


# #MODELLING IS STARTED AS WE HAVE OUR PREPROCESSING DATA
# 

# In[22]:


# Creating test- Train splits 20% test_size
X_train,X_test,y_train,y_test=train_test_split(X_conc,y_final,test_size=0.2, random_state=42)

# create the claassifier
logreg=LogisticRegression(max_iter=500)

#fit the classifier to the training data
logreg.fit(X_train,y_train)

#predict th lebels of the test set
y_pred=logreg.predict(X_test)

# compute and print the confision martis and the classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[23]:


#accuracy of model to predict correctly is 92.7%((TP+FN)/TOTAL)
#7.3% is failure


# In[24]:


import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit(maxiter=10000)
print(result.summary2())


# #BY MIN MAX SCALING METHOD USED IN PYTHON (2 ND PREPROCESSING METHOD)#NORMALIZATION

# In[25]:


data.drop(['sex', 'card', 'second', 'frequency', 'region','good'],axis=1,inplace=True)


# In[26]:


def sci_minmax(X):
    minmax_scale = MinMaxScaler(feature_range=(0, 1), copy=True)
    return minmax_scale.fit_transform(X)

data_normalized = sci_minmax(data)
data_variance=data_normalized.var()
data_variance


# In[27]:


data_normalized


# In[28]:


df1 = pd.DataFrame(data_normalized, columns = ['cardwdln', 'cardwdlt', 'cashcrn', 'cashcrt', 'cashwdn', 'cashwdt',
       'bankcolt', 'bankcoln', 'bankrn', 'bankrt', 'othcrn', 'othcrt', 'days',
       'age', 'cardwdlnd', 'cardwdltd', 'cashcrnd', 'cashcrtd', 'cashwdnd',
       'cashwdtd', 'bankcoltd', 'bankcolnd', 'bankrnd', 'bankrtd', 'othcrnd',
       'othcrtd', 'acardwdl', 'acashcr', 'acashwd', 'abankcol', 'abankr',
       'aothcr'])


# In[29]:


df1


# In[30]:


#df1 is my normalized frame 
#concat normalized numeric + categorical
X_conc1=pd.concat([df1,labeled],axis=1)


# In[31]:


X_conc1


# #MODELLING IS STARTED AS WE HAVE OUR PREPROCESSING DATA

# In[32]:


# Creating test- Train splits
X_train1,X_test1,y_train1,y_test1=train_test_split(X_conc1,y_final,test_size=0.2, random_state=42)

# create the claassifier
logreg1=LogisticRegression(max_iter=500)

#fit the classifier to the training data
logreg1.fit(X_train1,y_train1)

#predict th lebels of the test set
y_pred1=logreg1.predict(X_test1)

# compute and print the confision martis and the classification report
print(confusion_matrix(y_test1,y_pred1))
print(classification_report(y_test1,y_pred1))


# In[33]:


#model accuracy is 87.59% for right prediction


# #pCA PRERROCESSING METHOD

# In[34]:


labeled


# In[35]:


data


# In[36]:


final_x=pd.concat([data,labeled],axis=1)


# In[37]:


x1= StandardScaler().fit_transform(final_x)


# In[38]:


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x1)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf


# MODELLING WITH PCA TYPE PREPROCESSING

# In[ ]:





# In[39]:


y_final


# In[40]:


data1=pd.read_csv("Bank1.csv")


# In[41]:


finalDf = pd.concat([principalDf,data1['good'] ], axis = 1)


# In[42]:


x_train11,x_test11,y_train11,y_test11=train_test_split(finalDf[["principal component 1","principal component 2"]],finalDf["good"],test_size=0.2,random_state=0)


# In[43]:


y_train11.head()


# In[44]:


# create the claassifier
logreg15=LogisticRegression(random_state = 0)
#fit the classifier to the training data
logreg15.fit(x_train11,y_train11)


# In[45]:


y_predion = logreg15.predict(x_test11)


# In[46]:


y_predion


# In[47]:


x_test11


# In[48]:


x_train11,y_train11


# In[49]:


logreg15.score(x_test11, y_test11)


# In[50]:


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg15.score(x_test11, y_test11)))


# In[51]:


pca.explained_variance_ratio_.sum()


# In[52]:


final_x


# In[53]:


X_conc1


# In[54]:


y_final


# #DECISION TREE

# In[55]:


train1_x,test1_x,train1_y,test1_y=train_test_split(X_conc1,y_final,test_size=0.2, random_state=42)


# In[56]:


test1_y


# In[57]:


# Decision tree

clf = DecisionTreeClassifier()
clf.fit(train1_x,train1_y)

predict = clf.predict(test1_x)
print(classification_report(test1_y,predict))
print("confusion matrix")
print(confusion_matrix(test1_y,predict))


# In[58]:


#model accuracy ~ 92%

