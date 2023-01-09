#!/usr/bin/env python
# coding: utf-8

# In[379]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[380]:


df=pd.read_csv('CAR DETAILS.csv')
df.head()


# In[381]:


df.dtypes


# In[382]:


df.isnull().sum()


# In[383]:


df.shape


# In[384]:


df.columns


# In[385]:


df.describe().T


# In[386]:


df.info()


# In[387]:


# to know unique values
print(df.name.unique())
print(sorted(df.year.unique()))
print(df.fuel.unique())
print(df.seller_type.unique())
print(df.transmission.unique())
print(df.owner.unique())


# In[388]:


#add new coulmn by brand name split first part  
df["name"] = df.name.apply(lambda x : ' '.join(x.split(' ')[:1]))#Cuts first word only(car brand)from name column
df


# In[389]:


df.duplicated().sum()


# In[390]:


df[df.duplicated()]


# In[391]:


df['name'].value_counts()


# ###### There are duplicate values, but we will not delete them because it is natural that there are duplicate values, since same model car might be sold many times and similiarly with all the remaining columns

# #### EDA

# ###### Univariate Analysis

# In[392]:


sns.pairplot(df)


# In[393]:


df.name.value_counts().plot(kind='bar', figsize=(20,5))


# #### Conclusion:The best selling car is Maruti next comes Hyundai.

# In[394]:


plt.figure(figsize=(10, 8))

sns.histplot(data=df,x='year',bins=15)


# #### Conclusion:Most sales of the car were made in  the year 2017.

# In[395]:


plt.figure(figsize=(10,8))
sns.histplot(data=df, x='year', bins=15, hue='seller_type')


# #### Conclusion: Most of the sales are made  individually.

# In[396]:


plt.figure(figsize=(12, 10))
sns.barplot(x=df['name'],y=df['km_driven'],ci=None)
plt.xticks(rotation='vertical')
plt.show()


# #### Conclusion:Mitsubishi has driven more number of km than any other car.

# In[397]:


plt.figure(figsize=(12,10))
sns.countplot(data=df,x=df['name'],hue=df['fuel'])
plt.xticks(rotation=90)
plt.show()


# #### Conclusion:Mostly petrol is used as a means of fuels.

# In[398]:


sns.distplot(x=df['selling_price'])
plt.show()


# ###### Bivariate Analysis

# In[399]:


plt.figure(figsize=(16,8))
sns.barplot(x=df['year'], y=df['selling_price'],ci=None,hue=df['seller_type'])


# In[400]:


sns.barplot(x=df['owner'],y=df['selling_price'],ci=None,hue=df['seller_type'])
plt.xticks(rotation=90)
plt.show()


# In[401]:


sns.barplot(x='transmission',y='selling_price',data=df,palette='spring',ci=None)


# #### Conclusion:Most of the car transmission were automatic.

# In[402]:


sns.barplot(data=df,x='seller_type',y='selling_price',ci=None)


# #### Conclusion:Most of the seller type were Trustmark Dealer.

# In[403]:


plt.figure(figsize=(16, 4))
sns.barplot(x=df['name'],y=df['selling_price'],ci=None)
plt.xticks(rotation='vertical')
plt.show()


# #### Conclusion:Most expensive car is Land and the second most expensive car is BMW.

# #### Feature Engineering

# In[404]:


df1=df.iloc[:,[0,4,5,6,7,1,3,2]]
df1.head()


# In[405]:


df1.shape


# In[406]:


df1.dtypes


# ### Checking for outliers

# In[407]:


col_name = df1.select_dtypes(include=['int','float']).columns
for i in col_name:
  mean = df1[i].mean()
  med =  df1[i].median()
  print(f'Mean for {i} is {mean}')
  print(f'Median for {i} is {med}')


# In[408]:


df1.describe()


# In[409]:


plt.boxplot(df1.selling_price)
plt.show()


# ### Treating Outliers

# In[410]:


def outliers(col_name):
    Q1=df1[col_name].quantile(0.25)
    Q3 =df1[col_name].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3+(1.5*IQR)
    lower = Q1-(1.5*IQR)
    df1.drop(df1[(df1[col_name] > upper) | (df1[col_name] < lower)].index, inplace=True)


# In[411]:


outliers('selling_price')


# In[412]:


plt.boxplot(df1.selling_price)
plt.show()


# In[413]:


df1.shape


# In[414]:


df1.head()


# In[415]:


df1.corr()


# In[416]:


sns.heatmap(df1.corr(), cmap = 'RdBu', annot=True);


# #### Selecting dependent and independent features

# In[417]:


x=df1.drop(['selling_price'],axis=1)
y=df1['selling_price']
print(x.shape)
print(y.shape)
print(type(x))
print(type(y))


# In[418]:


y


# In[419]:


x


# In[420]:


from sklearn.model_selection import train_test_split


# In[421]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=100)
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)
print(type(x_train),type(x_test))
print(type(y_train),type(y_test))


# In[422]:


x_train


# In[423]:


y_train


# ### Scalling Data

# #### Scalling categorical data of x_train

# In[424]:


from sklearn.preprocessing import LabelEncoder


# In[425]:


lb=LabelEncoder()


# In[426]:


x_train['name']=lb.fit_transform(x_train['name'])
x_train['fuel']=lb.fit_transform(x_train['fuel'])
x_train['seller_type']=lb.fit_transform(x_train['seller_type'])
x_train['transmission']=lb.fit_transform(x_train['transmission'])
x_train['owner']=lb.fit_transform(x_train['owner'])


# In[427]:


x_train.head()


# #### Scalling numerical data of x_train

# In[428]:


from sklearn.preprocessing import StandardScaler


# In[429]:


sc=StandardScaler()


# In[430]:


x_train_num =x_train[['year','km_driven']]

x_train_num.head()


# In[431]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[432]:


x_train[['year','km_driven']]=pd.DataFrame(sc.fit_transform(x_train_num),columns=x_train_num.columns,index=x_train_num.index)


# In[433]:


x_train.head()


# #### Scalling categorical data on x_test

# In[434]:


x_test['name']=lb.fit_transform(x_test['name'])
x_test['fuel']=lb.fit_transform(x_test['fuel'])
x_test['seller_type']=lb.fit_transform(x_test['seller_type'])
x_test['transmission']=lb.fit_transform(x_test['transmission'])
x_test['owner']=lb.fit_transform(x_test['owner'])


# In[435]:


x_test.head()


# #### Scalling numerical data of x_test

# In[436]:


x_test_num =x_test[['year','km_driven']]
x_test_num.head()


# #### Scalling numerical data of x_test

# In[437]:


x_test[['year','km_driven']]=pd.DataFrame(sc.fit_transform(x_test_num),columns=x_test_num.columns,index=x_test_num.index)


# In[438]:


x_test.head()


# ### Model Building

# In[439]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor


# In[440]:


Lr=LinearRegression()
Lr.fit(x_train,y_train)


# In[441]:


ypred_Lr=Lr.predict(x_test)


# In[442]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,ypred_Lr))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,ypred_Lr))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,ypred_Lr)))


# In[443]:


Lr.score(x_train,y_train)


# In[444]:


print('Training Score',Lr.score(x_train,y_train))
print('Testing Score',Lr.score(x_test,y_test))


# #### KNN

# In[445]:


knn=KNeighborsRegressor(n_neighbors=15)
knn.fit(x_train,y_train)


# In[446]:


ypred_knn=knn.predict(x_test)


# In[447]:


print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,ypred_knn))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,ypred_knn))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,ypred_knn)))


# In[448]:


print('Training Score',knn.score(x_train,y_train))
print('Testing Score',knn.score(x_test,y_test))


# #### Ridge

# In[449]:


Ridge=Ridge()
Ridge.fit(x_train,y_train)


# In[450]:


ypred_ridge=Ridge.predict(x_test)


# In[451]:


print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,ypred_ridge))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,ypred_ridge))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,ypred_ridge)))


# In[452]:


print('Training Score',Ridge.score(x_train,y_train))
print('Testing Score',Ridge.score(x_test,y_test))


# #### Lasso

# In[453]:


Lasso=Lasso()
Lasso.fit(x_train,y_train)


# In[454]:


ypred_lasso=Lasso.predict(x_test)


# In[455]:


print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,ypred_lasso))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,ypred_lasso))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,ypred_lasso)))


# In[456]:


print('Training Score',Lasso.score(x_train,y_train))
print('Testing Score',Lasso.score(x_test,y_test))


# #### Decision Tree Regressor

# In[457]:


from sklearn.tree import DecisionTreeRegressor


# In[458]:


dt=DecisionTreeRegressor()


# In[459]:


dt.fit(x_train,y_train)


# In[460]:


ypred_dt=dt.predict(x_test)


# In[461]:


print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,ypred_dt))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,ypred_dt))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,ypred_dt)))


# In[462]:


print('Training Score',dt.score(x_train,y_train))
print('Testing Score',dt.score(x_test,y_test))


# #### Random Forest Regressor

# In[463]:


from sklearn.ensemble import RandomForestRegressor


# In[464]:


rf=RandomForestRegressor()


# In[465]:


rf.fit(x_train,y_train)


# In[466]:


ypred_rf=rf.predict(x_test)


# In[467]:


print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,ypred_rf))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,ypred_rf))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,ypred_rf)))


# In[468]:


print('Training Score',rf.score(x_train,y_train))
print('Testing Score',rf.score(x_test,y_test))


# ### Gradient Boosting

# In[469]:


from sklearn.ensemble import GradientBoostingRegressor


# In[470]:


gb= GradientBoostingRegressor()


# In[471]:


gb.fit(x_train,y_train)


# In[472]:


ypred_gb=gb.predict(x_test)


# In[473]:


print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,ypred_gb))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,ypred_gb))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,ypred_gb)))


# In[474]:


print('Training Score',gb.score(x_train,y_train))
print('Testing Score',gb.score(x_test,y_test))


# ### Inference

# #### Based on accuracy,Gradient Boosting Regressor performs better compared to other models getting an Training score of 70%.

# ### Saving the Gradient Boosting Regressor model

# ### Dumping the model

# In[475]:


from pickle import dump


# In[476]:


dump(lb,open('label_encoder.pkl','wb'))
dump(sc,open('standard_scaler.pkl','wb'))
dump(gb,open('gb.pkl','wb'))


# ### Loading the model

# In[477]:


from pickle import load


# In[478]:


lb=load(open('label_encoder.pkl','rb'))
sc=load(open('standard_scaler.pkl','rb'))
gb=load(open('gb.pkl','rb'))


# In[479]:


df.head()


# In[480]:


x_train.head()


# In[481]:


name = input("enter name : ")
fuel = input("enter fuel type : ")
st = input("enter seller type : ")
trns = input("enter transmission type : ")
own = input("enter owner type : ")
yr = int(input("enter year : "))
kmd = int(input("enter km driven : "))


# In[482]:


query_cat=pd.DataFrame({'name':[name], 'fuel':[fuel],'seller_type':[st],'transmission':[trns],'owner':[own]})
query_num=pd.DataFrame({'year':[yr], 'km_driven':[kmd]})


# In[483]:


query_num


# In[484]:


query_num[['year','km_driven']]=pd.DataFrame(sc.fit_transform(query_num),columns=query_num.columns,index=query_num.index)


# In[485]:


query_num


# In[486]:


query_cat['name']=lb.fit_transform(query_cat['name'])
query_cat['fuel']=lb.fit_transform(query_cat['fuel'])
query_cat['seller_type']=lb.fit_transform(query_cat['seller_type'])
query_cat['transmission']=lb.fit_transform(query_cat['transmission'])
query_cat['owner']=lb.fit_transform(query_cat['owner'])


# In[487]:


query_cat


# In[488]:


query=pd.concat([pd.DataFrame(query_cat),pd.DataFrame(query_num)],axis=1)


# In[489]:


query


# In[490]:


Selling_price=gb.predict(query)


# In[491]:


print(f"Selling Price is Rs {round(Selling_price[0],0)}")


# In[492]:


df5=df.copy()


# In[493]:


df5.to_csv('car_df.csv')


# ### Getting samples from datasets

# In[494]:


sample=df.sample(20)
sample


# In[495]:


sample.to_csv('sample.csv')


# In[496]:


sample


# In[497]:


a=sample.drop(['selling_price'],axis=1)
b=sample['selling_price']


# In[498]:


a.head()


# In[499]:


b.head()


# #### Scalling

# In[500]:


a['name']=lb.fit_transform(a['name'])
a['fuel']=lb.fit_transform(a['fuel'])
a['seller_type']=lb.fit_transform(a['seller_type'])
a['transmission']=lb.fit_transform(a['transmission'])
a['owner']=lb.fit_transform(a['owner'])


# In[501]:


a.head()


# In[502]:


a_num =a[['year','km_driven']]
a_num.head()


# In[505]:


a[['year','km_driven']]=pd.DataFrame(sc.fit_transform(a_num),columns=a_num.columns,index=a_num.index)


# In[507]:


a.head()


# In[528]:


b=a.iloc[:,[0,3,4,5,6,1,2]]
b.head()


# In[529]:


pred=gb.predict(b)


# In[537]:


pred


# In[530]:


predicted=pd.DataFrame(pred, 
             columns=['predicted value'])


# In[533]:


earlier=pd.concat([x,y], axis=1)


# In[534]:


earlier


# In[535]:


prediction=pd.concat([earlier,predicted],axis=1)


# In[536]:


prediction


# In[ ]:




