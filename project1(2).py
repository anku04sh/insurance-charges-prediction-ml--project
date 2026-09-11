import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

df= pd.read_csv("i.csv")


#EDA


print(df.head())

#it shows total rows and columns
print(df.shape)

#it shows datatypes of the column 
print(df.info())

#it shows mean median mode amx min std (only for numeric columns)
print(df.describe())

#it shows null values
print(df.isnull().sum())

print(df.columns)


#data viisualisation


numeric_col = ['age', 'bmi', 'children', 'charges']
for col in numeric_col:
    plt.figure(figsize=(6,4))
    sns.histplot (df[col],kde=True,bins=20)
    #plt.show()
   
    
sns.countplot( x = df['children'])  
#plt.show()

sns.countplot(x=df['sex'])
#plt.show()

sns.countplot(x=df['smoker'])
#plt.show()

sns.countplot(x=df['region'])
#plt.show()

for col in numeric_col:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    #plt.show()


plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)
#plt.show()



#Data cleaning and Preprocessing



df_cleaned = df.copy()
print(df_cleaned)
print(df_cleaned.shape)
print(df_cleaned.drop_duplicates(inplace=True))
print(df_cleaned.shape)

#missing values
print(df.isnull().sum())

print(df_cleaned.dtypes)

#valuecounts
print(df_cleaned['sex'].value_counts())

#label encoding
df_cleaned["sex"]=df_cleaned["sex"].map({"male":0,"female":1})

df_cleaned["smoker"]=df_cleaned["smoker"].map({"no":0,"yes":1 })

#valuecount
print(df_cleaned['region'].value_counts())

#for region coulmn there are 4 values so we use One-hot Encoding
df_cleaned=pd.get_dummies(df_cleaned,columns=['region'])

#True and False into int
df_cleaned=df_cleaned.astype(int)



#feature engineering and extraction


df_cleaned['bmi_category']=pd.cut(
    df_cleaned["bmi"],
    bins=[0,18.5,24.9,29.9,float("inf")],
    labels=['Underweight','normal','overweight','Obese'])


df_cleaned=pd.get_dummies(df_cleaned,columns=['bmi_category'])

df_cleaned=df_cleaned.astype(int)



#Feature Scaling


print(df_cleaned.columns)

from sklearn.preprocessing import StandardScaler
col=['age','bmi','children']
scaler=StandardScaler()
df_cleaned[col]=scaler.fit_transform(df_cleaned[col])

print(df_cleaned.head(20))
print(df_cleaned.shape)


#ml
from sklearn.model_selection import train_test_split

X=df_cleaned.drop('charges',axis=1)
Y=df_cleaned['charges']

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=42)

#linear regression model 
from sklearn.linear_model import LinearRegression
model=LinearRegression()

#training
model.fit(X_train,y_train)

#testing 
y_pred=model.predict(X_test)
print(y_pred)

from sklearn.metrics import r2_score
r2= r2_score(y_test,y_pred)
print(r2)

#adjusted r2
n=X_test.shape[0]
p=X_test.shape[1]
adjusted_r2= 1- ((1-r2) * (n-1) / (n-p-1))
print(adjusted_r2)
