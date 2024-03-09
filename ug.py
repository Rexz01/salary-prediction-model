import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sal_data = pd.read_csv('SalaryData1.csv')
print(sal_data.head() )
sal_data.shape
sal_data.columns=['Age','Gender','Degree','Job_Title','Experience_years','Salary']
print(sal_data.dtypes)
sal_data.info()
sal_data[sal_data.duplicated()]
sal_data1=sal_data.drop_duplicates(keep='first')
sal_data1.isnull().sum()
sal_data1.dropna(how ='any',inplace=True)
sal_data1.shape 
print(sal_data1.head())
print(sal_data1.describe())
corr =sal_data1[['Age','Experience_years','Salary']].corr()
print(corr)
sns.heatmap(corr,annot =True)
print(sal_data1['Degree'].value_counts())
print(sal_data1[ 'Job_Title'].unique())
print(sal_data1 ['Gender'].value_counts())
print(sal_data1.head())
from sklearn.preprocessing import LabelEncoder
Label_Encoder =LabelEncoder()
sal_data1['Gender_Encode']=Label_Encoder.fit_transform(sal_data1['Gender'])
sal_data1['Degree_Encode']=Label_Encoder.fit_transform(sal_data1['Degree'])
sal_data1['Job_Title_Encode']=Label_Encoder.fit_transform(sal_data1['Job_Title'])
print(sal_data1.head)
from sklearn.preprocessing import StandardScaler
std_scaler =StandardScaler()
sal_data1['Age_scaled']=std_scaler.fit_transform(sal_data1[['Age']])
sal_data1['Experience_years_scaled']=std_scaler.fit_transform(sal_data1[['Experience_years']])
print(sal_data1.head)
X=sal_data1[['Age_scaled','Gender_Encode','Degree_Encode','Job_Title_Encode','Experience_years_scaled']]
Y=sal_data1['Salary']
print(X.head())
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.2,random_state=42)
print(X_train.head())
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LinearRegression
Linear_regression_model =LinearRegression()
Linear_regression_model.fit(X_train, Y_train)
Y_pred_lr=Linear_regression_model.predict(X_test)
Y_pred_lr
df=pd.DataFrame({'Y_Actual' :Y_test,'Y_predicted':Y_pred_lr})
print(df.head())
df['Error']=df['Y_Actual']- df['Y_predicted']
print(df.head())
df['abs_error']=abs(df['Error'])
print(df.head())
Mean_absolute_Error=df['abs_error'].mean()
print(Mean_absolute_Error)
from sklearn.metrics import accuracy_score,r2_score
from sklearn.metrics import mean_squared_error , mean_absolute_error
print(r2_score(Y_test,Y_pred_lr))
print(f"Accuracy of the model ={round(r2_score(Y_test,Y_pred_lr),4)*100}%") 
print(round(mean_absolute_error(Y_test,Y_pred_lr),2))
print(f"Mean_absolute_error ={round(mean_absolute_error(Y_test,Y_pred_lr),2)}")
mse=round(mean_squared_error(Y_test,Y_pred_lr),2)
print(mse)
print(f"Mean Squared error = {round(mean_squared_error(Y_test,Y_pred_lr),2)}")
print('root Mean Squared error(RMSE)=', mse**(0.5))
Linear_regression_model.coef_
Linear_regression_model.intercept_
print(sal_data1.head())
Age1=std_scaler.transform([[32]])
Age =3.30432246
Gender =0
Degree =2
Job_Title =22
Experience_years1=std_scaler.transform([[9]])
Experience_years = 0.74415815
print(std_scaler.transform([[32]]))
Emp_Salary =Linear_regression_model.predict([[Age,Gender,Degree,Job_Title,Experience_years]])
print(Emp_Salary,)
print("salary =",Emp_Salary[0])
