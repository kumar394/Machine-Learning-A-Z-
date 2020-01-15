# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print("Hello World")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading data
dataset= pd.read_csv("Data.csv")
x= dataset.iloc[:, :-1].values
y= dataset.iloc[:, 3].values
#taking care of missing data
from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=np.nan,strategy="mean")
imputer= imputer.fit(x[:, 1:3])
x[:, 1:3]= imputer.transform(x[:, 1:3])

#handling categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:, 0]= labelencoder_x.fit_transform(x[:, 0])
onehotencoder= OneHotEncoder(categorical_features=[0])
x= onehotencoder.fit_transform(x).toarray()
labelencoder_y=LabelEncoder()
y= labelencoder_y.fit_transform(y)

#splitting into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=0) 

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train= sc_x.fit_transform(x_train)
x_test= sc_x.transform(x_test)

#Simple Linear Regression
slr_data=pd.read_csv("Salary_Data.csv")
slr_x= slr_data.iloc[:, :-1].values
slr_y= slr_data.iloc[:, 1].values
from sklearn.model_selection import train_test_split
slr_x_train, slr_x_test, slr_y_train, slr_y_test= train_test_split(slr_x,slr_y,test_size=1/3, random_state=0)
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(slr_x_train, slr_y_train)
slr_y_pred= regressor.predict(slr_x_test)

#visualization training set results
plt.scatter(slr_x_train, slr_y_train, color="red")
plt.plot(slr_x_train, regressor.predict(slr_x_train), color="blue")
plt.title("Salary vs Experience (Train data)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#visualization test set results
plt.scatter(slr_x_test, slr_y_test, color="red")
plt.plot(slr_x_train, regressor.predict(slr_x_train), color="blue")
plt.title("Salary vs Experience (Test data)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#multiple linear regression
mlr_data= pd.read_csv("50_Startups.csv")
mlr_x= mlr_data.iloc[:, :-1].values
mlr_y= mlr_data.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
mlr_labelencoder= LabelEncoder()
mlr_x[:,3]=mlr_labelencoder.fit_transform(mlr_x[:, 3])
mlr_onehotencoder= OneHotEncoder(categorical_features=[3])
mlr_x= mlr_onehotencoder.fit_transform(mlr_x).toarray()

mlr_x=mlr_x[:, 1:]#dummy trap
from sklearn.model_selection import train_test_split
mlr_x_train, mlr_x_test, mlr_y_train, mlr_y_test= train_test_split(mlr_x,mlr_y,test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
mlr_regressor= LinearRegression()
mlr_regressor.fit(mlr_x_train, mlr_y_train)

mlr_y_pred= mlr_regressor.predict(mlr_x_test)

#backward multiple regression
import statsmodels.formula.api as sm
mlr_x= np.append(arr= np.ones((50,1)).astype(int), values= mlr_x, axis=1)
mlr_x_opt= mlr_x[:, [0, 1, 2, 3, 4, 5]]
mlr_regressor_ols= sm.OLS(endog= mlr_y, exog= mlr_x_opt).fit()
mlr_regressor_ols.summary()

mlr_x_opt= mlr_x[:, [0, 1, 3, 4, 5]]
mlr_regressor_ols= sm.OLS(endog= mlr_y, exog= mlr_x_opt).fit()
mlr_regressor_ols.summary()

mlr_x_opt= mlr_x[:, [0, 3, 4, 5]]
mlr_regressor_ols= sm.OLS(endog= mlr_y, exog= mlr_x_opt).fit()
mlr_regressor_ols.summary()

mlr_x_opt= mlr_x[:, [0, 3, 5]]
mlr_regressor_ols= sm.OLS(endog= mlr_y, exog= mlr_x_opt).fit()
mlr_regressor_ols.summary()

mlr_x_opt= mlr_x[:, [0, 3]]
mlr_regressor_ols= sm.OLS(endog= mlr_y, exog= mlr_x_opt).fit()
mlr_regressor_ols.summary()

#polynomial linear regression
plr_dataset= pd.read_csv("Position_Salaries.csv")
plr_x= plr_dataset.iloc[:, 1:2].values
plr_y= plr_dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(plr_x, plr_y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=4)
poly_x= poly_reg.fit_transform(plr_x)
plr_reg= LinearRegression()
plr_reg.fit(poly_x,plr_y)

#visualization for LR model
plt.scatter(plr_x, plr_y, color= "red")
plt.plot(plr_x, lin_reg.predict(plr_x), color= "blue")
plt.title("Truth or Bluff (LR)")
plt.xlabel("Position Level")
ply.ylabel("Salary")
plt.show()

#visualization for PLR model
x_grid= np.arange(min(plr_x), max(plr_x), 0.1)
x_grid= x_grid.reshape(len(x_grid),1)
plt.scatter(plr_x, plr_y, color= "red")
plt.plot(x_grid, plr_reg.predict(poly_reg.fit_transform(x_grid)), color= "blue")
plt.title("Truth or Bluff (PLR)")
plt.xlabel("Position Level")
ply.ylabel("Salary")
plt.show()

#predicting values
lin_reg.predict([[6.5]])
plr_reg.predict(poly_reg.fit_transform(np.array([6.5]).reshape(1,1)))
x=[[6.5]]
a= np.array([6.5]).reshape(1,1)

#SVR
svr_data= pd.read_csv("Position_Salaries.csv")
svr_x= svr_data.iloc[:, 1:2].values
svr_y= svr_data.iloc[:, 2:3].values
#feature scaling
from sklearn.preprocessing import StandardScaler
svr_x_sc= StandardScaler()
svr_y_sc= StandardScaler()
svr_x= svr_x_sc.fit_transform(svr_x)
svr_y= svr_y_sc.fit_transform(svr_y)
#fitting the model
from sklearn.svm import SVR
svr_regressor= SVR(kernel="rbf")
svr_regressor.fit(svr_x,svr_y)
svr_y_pred= svr_y_sc.inverse_transform(svr_regressor.predict(svr_x_sc.transform(np.array([[6.5]]))))
#visualization
plt.scatter(svr_x, svr_y, color="red")
plt.plot(svr_x, svr_regressor.predict(svr_x), color= "blue")
plt.title("SVR model")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show
#visualization with fine resolution
x_grid= np.arange(min(svr_x), max(svr_x), 0.1)
x_grid= x_grid.reshape(len(x_grid),1)
plt.scatter(svr_x, svr_y, color="red")
plt.plot(x_grid, svr_regressor.predict(x_grid), color= "blue")
plt.title("SVR model (fine resolution)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show

#Decision Tree
dt_data= pd.read_csv("Position_Salaries.csv")
dt_x= dt_data.iloc[:, 1:2].values
dt_y= dt_data.iloc[:,2].values
from sklearn.tree import DecisionTreeRegressor
dt_regressor= DecisionTreeRegressor(random_state=0)
dt_regressor.fit(dt_x,dt_y)
dt_y_pred= dt_regressor.predict([[6.5]])
#visualization
plt.scatter(dt_x, dt_y, color="red")
plt.plot(dt_x, dt_regressor.predict(dt_x), color= "blue")
plt.title("DTR model")
plt.xlabel("Position Level")
plt.ylabel("Salaries")
plt.show()
#visualization (fine resolution)
dt_x_grid= np.arange(min(dt_x), max(dt_x), 0.01)
dt_x_grid= dt_x_grid.reshape(len(dt_x_grid),1)
plt.scatter(dt_x, dt_y, color="red")
plt.plot(dt_x_grid, dt_regressor.predict(dt_x_grid), color= "blue")
plt.title("DTR model")
plt.xlabel("Position Level")
plt.ylabel("Salaries")
plt.show()

#Random Forest Tree
rf_data= pd.read_csv("Position_Salaries.csv")
rf_x= dt_data.iloc[:, 1:2].values
rf_y= dt_data.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
rf_regressor= RandomForestRegressor(n_estimators=300, random_state=0)
rf_regressor.fit(rf_x,rf_y)

rf_y_pred= rf_regressor.predict([[6.5]])

#visualization (fine resolution)
rf_x_grid= np.arange(min(rf_x), max(rf_x), 0.01)
rf_x_grid= rf_x_grid.reshape(len(rf_x_grid),1)
plt.scatter(rf_x, rf_y, color="red")
plt.plot(rf_x_grid, rf_regressor.predict(rf_x_grid), color= "blue")
plt.title("RF model")
plt.xlabel("Position Level")
plt.ylabel("Salaries")
plt.show()

#Logistic Regression
lr_data= pd.read_csv("Social_Network_Ads.csv")
lr_x= lr_data.iloc[:, [2,3]].values
lr_y= lr_data.iloc[:,4].values
#train_test split
from sklearn.model_selection import train_test_split
lr_x_train, lr_x_test, lr_y_train, lr_y_test= train_test_split(lr_x, lr_y, test_size=0.25, random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
lr_sc= StandardScaler()
lr_x_train= lr_sc.fit_transform(lr_x_train)
lr_x_test= lr_sc.fit_transform(lr_x_test)
#fitting logistic regression
from sklearn.linear_model import LogisticRegression
lr_classifier= LogisticRegression(random_state=0)
lr_classifier.fit(lr_x_train, lr_y_train)
#predicting
lr_y_pred= lr_classifier.predict(lr_x_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
lr_cm= confusion_matrix(lr_y_test, lr_y_pred)
#visualizing the training set result
from matplotlib.colors import ListedColormap
x_set, y_set= lr_x_train, lr_y_train
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
"""a,b=np.meshgrid(np.arange(start=0, stop=4, step=1),
                np.arange(start=0, stop=2, step=1))"""
plt.contourf(x1,x2, lr_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green")))
#c= np.array([[1,2,3], [2,3,4]])
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green"))(i), label=j)
plt.title("LR(Train_Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
#visualizing the test set result
from matplotlib.colors import ListedColormap
x_set, y_set= lr_x_test, lr_y_test
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
"""a,b=np.meshgrid(np.arange(start=0, stop=4, step=1),
                np.arange(start=0, stop=2, step=1))"""
plt.contourf(x1,x2, lr_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green")))
#c= np.array([[1,2,3], [2,3,4]])
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green"))(i), label=j)
plt.title("LR(Train_Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

"""c_set=[0,1,3]
for i,j in enumerate (np.unique(c_set)):
    print(i,j)"""
#a2= x_set[y_set== 0,0]
    
#KNN 
knn_data= pd.read_csv("Social_Network_Ads.csv")
knn_x= lr_data.iloc[:, [2,3]].values
knn_y= lr_data.iloc[:,4].values
#train_test split
from sklearn.model_selection import train_test_split
knn_x_train, knn_x_test, knn_y_train, knn_y_test= train_test_split(knn_x, knn_y, test_size=0.25, random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
knn_sc= StandardScaler()
knn_x_train= knn_sc.fit_transform(knn_x_train)
knn_x_test= knn_sc.transform(knn_x_test)
#fitting logistic regression
from sklearn.neighbors import KNeighborsClassifier
knn_classifier= KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
knn_classifier.fit(knn_x_train, knn_y_train)
#predicting
knn_y_pred= knn_classifier.predict(knn_x_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
knn_cm= confusion_matrix(knn_y_test, knn_y_pred)
#visualizing the training set result
from matplotlib.colors import ListedColormap
x_set, y_set= knn_x_train, knn_y_train
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
"""a,b=np.meshgrid(np.arange(start=0, stop=4, step=1),
                np.arange(start=0, stop=2, step=1))"""
plt.contourf(x1,x2, knn_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green")))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green"))(i), label=j)
plt.title("K-NN(Train_Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
#visualizing the test set result
from matplotlib.colors import ListedColormap
x_set, y_set= knn_x_test, knn_y_test
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
plt.contourf(x1,x2, knn_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green")))
#c= np.array([[1,2,3], [2,3,4]])
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green"))(i), label=j)
plt.title("KNN(Test_Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

#SVM 
svm_data= pd.read_csv("Social_Network_Ads.csv")
svm_x= svm_data.iloc[:, [2,3]].values
svm_y= svm_data.iloc[:,4].values
#train_test split
from sklearn.model_selection import train_test_split
svm_x_train, svm_x_test, svm_y_train, svm_y_test= train_test_split(svm_x, svm_y, test_size=0.25, random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
svm_sc= StandardScaler()
svm_x_train= svm_sc.fit_transform(svm_x_train)
svm_x_test= svm_sc.transform(svm_x_test)
#fitting logistic regression
from sklearn.svm import SVC
svm_classifier= SVC(kernel= "rbf", random_state=0)
svm_classifier.fit(svm_x_train, svm_y_train)
#predicting
svm_y_pred= svm_classifier.predict(svm_x_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
svm_cm= confusion_matrix(svm_y_test, svm_y_pred)
#visualizing the training set result
from matplotlib.colors import ListedColormap
x_set, y_set= svm_x_train, svm_y_train
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
plt.contourf(x1,x2, svm_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green")))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green"))(i), label=j)
plt.title("SVM(Train_Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
#visualizing the test set result
from matplotlib.colors import ListedColormap
x_set, y_set= svm_x_test, svm_y_test
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
plt.contourf(x1,x2, svm_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green")))
#c= np.array([[1,2,3], [2,3,4]])
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green"))(i), label=j)
plt.title("SVM(Test_Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

#Kernel SVM
#same as SVM, only need to change kernel in the SVC class

#Naive Bayes
nb_data= pd.read_csv("Social_Network_Ads.csv")
nb_x= nb_data.iloc[:, [2,3]].values
nb_y= nb_data.iloc[:,4].values
#train_test split
from sklearn.model_selection import train_test_split
nb_x_train, nb_x_test, nb_y_train, nb_y_test= train_test_split(nb_x, nb_y, test_size=0.25, random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
nb_sc= StandardScaler()
nb_x_train= nb_sc.fit_transform(nb_x_train)
nb_x_test= nb_sc.transform(nb_x_test)
#fitting logistic regression
from sklearn.naive_bayes import GaussianNB
nb_classifier= GaussianNB()
nb_classifier.fit(nb_x_train, nb_y_train)
#predicting
nb_y_pred= nb_classifier.predict(nb_x_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
nb_cm= confusion_matrix(nb_y_test, nb_y_pred)
#visualizing the training set result
from matplotlib.colors import ListedColormap
x_set, y_set= nb_x_train, nb_y_train
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
plt.contourf(x1,x2, nb_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green")))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green"))(i), label=j)
plt.title("NB(Train_Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
#visualizing the test set result
from matplotlib.colors import ListedColormap
x_set, y_set= nb_x_test, nb_y_test
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
plt.contourf(x1,x2, nb_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green")))
#c= np.array([[1,2,3], [2,3,4]])
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green"))(i), label=j)
plt.title("NB(Test_Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

#Decision Tree Classifier
dtc_data= pd.read_csv("Social_Network_Ads.csv")
dtc_x= dtc_data.iloc[:, [2,3]].values
dtc_y= dtc_data.iloc[:,4].values
#train_test split
from sklearn.model_selection import train_test_split
dtc_x_train, dtc_x_test, dtc_y_train, dtc_y_test= train_test_split(dtc_x, dtc_y, test_size=0.25, random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
dtc_sc= StandardScaler()
dtc_x_train= dtc_sc.fit_transform(dtc_x_train)
dtc_x_test= dtc_sc.transform(dtc_x_test)
#fitting logistic regression
from sklearn.tree import DecisionTreeClassifier
dtc_classifier= DecisionTreeClassifier(criterion="entropy", random_state=0)
dtc_classifier.fit(dtc_x_train, dtc_y_train)
#predicting
dtc_y_pred= dtc_classifier.predict(dtc_x_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
dtc_cm= confusion_matrix(dtc_y_test, dtc_y_pred)
#visualizing the training set result
from matplotlib.colors import ListedColormap
x_set, y_set= dtc_x_train, dtc_y_train
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
plt.contourf(x1,x2, dtc_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green")))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green"))(i), label=j)
plt.title("Decision Tree Classifier(Train_Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
#visualizing the test set result
from matplotlib.colors import ListedColormap
x_set, y_set= dtc_x_test, dtc_y_test
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
plt.contourf(x1,x2, dtc_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green")))
#c= np.array([[1,2,3], [2,3,4]])
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green"))(i), label=j)
plt.title("Decision Tree Classifier(Test_Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

#RandomForest Classifier
rfc_data= pd.read_csv("Social_Network_Ads.csv")
rfc_x= rfc_data.iloc[:, [2,3]].values
rfc_y= rfc_data.iloc[:,4].values
#train_test split
from sklearn.model_selection import train_test_split
rfc_x_train, rfc_x_test, rfc_y_train, rfc_y_test= train_test_split(rfc_x, rfc_y, test_size=0.25, random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
rfc_sc= StandardScaler()
rfc_x_train= rfc_sc.fit_transform(rfc_x_train)
rfc_x_test= rfc_sc.transform(rfc_x_test)
#fitting RF Classification
from sklearn.ensemble import RandomForestClassifier
rfc_classifier= RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
rfc_classifier.fit(rfc_x_train, rfc_y_train)
#predicting
rfc_y_pred= rfc_classifier.predict(rfc_x_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
rfc_cm= confusion_matrix(rfc_y_test, rfc_y_pred)
#visualizing the training set result
from matplotlib.colors import ListedColormap
x_set, y_set= rfc_x_train, rfc_y_train
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
plt.contourf(x1,x2, rfc_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green")))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green"))(i), label=j)
plt.title("RandomForest Classifier(Train_Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
#visualizing the test set result
from matplotlib.colors import ListedColormap
x_set, y_set= rfc_x_test, rfc_y_test
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
plt.contourf(x1,x2, rfc_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green")))
#c= np.array([[1,2,3], [2,3,4]])
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green"))(i), label=j)
plt.title("RandomForest Classifier(Test_Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

#K-Means Clustering
#%reset -f

km_data= pd.read_csv("Mall_Customers.csv")
km_x= km_data.iloc[:, [3,4]].values
from sklearn.cluster import KMeans
#plotting elbow graph
wcss=[]
for i in range(1,11):
    kmeans= KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0)
    kmeans.fit(km_x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.xticks(np.arange(1,11,1))
plt.title("Elbow Method Graph")
plt.xlabel("# of clusters")
plt.ylabel("wcss")    
plt.show()
#applying k-means with number of cluster-5
kmeans=KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=300, random_state=0)
km_y_pred= kmeans.fit_predict(km_x)
#visualising the clusters
plt.scatter(km_x[km_y_pred== 0,0], km_x[km_y_pred== 0,1], s=100, c="red", label= "Careful")  
plt.scatter(km_x[km_y_pred== 1,0], km_x[km_y_pred== 1,1], s=100, c="blue", label= "Standard")           
plt.scatter(km_x[km_y_pred== 2,0], km_x[km_y_pred== 2,1], s=100, c="green", label= "Target")
plt.scatter(km_x[km_y_pred== 3,0], km_x[km_y_pred== 3,1], s=100, c="cyan", label= "Careless")
plt.scatter(km_x[km_y_pred== 4,0], km_x[km_y_pred== 4,1], s=100, c="magenta", label= "Sensible")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c="yellow", label="Centroids")
plt.title("Cluster of clients")
plt.xlabel("Annual Income(k$)")
plt.ylabel("Spending Index(1-100)")
plt.legend()
plt.show()

#Hierarchical Clustering
hc_data= pd.read_csv("Mall_Customers.csv")
hc_x= hc_data.iloc[:, [3,4]].values
#creating dendogram
import scipy.cluster.hierarchy as sch
dendogram= sch.dendrogram(sch.linkage(hc_x, method= "ward"))
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show() 
#fitting the hierarchical clustering model to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage= "ward")
hc_y_pred= hc.fit_predict(hc_x)  
#visualization
plt.scatter(hc_x[hc_y_pred== 0,0], hc_x[hc_y_pred== 0,1], s=100, c= "red", label= "c1")
plt.scatter(hc_x[hc_y_pred== 1,0], hc_x[hc_y_pred== 1,1], s=100, c= "blue", label= "c2")
plt.scatter(hc_x[hc_y_pred== 2,0], hc_x[hc_y_pred== 2,1], s=100, c= "green", label= "c3")
plt.scatter(hc_x[hc_y_pred== 3,0], hc_x[hc_y_pred== 3,1], s=100, c= "cyan", label= "c4")
plt.scatter(hc_x[hc_y_pred== 4,0], hc_x[hc_y_pred== 4,1], s=100, c= "magenta", label= "c5") 
plt.title("Hierarichal Clustering")
plt.xlabel("Annual Income(k$)")
plt.ylabel("Spending Index(1-100)")
plt.show()
  
#Association Rule Learning
# Apriori Model
ap_data= pd.read_csv("Market_Basket_Optimisation.csv", header= None)
transaction= []
for i in range(0, 7501):
    transaction.append([str(ap_data.values[i, j]) for j in range(0,20)])
#training Apriori model on the dataset
from apyori import apriori
rules= apriori(transaction, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)
result= list(rules)
#print(result[0])

#Reinforement Learning

#Upper Confidence Bound model
ucb_data= pd.read_csv("Ads_CTR_Optimisation.csv")
#random sampling to show Ads to each user
import random
N= range(0, 10000)
ads_selected=[]
total_reward=0
for i in N:
    ads= random.randrange(10)
    ads_selected.append(ads)
    reward= ucb_data.values[i, ads]
    total_reward= total_reward+ reward
#visualization
plt.hist(ads_selected)
plt.title("Histogram of Ads selection")
plt.xlabel("Ads")
plt.ylabel("frequency")    
#Upper Confidence Bound
import math
N=10000
d=10
ads_created=[]
sample_selection_count=[0]*d
sum_of_rewards= [0]*d
total_reward=0
for n in range(0,N):
    ad=0
    max_upper_bound=0
    for i in range(0, d):
        if sample_selection_count[i] > 0:
            av_reward= sum_of_rewards[i]/sample_selection_count[i]
            #print(av_reward)
            delta= math.sqrt((3/2)*(math.log(n+1)/sample_selection_count[i]))
            upper_bound= av_reward + delta
        else:
            upper_bound= 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound= upper_bound
            ad=i
    ads_created.append(ad)
    reward= ucb_data.values[n, ad]
    sum_of_rewards[ad]= sum_of_rewards[ad]+ reward
    sample_selection_count[ad]= sample_selection_count[ad] + 1
    total_reward= total_reward + reward   

#visualization of ads selection
plt.hist(ads_created)
plt.title("Histogram of Ads selection using UCB")
plt.xlabel("Ads")
plt.ylabel("frequency")

#Thompson Sampling Model
import random 
ts_data= pd.read_csv("Ads_CTR_Optimisation.csv")
N=10000
d=10
ads_created=[]
number_of_reward_1= [0]*d
number_of_reward_0= [0]*d
total_rewards=0
for n in range(0, N):
    ad= 0
    max_random= 0
    for i in range(0, d):
        random_beta= random.betavariate(number_of_reward_1[i]+1, number_of_reward_0[i]+1)
        if random_beta> max_random:
            max_random= random_beta
            ad= i
    ads_created.append(ad)
    reward= ts_data.values[n, ad]
    total_rewards= reward+ total_rewards
    if reward== 1:
        number_of_reward_1[ad]= number_of_reward_1[ad] + 1
    else:
        number_of_reward_0[ad]= number_of_reward_0[ad] + 1
        
#visualization of Thompson Sampling Result
plt.hist(ads_created)
plt.title("Histogram of Ads selection using TS")
plt.xlabel("Ads")
plt.ylabel("frequency")

#Natural Language Processing

nlp_data= pd.read_csv("Restaurant_reviews.tsv", delimiter= "\t", quoting= 3)
#cleaning the data
import re
import nltk
from nltk.stem.porter import PorterStemmer 
#nltk.download("stopwords")
from nltk.corpus import stopwords
corpus=[]
for i in range(0,1000):
    review= re.sub("[^a-zA-Z]", " ", nlp_data["Review"][i])
    review= review.lower()
    review= review.split()
    ps= PorterStemmer()
    review= [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]#set is used as it is faster algorithm in case we need to do it for longer articles or books
    review= " ".join(review)
    corpus.append(review)
#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=1500)#keeps top 1500 words with maximum frequency
nlp_x= cv.fit_transform(corpus).toarray() 
nlp_y= nlp_data.iloc[:, 1].values 
#for self doubts
"""a=["ram is a good good guy", "she is a bad bad girl"]  
corpus_a=[] 
for i in a:
    review= re.sub("[^a-zA-Z]", " ", i)
    review= review.lower()
    review= review.split()
    ps= PorterStemmer()
    review= [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]#set is used as it is faster algorithm in case we need to do it for longer articles or books
    review= " ".join(review)
    corpus_a.append(review) 
cv_a= CountVectorizer()#keeps top 1500 words with maximum frequency
nlp_a= cv_a.fit_transform(corpus_a).toarray()"""  #this code gives frequency of words of each row  
#creating Classification Model on this data
from sklearn.model_selection import train_test_split
nlp_x_train, nlp_x_test, nlp_y_train, nlp_y_test= train_test_split(nlp_x, nlp_y, test_size= 0.2, random_state= 0)
#creating Naives Bayes Model on this data
from sklearn.naive_bayes import GaussianNB
nlp_nb= GaussianNB()
nlp_nb.fit(nlp_x_train, nlp_y_train)
nlp_y_pred_nb= nlp_nb.predict(nlp_x_test)
from sklearn.metrics import confusion_matrix
nlp_cm_nb= confusion_matrix(nlp_y_test, nlp_y_pred_nb)
nlp_nb_accuracy= sklearn.metrics.accuracy_score(nlp_y_test, nlp_y_pred_nb)
#creating Logistics Regression Model on this data
from sklearn.linear_model import LogisticRegression
nlp_lr= LogisticRegression(random_state=0)
nlp_lr.fit(nlp_x_train, nlp_y_train)
nlp_y_pred_lr= nlp_lr.predict(nlp_x_test)
nlp_cm_lr= confusion_matrix(nlp_y_test, nlp_y_pred_nb)
nlp_lr_accuracy= sklearn.metrics.accuracy_score(nlp_y_test, nlp_y_pred_lr)
#creating KNN Model on this data
from sklearn.neighbors import KNeighborsClassifier
nlp_knn= KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
nlp_knn.fit(nlp_x_train, nlp_y_train)
nlp_y_pred_knn= nlp_knn.predict(nlp_x_test)
nlp_cm_knn= confusion_matrix(nlp_y_test, nlp_y_pred_knn)
nlp_knn_accuracy= sklearn.metrics.accuracy_score(nlp_y_test, nlp_y_pred_knn)
#creating SVM Model on this data
from sklearn.svm import SVC
nlp_svc= SVC(kernel= "rbf", random_state=0)
nlp_svc.fit(nlp_x_train, nlp_y_train)
nlp_y_pred_svc= nlp_svc.predict(nlp_x_test)
nlp_cm_svc= confusion_matrix(nlp_y_test, nlp_y_pred_svc)
nlp_svc_accuracy= sklearn.metrics.accuracy_score(nlp_y_test, nlp_y_pred_svc)
nlp_prf_score= sklearn.metrics.precision_recall_fscore_support(nlp_y_test, nlp_y_pred_svc)
#creating Decision Tree Model on this data
from sklearn.tree import DecisionTreeClassifier
nlp_dc= DecisionTreeClassifier(criterion="entropy", random_state=0)
nlp_dc.fit(nlp_x_train, nlp_y_train)
nlp_y_pred_dc= nlp_dc.predict(nlp_x_test)
nlp_cm_dc= confusion_matrix(nlp_y_test, nlp_y_pred_dc)
nlp_dc_accuracy= sklearn.metrics.accuracy_score(nlp_y_test, nlp_y_pred_dc)
#creating Random Forest Model on this data
from sklearn.ensemble import RandomForestClassifier
nlp_rfc= RandomForestClassifier(n_estimators=10, criterion= "entropy", random_state=0)
nlp_rfc.fit(nlp_x_train, nlp_y_train)
nlp_y_pred_rfc= nlp_rfc.predict(nlp_x_test)
nlp_cm_rfc= confusion_matrix(nlp_y_test, nlp_y_pred_rfc)
nlp_rfc_accuracy= sklearn.metrics.accuracy_score(nlp_y_test, nlp_y_pred_rfc)

#Deep Learning
#Artificial Neural Network

ann_data= pd.read_csv("Churn_Modelling.csv")
ann_x= ann_data.iloc[:, 3:13].values
ann_y= ann_data.iloc[:, 13].values
#labelling categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ann_x1= LabelEncoder()
ann_x[:, 1] = ann_x1.fit_transform(ann_x[:, 1])
ann_x2= LabelEncoder()
ann_x[:, 2]= ann_x2.fit_transform(ann_x[:, 2])
ohe= OneHotEncoder(categorical_features= [1])
ann_x=  ohe.fit_transform(ann_x).toarray()
ann_x= ann_x[:, 1:]
#train test split
from sklearn.model_selection import train_test_split
ann_x_train, ann_x_test, ann_y_train, ann_y_test= train_test_split(ann_x, ann_y, test_size=0.2, random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
ann_sc= StandardScaler()
ann_x_train= ann_sc.fit_transform(ann_x_train)
ann_x_test= ann_sc.transform(ann_x_test)
#importing ANN library
import keras
from keras.models import Sequential
from keras.layers import Dense
ann_classifier= Sequential()#initialising ANN
#adding input layer and first hidden layer
ann_classifier.add(Dense(output_dim= 6, init= "uniform", activation= "relu", input_dim=11))
#adding the second hidden layer
ann_classifier.add(Dense(output_dim= 6, init= "uniform", activation= "relu"))
#adding the output layer
ann_classifier.add(Dense(1, kernel_initializer= "uniform", activation= "sigmoid"))
#compiling the ANN
ann_classifier.compile(optimizer= "adam", loss= "binary_crossentropy", metrics= ["accuracy"])
#fitting ANN to the trainning set
ann_classifier.fit(ann_x_train, ann_y_train, batch_size=10, epochs=100)
#predicting on test set
ann_y_pred= ann_classifier.predict(ann_x_test)
ann_y_pred= (ann_y_pred > 0.5)
from sklearn.metrics import confusion_matrix
ann_cm= confusion_matrix(ann_y_test, ann_y_pred)
ann_accuracy= sklearn.metrics.accuracy_score(ann_y_test, ann_y_pred)

#CNN Model

#importing the keras libraries and packages
import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#initializing the CNN
cnn_classifier= Sequential()
#step1- convolution
cnn_classifier.add(Conv2D(32, (3, 3), input_shape= (64, 64, 3), activation= "relu"))
#step2- pooling
cnn_classifier.add(MaxPooling2D(pool_size= (2,2)))

#adding second convolution layer
cnn_classifier.add(Conv2D(32, (3, 3), activation= "relu"))
cnn_classifier.add(MaxPooling2D(pool_size= (2,2)))

#step3- Flattening
cnn_classifier.add(Flatten())
#step4- Full Connection
cnn_classifier.add(Dense(activation= "relu", units= 128))
cnn_classifier.add(Dense(activation= "sigmoid", units= 1))
#compiling
cnn_classifier.compile(optimizer= "adam", loss= "binary_crossentropy", metrics= ["accuracy"])
 #fitting CNN to the images 
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

cnn_classifier.fit_generator(training_set,
                             steps_per_epoch=8000,
                             epochs=25,
                             validation_data= test_set,
                             validation_steps=2000)

#Dimensionality Reduction

#PCA in Logistic Regression
pca_data= pd.read_csv("Wine.csv")
pca_x= pca_data.iloc[:, 0:13].values
pca_y= pca_data.iloc[:, 13].values
#train_test split
from sklearn.model_selection import train_test_split
pca_x_train, pca_x_test, pca_y_train, pca_y_test= train_test_split(pca_x, pca_y, test_size=0.2, random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
pca_sc= StandardScaler()
pca_x_train= pca_sc.fit_transform(pca_x_train)
pca_x_test= pca_sc.transform(pca_x_test)
#Applying PCA
from sklearn.decomposition import PCA
pca= PCA(n_components= 2)
pca_x_train= pca.fit_transform(pca_x_train)
pca_x_test= pca.transform(pca_x_test)
explained_variance= pca.explained_variance_ratio_
#fitting logistic regression
from sklearn.linear_model import LogisticRegression
pca_classifier= LogisticRegression(random_state=0)
pca_classifier.fit(pca_x_train, pca_y_train)
#predicting
pca_y_pred= pca_classifier.predict(pca_x_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
pca_cm= confusion_matrix(pca_y_test, pca_y_pred)
pca_aacuracy= sklearn.metrics.accuracy_score(pca_y_test, pca_y_pred)
#visualizing the training set result
from matplotlib.colors import ListedColormap
x_set, y_set= pca_x_train, pca_y_train
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
"""a,b=np.meshgrid(np.arange(start=0, stop=4, step=1),
                np.arange(start=0, stop=2, step=1))"""
plt.contourf(x1,x2, pca_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green", "blue")))
#c= np.array([[1,2,3], [2,3,4]])
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green", "blue"))(i), label=j)
plt.title("LR(Train_Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()
#visualizing the test set result
from matplotlib.colors import ListedColormap
x_set, y_set= pca_x_test, pca_y_test
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
"""a,b=np.meshgrid(np.arange(start=0, stop=4, step=1),
                np.arange(start=0, stop=2, step=1))"""
plt.contourf(x1,x2, pca_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green", "blue")))
#c= np.array([[1,2,3], [2,3,4]])
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green", "blue"))(i), label=j)
plt.title("LR(Train_Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

#LDA in Logistic Regression
lda_data= pd.read_csv("Wine.csv")
lda_x= lda_data.iloc[:, 0:13].values
lda_y= lda_data.iloc[:, 13].values
#train_test split
from sklearn.model_selection import train_test_split
lda_x_train, lda_x_test, lda_y_train, lda_y_test= train_test_split(lda_x, lda_y, test_size=0.2, random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
lda_sc= StandardScaler()
lda_x_train= lda_sc.fit_transform(lda_x_train)
lda_x_test= lda_sc.transform(lda_x_test)
#Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda= LDA(n_components= 2)
lda_x_train= lda.fit_transform(lda_x_train, lda_y_train)
lda_x_test= lda.transform(lda_x_test)
#fitting logistic regression
from sklearn.linear_model import LogisticRegression
lda_classifier= LogisticRegression(multi_class= "auto", random_state=0)
lda_classifier.fit(lda_x_train, lda_y_train)
#predicting
lda_y_pred= lda_classifier.predict(lda_x_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
lda_cm= confusion_matrix(lda_y_test, lda_y_pred)
lda_aacuracy= sklearn.metrics.accuracy_score(lda_y_test, lda_y_pred)
#visualizing the training set result
from matplotlib.colors import ListedColormap
x_set, y_set= lda_x_train, lda_y_train
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
"""a,b=np.meshgrid(np.arange(start=0, stop=4, step=1),
                np.arange(start=0, stop=2, step=1))"""
plt.contourf(x1,x2, lda_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green", "blue")))
#c= np.array([[1,2,3], [2,3,4]])
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green", "blue"))(i), label=j)
plt.title("LR(Train_Set)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()
#visualizing the test set result
from matplotlib.colors import ListedColormap
x_set, y_set= lda_x_test, lda_y_test
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
"""a,b=np.meshgrid(np.arange(start=0, stop=4, step=1),
                np.arange(start=0, stop=2, step=1))"""
plt.contourf(x1,x2, lda_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green", "blue")))
#c= np.array([[1,2,3], [2,3,4]])
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green", "blue"))(i), label=j)
plt.title("LR(Train_Set)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()

#Kernel PCA
lr_data= pd.read_csv("Social_Network_Ads.csv")
lr_x= lr_data.iloc[:, [2,3]].values
lr_y= lr_data.iloc[:,4].values
#train_test split
from sklearn.model_selection import train_test_split
lr_x_train, lr_x_test, lr_y_train, lr_y_test= train_test_split(lr_x, lr_y, test_size=0.25, random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
lr_sc= StandardScaler()
lr_x_train= lr_sc.fit_transform(lr_x_train)
lr_x_test= lr_sc.transform(lr_x_test)
#applyfing Kernel PCA
from sklearn.decomposition import KernelPCA
kpca= KernelPCA(n_components=2, kernel= "rbf")
lr_x_train= kpca.fit_transform(lr_x_train)
lr_x_test= kpca.transform(lr_x_test)
#fitting logistic regression
from sklearn.linear_model import LogisticRegression
lr_classifier= LogisticRegression(random_state=0)
lr_classifier.fit(lr_x_train, lr_y_train)
#predicting
lr_y_pred= lr_classifier.predict(lr_x_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
lr_cm= confusion_matrix(lr_y_test, lr_y_pred)
#visualizing the training set result
from matplotlib.colors import ListedColormap
x_set, y_set= lr_x_train, lr_y_train
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
"""a,b=np.meshgrid(np.arange(start=0, stop=4, step=1),
                np.arange(start=0, stop=2, step=1))"""
plt.contourf(x1,x2, lr_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green")))
#c= np.array([[1,2,3], [2,3,4]])
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green"))(i), label=j)
plt.title("LR(Train_Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()
#visualizing the test set result
from matplotlib.colors import ListedColormap
x_set, y_set= lr_x_test, lr_y_test
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:, 0].max()+1, step=0.01),
                   np.arange(start= x_set[:, 1].min()-1, stop= x_set[:,1].max()+1, step=0.01))
"""a,b=np.meshgrid(np.arange(start=0, stop=4, step=1),
                np.arange(start=0, stop=2, step=1))"""
plt.contourf(x1,x2, lr_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap= ListedColormap(("red", "green")))
#c= np.array([[1,2,3], [2,3,4]])
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(("red", "green"))(i), label=j)
plt.title("LR(Train_Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

#Model Selection

#k-Fold Cross Validation and Grid Search on SVM Model
svm_data= pd.read_csv("Social_Network_Ads.csv")
svm_x= svm_data.iloc[:, [2,3]].values
svm_y= svm_data.iloc[:,4].values
#train_test split
from sklearn.model_selection import train_test_split
svm_x_train, svm_x_test, svm_y_train, svm_y_test= train_test_split(svm_x, svm_y, test_size=0.25, random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
svm_sc= StandardScaler()
svm_x_train= svm_sc.fit_transform(svm_x_train)
svm_x_test= svm_sc.transform(svm_x_test)
#fitting logistic regression
from sklearn.svm import SVC
svm_classifier= SVC(kernel= "rbf", random_state=0)
svm_classifier.fit(svm_x_train, svm_y_train)
#predicting
svm_y_pred= svm_classifier.predict(svm_x_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
svm_cm= confusion_matrix(svm_y_test, svm_y_pred)
#applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(svm_classifier, X= svm_x_train, y= svm_y_train, cv= 10)
accuracies.mean()
accuracies.std()
#applying Grid Search
from sklearn.model_selection import GridSearchCV
parameters= [{"C": [1,10,100,1000], "kernel":["linear"]},
             {"C": [1,10,100,1000], "kernel":["rbf"], "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
            ]
grid_search= GridSearchCV(estimator= svm_classifier,
                          param_grid= parameters,
                          scoring= "accuracy",
                          cv=10,
                          n_jobs= -1)
grid_search= grid_search.fit(svm_x_train, svm_y_train)
best_accuracy= grid_search.best_score_
best_parameters= grid_search.best_params_

#XGBoost

#reading data
xgb_data= pd.read_csv("Churn_Modelling.csv")
xgb_x= xgb_data.iloc[:, 3:13].values
xgb_y= xgb_data.iloc[:, 13].values
#labelling categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
xgb_x1= LabelEncoder()
xgb_x[:, 1] = xgb_x1.fit_transform(xgb_x[:, 1])
xgb_x2= LabelEncoder()
xgb_x[:, 2]= xgb_x2.fit_transform(xgb_x[:, 2])
ohe= OneHotEncoder(categorical_features= [1])
xgb_x=  ohe.fit_transform(xgb_x).toarray()
xgb_x= xgb_x[:, 1:]
#train test split
from sklearn.model_selection import train_test_split
xgb_x_train, xgb_x_test, xgb_y_train, xgb_y_test= train_test_split(xgb_x, xgb_y, test_size=0.2, random_state=0)
#fitting xgboost model
from xgboost import XGBClassifier
xgb_classifier= XGBClassifier()
xgb_classifier.fit(xgb_x_train, xgb_y_train)
#predicting the results
xgb_y_pred= xgb_classifier.predict(xgb_x_test)
#confusion metrix
from sklearn.metrics import confusion_matrix
xgb_cm= confusion_matrix(xgb_y_test, xgb_y_pred)
xgb_accuracy= sklearn.metrics.accuracy_score(xgb_y_test, xgb_y_pred)
# k-Fold cross validation
from sklearn.model_selection import cross_val_score
xgb_accuracies= cross_val_score(estimator= xgb_classifier, 
                                X= xgb_x_train, 
                                y= xgb_y_train,
                                cv=10)
xgb_accuracies.mean()
xgb_accuracies.std()



















 