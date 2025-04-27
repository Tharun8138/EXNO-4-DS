# EXNO:4-DS
## AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

## ALGORITHM:

STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

## FEATURE SCALING:

1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

## FEATURE SELECTION:

Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

## CODING AND OUTPUT:
```
Dev by : THARUN.V
Reg no : 212224230290
```
```
import pandas as pd
from scipy import stats
import numpy as

df=pd.read_csv("/content/bmi.csv")
df.head()
```

## output: 
![ex 4 s1](https://github.com/user-attachments/assets/80d4dd6b-d414-4062-ae7f-b69f602da959)

```
df_null_sum=df.isnull().sum()
df_null_sum
```
## output: 
![ex 4 s2](https://github.com/user-attachments/assets/464118d3-7c78-4044-b0b0-35f39a49aa9c)

```
df.dropna()
```
## output: 
![ex 4 s3](https://github.com/user-attachments/assets/b3f0ca50-461d-44f6-9a77-eed7547ff54d)

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
## output: 
![ex 4 s4](https://github.com/user-attachments/assets/0b93ea2c-ba39-429d-9e0f-72a8c72d5b0a)

```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```
## output: 
![ex 4 s5](https://github.com/user-attachments/assets/709bf354-e6a0-4a80-9a23-cbbb8aee6b51)

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
## output: 
![ex 4 s6](https://github.com/user-attachments/assets/30d1ccee-69ad-41f0-82b4-c129c2fd9fc0)

## MIN MAX SCALING:
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
## output: 
![ex 4 s7](https://github.com/user-attachments/assets/5358f107-c6d1-40e2-8dd2-414a2324bc66)


## MAXIMUM ABSOLUTE SCALING:
```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
```

## output: 
![ex 4 s8](https://github.com/user-attachments/assets/3dbc88f9-de9e-4321-b2bb-ade46f6c8260)

## ROBUST SCALING:
```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4=pd.read_csv("/content/bmi.csv")
df4.head()
```

## output: 
![ex 4 s9](https://github.com/user-attachments/assets/e4bfaec8-5ec8-4512-b42a-f1e49d433923)


## Feature Selection :
```
import pandas as pd
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
## output: 
![ex 4 s10](https://github.com/user-attachments/assets/4d695694-161e-4c7f-8e6d-1e4b79ca6294)

```
df_null_sum=df.isnull().sum()
df_null_sum
```
## output: 
![ex 4 s11](https://github.com/user-attachments/assets/e737a998-b914-419e-a639-0079ec467af6)

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
## output: 
![ex 4 s12](https://github.com/user-attachments/assets/0c270326-0b9c-4c5a-908a-a54a485e24bc)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
## output: 
![ex 4 s13](https://github.com/user-attachments/assets/615431f3-e941-482b-b43b-2ca14d5d4d43)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
## output: 
![ex 4 s14](https://github.com/user-attachments/assets/3cac87c1-feed-48f8-8e32-824abfa1ec6e)

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
## output: 
![ex 4 s15](https://github.com/user-attachments/assets/c52fc61e-9ecf-40e1-a8c5-f7addfaf58b7)


# FILTER METHOD:
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
## output: 
![ex 4 s16](https://github.com/user-attachments/assets/1a3b20d4-5e62-4b55-b3c8-db6d60c7d18d)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
## output: 
![ex 4 s17](https://github.com/user-attachments/assets/7efde1e3-66df-4ef6-bb4e-bacbf6b3497e)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
## output: 
![ex 4 s18](https://github.com/user-attachments/assets/efa36454-3f5b-4d65-8369-1f081aed3348)


## Model :
```
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
## output: 

![ex 4 s19](https://github.com/user-attachments/assets/751e956e-ccf1-4b54-9673-5cd07a86b8ee)
```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
## output: 
![ex 4 s20](https://github.com/user-attachments/assets/677df375-66c2-4569-a622-ad29d7a33c9b)

# @title
!pip install skfeature-chappers

## output: 
![ex 4 s21](https://github.com/user-attachments/assets/327f661b-5185-42c7-a4cd-25331c64a09f)

# @title
```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
# @title
```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
```
# @title
```
df[categorical_columns]
```
## output: 
![ex 4 s22](https://github.com/user-attachments/assets/800b2a53-5366-43ed-b6b2-78710612e2a6)


# @title
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
```
# @title
```
df[categorical_columns]
```
## output: 
![ex 4 s23](https://github.com/user-attachments/assets/eac844e1-ce60-423a-a72c-d1ca77c358ee)
```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
X_anova = selector_anova.fit_transform(X, y)

selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
## output:
![ex 4 s24](https://github.com/user-attachments/assets/b19aac54-0339-459f-b509-40be71e49ece)

# WRAPPER METHOD:
```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
## output: 
![ex 4 s25](https://github.com/user-attachments/assets/9367a707-c931-45f8-9eb6-ea832a482af6)
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
## output: 
![ex 4 s26](https://github.com/user-attachments/assets/e72c1bd4-ef1d-4197-a701-160fc2192a19)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select = 6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
## output: 
![ex 4 s27](https://github.com/user-attachments/assets/8ef05db4-c725-4916-8660-2de95d57d1f7)
```
selected_features = X.columns[rfe.support_]
print("Selected features using RFE:")
print(selected_features)
```

## output: 
![ex 4 s28](https://github.com/user-attachments/assets/dc355a0c-a6e7-4c58-a666-4aab2d48c9eb)

```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using Fisher Score selected features: {accuracy}")
```
## output: 
![ex 4 s29](https://github.com/user-attachments/assets/1f2c2886-d20c-4354-878f-4b746649d544)

# RESULT:

Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.

