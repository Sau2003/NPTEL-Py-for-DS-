import pandas as pd
import numpy as np

# Load necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
data_income = pd.read_csv('income.csv')

# Display the first few rows of the dataset
print(data_income.head())

# Make a copy of the dataset
data = data_income.copy()

# Get information about the dataset
print(data.info())

# Check for missing values in the dataset
print(data.isnull()) 

# Sum up the missing values in each column
print(data.isnull().sum())

# Get summary statistics for numerical variables
summary_num = data.describe()  
print(summary_num)

# Get summary statistics for categorical variables
summary_caty = data.describe(include="O")
print(summary_caty)

# Calculate the frequency of values in the 'JobType' column
print(data['JobType'].value_counts())

# Calculate the frequency of values in the 'occupation' column
print(data['occupation'].value_counts())  

# Find unique values in the 'JobType' column
print(np.unique(data['JobType']))

# Find unique values in the 'occupation' column
print(np.unique(data['occupation'])) 

# Handle missing values by replacing ' ?' with NaN
data = pd.read_csv('income.csv', na_values=" ?")
data.isnull().sum()

# Drop rows with missing values
data.dropna(axis=0, inplace=True)

subset_columns = ['age', 'capitalgain', 'capitalloss', 'hoursperweek']
data_subset = data[subset_columns]

# Calculate correlation matrix for the subset of variables
correlation = data_subset.corr()
print(correlation)     # Relationship bet the independent variables 


print(data.columns)

# Gender proportion table 
gender=pd.crosstab(index=data["gender"],
                   columns='count',
                   normalize=True)
print(gender)

# gender vs the salary status 
gender_salsat=pd.crosstab(index=data['gender'],
                          columns=data['SalStat'],
                          margins=True,
                          normalize='index')
print(gender_salsat)

# # Plot the count of values in the 'SalStat' column
# SalStat = data['SalStat'].value_counts().plot(kind='bar')
# plt.show()

# 75% people salary is greater than 50k and 25% people salary is less than 25k

# # Plot the histogram of the 'age' column
# sns.histplot(data['age'], bins=10, kde=False)
# plt.show()

# sns.boxplot(x='SalStat', y='age', data=data)

# # Display the median age for each 'SalStat' category
# print(data.groupby('SalStat')['age'].median())
# plt.show()

## people with 35-50 age are more likely to earn > 50000 USD p.a
## people with 25-35 age are more likely to earn <= 50000 USD p.a

JobType= sns.countplot(y=data['JobType'],hue = 'SalStat', data=data)
job_salstat =pd.crosstab(index = data["JobType"],columns = data['SalStat'], margins = True, normalize =  'index')  
round(job_salstat*100,1)

Education   = sns.countplot(y=data['EdType'],hue = 'SalStat', data=data)
EdType_salstat = pd.crosstab(index = data["EdType"], columns = data['SalStat'],margins = True,normalize ='index')  
round(EdType_salstat*100,1)


Occupation  = sns.countplot(y=data['occupation'],hue = 'SalStat', data=data)
occ_salstat = pd.crosstab(index = data["occupation"], columns =data['SalStat'],margins = True,normalize = 'index')  
round(occ_salstat*100,1)

# #*** Capital gain
# plt.figure(figsize=(10, 6))
# sns.histplot(data['capitalgain'], bins=10, kde=False)
# plt.title('Histogram of Capital Gain')
# plt.xlabel('Capital Gain')
# plt.ylabel('Frequency')
# plt.show()

# # Plot histogram for 'capitalloss'
# plt.figure(figsize=(10, 6))
# sns.histplot(data['capitalloss'], bins=10, kde=False)
# plt.title('Histogram of Capital Loss')
# plt.xlabel('Capital Loss')
# plt.ylabel('Frequency')
# plt.show()


# LOGISTIC REGRESSION

# Reindexing the salary status names to 0,1
data['SalStat']=data['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data['SalStat'])
new_data=pd.get_dummies(data, drop_first=True)

# Storing the column names 
columns_list=list(new_data.columns)
print(columns_list)

# Separating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

# Storing the output values in y
y=new_data['SalStat'].values
print(y)

# Storing the values from input features
x = new_data[features].values
print(x)

# Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

# Make an instance of the Model
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

# Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

# Confusion matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

# Calculating the accuracy
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)


print('Misclassified samples: %d' % (test_y != prediction).sum())

# LOGISTIC REGRESSION - REMOVING INSIGNIFICANT VARIABLES
# =============================================================================

# Reindexing the salary status names to 0,1
data['SalStat']=data['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data['SalStat'])

cols = ['gender','nativecountry','race','JobType']
new_data = data.drop(cols,axis = 1)

new_data=pd.get_dummies(new_data, drop_first=True)

# Storing the column names 
columns_list2=list(new_data.columns)
print(columns_list2)

# Separating the input names from data
features2=list(set(columns_list2)-set(['SalStat']))
print(features2)

# Storing the output values in y
y2=new_data['SalStat'].values
print(y2)

# Storing the values from input features
x2 = new_data[features2].values
print(x2)

# Splitting the data into train and test
train_x2,test_x2,train_y2,test_y2 = train_test_split(x2,y2,test_size=0.3, random_state=0)

# Make an instance of the Model
logistic2 = LogisticRegression()

# Fitting the values for x and y
logistic2.fit(train_x2,train_y2)

# Prediction from test data
prediction2 = logistic2.predict(test_x2)

# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y2 != prediction2).sum())