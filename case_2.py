import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

# setting the dimensions for the plot 
plt.figure(figsize=(11.7, 8.27))

cars_data = pd.read_csv("cars.csv")
cars = cars_data.copy()

cars.info() # structure of the dataset 
cars.describe() # summarizing the data 
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(cars.describe())

# To display maximum set of columns
pd.set_option('display.max_columns', 500)
print(cars.describe())

# Dropping unwanted columns
col = ['name', 'dateCrawled', 'dateCreated', 'postalCode', 'lastSeen']
cars = cars.drop(columns=col, axis=1)

# Removing duplicate records
cars.drop_duplicates(keep='first', inplace=True)
# 470 duplicate records

# data cleaning 
print(cars.isnull().sum())

# Working range of data
# =============================================================================

# Working range of data

cars = cars[
        (cars.yearOfRegistration <= 2018) 
      & (cars.yearOfRegistration >= 1950) 
      & (cars.price >= 100) 
      & (cars.price <= 150000) 
      & (cars.powerPS >= 10) 
      & (cars.powerPS <= 500)]
# ~6700 records are dropped

# Further to simplify- variable reduction
# Combining yearOfRegistration and monthOfRegistration

cars['monthOfRegistration']/=12

# Creating new varible Age by adding yearOfRegistration and monthOfRegistration
cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']=round(cars['Age'],2)
cars['Age'].describe()

# Dropping yearOfRegistration and monthOfRegistration
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'], axis=1)



# Visualizing parameters 

# Age
plt.subplot(2, 1, 1)
plt.hist(cars['Age'], bins=30, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.boxplot(cars['Age'])
plt.ylabel('Age')
plt.title('Boxplot of Age')
plt.grid(True)

plt.tight_layout()
plt.show()

# price
plt.subplot(2, 1, 1)
plt.hist(cars['price'], bins=30, edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Price')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.boxplot(cars['price'])
plt.ylabel('Price')
plt.title('Boxplot of Price')
plt.grid(True)

plt.tight_layout()
plt.show()

# powerPS
plt.subplot(2, 1, 1)
plt.hist(cars['powerPS'], bins=30, edgecolor='black')
plt.xlabel('PowerPS')
plt.ylabel('Frequency')
plt.title('Histogram of PowerPS')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.boxplot(cars['powerPS'])
plt.ylabel('PowerPS')
plt.title('Boxplot of PowerPS')
plt.grid(True)

plt.tight_layout()
plt.show()

# Visualizing parameters after narrowing working range
# Age vs price
plt.scatter(x=cars['Age'], y=cars['price'], alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Price')
plt.title('Scatter Plot of Price vs Age')
plt.grid(True)
plt.show()

# powerPS vs price
plt.scatter(x=cars['powerPS'], y=cars['price'], alpha=0.5)
plt.xlabel('PowerPS')
plt.ylabel('Price')
plt.title('Scatter Plot of Price vs PowerPS')
plt.grid(True)
plt.show()

# Variable seller
seller_counts = cars['seller'].value_counts()
plt.bar(seller_counts.index, seller_counts.values)
plt.xlabel('Seller')
plt.ylabel('Count')
plt.title('Count of Sellers')
plt.grid(True)
plt.show()

# Variable offerType
offerType_counts = cars['offerType'].value_counts()
plt.bar(offerType_counts.index, offerType_counts.values)
plt.xlabel('Offer Type')
plt.ylabel('Count')
plt.title('Count of Offer Types')
plt.grid(True)
plt.show()

# Variable abtest
abtest_counts = cars['abtest'].value_counts()
plt.bar(abtest_counts.index, abtest_counts.values)
plt.xlabel('AB Test')
plt.ylabel('Count')
plt.title('Count of AB Tests')
plt.grid(True)
plt.show()

# Variable vehicleType
vehicleType_counts = cars['vehicleType'].value_counts()
plt.bar(vehicleType_counts.index, vehicleType_counts.values)
plt.xlabel('Vehicle Type')
plt.ylabel('Count')
plt.title('Count of Vehicle Types')
plt.grid(True)
plt.show()

# Variable gearbox
gearbox_counts = cars['gearbox'].value_counts()
plt.bar(gearbox_counts.index, gearbox_counts.values)
plt.xlabel('Gearbox')
plt.ylabel('Count')
plt.title('Count of Gearboxes')
plt.grid(True)
plt.show()

# Variable model
model_counts = cars['model'].value_counts()
plt.figure(figsize=(12, 6))
plt.bar(model_counts.index, model_counts.values)
plt.xlabel('Model')
plt.ylabel('Count')
plt.title('Count of Models')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# Variable kilometer
plt.boxplot(cars['kilometer'])
plt.ylabel('Kilometer')
plt.title('Boxplot of Kilometer')
plt.grid(True)
plt.show()

plt.hist(cars['kilometer'], bins=8, edgecolor='black')
plt.xlabel('Kilometer')
plt.ylabel('Frequency')
plt.title('Histogram of Kilometer')
plt.grid(True)
plt.show()

plt.scatter(x=cars['kilometer'], y=cars['price'], alpha=0.5)
plt.xlabel('Kilometer')
plt.ylabel('Price')
plt.title('Scatter Plot of Price vs Kilometer')
plt.grid(True)
plt.show()

# Variable fuelType
fuelType_counts = cars['fuelType'].value_counts()
plt.bar(fuelType_counts.index, fuelType_counts.values)
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.title('Count of Fuel Types')
plt.grid(True)
plt.show()

# Variable brand
brand_counts = cars['brand'].value_counts()
plt.figure(figsize=(12, 6))
plt.bar(brand_counts.index, brand_counts.values)
plt.xlabel('Brand')
plt.ylabel('Count')
plt.title('Count of Brands')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# Variable notRepairedDamage
notRepairedDamage_counts = cars['notRepairedDamage'].value_counts()
plt.bar(notRepairedDamage_counts.index, notRepairedDamage_counts.values)
plt.xlabel('Not Repaired Damage')
plt.ylabel('Count')
plt.title('Count of Not Repaired Damages')
plt.grid(True)
plt.show()

# =============================================================================
# Removing insignificant variables
# =============================================================================

cars = cars.drop(columns=['seller', 'offerType', 'abtest'], axis=1)

#
# =============================================================================
# Correlation
# =============================================================================

cars_select1 = cars.select_dtypes(exclude=[object])
correlation = cars_select1.corr()
round(correlation, 3)   
cars_select1.corr().loc[:, 'price'].abs().sort_values(ascending=False)[1:]                          

# =============================================================================
# OMITTING MISSING VALUES
# =============================================================================

cars_omit = cars.dropna(axis=0)

# Converting categorical variables to dummy variables
cars_omit = pd.get_dummies(cars_omit, drop_first=True) 

# =============================================================================
# IMPORTING NECESSARY LIBRARIES
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# =============================================================================
# MODEL BUILDING WITH OMITTED DATA
# =============================================================================

# Separating input and output features
x1 = cars_omit.drop(['price'], axis='columns', inplace=False)
y1 = cars_omit['price']

# Plotting the variable price
prices = pd.DataFrame({"1. Before": y1, "2. After": np.log(y1)})
prices.hist()

# Transforming price as a logarithmic value
y1 = np.log(y1)

# Splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# =============================================================================
# BASELINE MODEL FOR OMITTED DATA
# =============================================================================

# finding the mean for test data value
base_pred = np.mean(y_test)
print(base_pred)

# Repeating same value till length of test data
base_pred = np.repeat(base_pred, len(y_test))

# finding the RMSE
base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
                               
print(base_root_mean_square_error)

# =============================================================================
# LINEAR REGRESSION WITH OMITTED DATA
# =============================================================================

# Setting intercept as true
lgr = LinearRegression(fit_intercept=True)

# Model
model_lin1 = lgr.fit(X_train, y_train)

# Predicting model on test set
cars_predictions_lin1 = lgr.predict(X_test)

# Computing MSE and RMSE
lin_mse1 = mean_squared_error(y_test, cars_predictions_lin1)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)

# R squared value
r2_lin_test1 = model_lin1.score(X_test, y_test)
r2_lin_train1 = model_lin1.score(X_train, y_train)
print(r2_lin_test1, r2_lin_train1)

# Regression diagnostics- Residual plot analysis
residuals1 = y_test - cars_predictions_lin1
plt.scatter(x=cars_predictions_lin1, y=residuals1, alpha=0.5)
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot of Linear Regression')
plt.grid(True)
plt.show()
residuals1.describe()

# =============================================================================
# RANDOM FOREST WITH OMITTED DATA
# =============================================================================

# Model parameters
rf = RandomForestRegressor(n_estimators=100, max_features='sqrt',
                           max_depth=100, min_samples_split=10,
                           min_samples_leaf=4, random_state=1)


# Model
model_rf1 = rf.fit(X_train, y_train)

# Predicting model on test set
cars_predictions_rf1 = rf.predict(X_test)

# Computing MSE and RMSE
rf_mse1 = mean_squared_error(y_test, cars_predictions_rf1)
rf_rmse1 = np.sqrt(rf_mse1)
print(rf_rmse1)

# R squared value
r2_rf_test1 = model_rf1.score(X_test, y_test)
r2_rf_train1 = model_rf1.score(X_train, y_train)
print(r2_rf_test1, r2_rf_train1)   

# =============================================================================
# MODEL BUILDING WITH IMPUTED DATA
# =============================================================================

cars_imputed = cars.apply(lambda x: x.fillna(x.median()) \
                  if x.dtype == 'float' else \
                  x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()

# Converting categorical variables to dummy variables
cars_imputed = pd.get_dummies(cars_imputed, drop_first=True) 


# =============================================================================
# MODEL BUILDING WITH IMPUTED DATA
# =============================================================================

# Separating input and output feature
x2 = cars_imputed.drop(['price'], axis='columns', inplace=False)
y2 = cars_imputed['price']


# Plotting the variable price
prices = pd.DataFrame({"1. Before": y2, "2. After": np.log(y2)})
prices.hist()

# Transforming price as a logarithmic value
y2 = np.log(y2)

# Splitting data into test and train
X_train1, X_test1, y_train1, y_test1 = train_test_split(x2, y2, test_size=0.3, random_state=3)
print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)


# =============================================================================
# BASELINE MODEL FOR IMPUTED DATA
# =============================================================================

# finding the mean for test data value
base_pred = np.mean(y_test1)
print(base_pred)

# Repeating same value till length of test data
base_pred = np.repeat(base_pred, len(y_test1))

# finding the RMSE
base_root_mean_square_error_imputed = np.sqrt(mean_squared_error(y_test1, base_pred))
                               
print(base_root_mean_square_error_imputed)
