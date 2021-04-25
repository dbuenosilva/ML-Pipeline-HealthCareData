# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 19:19:55 2021

@author: vanessa silva
"""

#Libraries used to pre-processing
import pandas as pd 
from pandas import read_csv
import numpy as np
from numpy import unique
from sklearn.feature_selection import VarianceThreshold


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error


# path to open the file
path = path= 'C:/users/vanessa/Documents/GitHub/ML-Pipeline-Hotel-booking-demand/data/hotel_bookings.csv'

#Function to open file
def openfile(path):
    db_file = read_csv(path, encoding='unicode_escape', header=None)
    print(db_file)
    return db_file


#calling the function to Load database
db = openfile(path)


print(db.shape)
print(db.head())

# Get database information
db.info()


## process to cleaning data
# Print imformation about null data in the column
print(np.sum(db.isnull()))

#Remove the columns with more than 60% of null value
for v_col in db.columns:
    if np.sum(db[v_col].isnull())>(db.shape[0] * 0.6):
 
        db.drop(columns=v_col, inplace=True, axis=1)
print(db.shape)
print(np.sum(db.isnull()))


#Print duplicates
dups = db.duplicated()

# report if there are any duplicates
print('any duplicated= ', dups.any())
# list all duplicate rows
print('Num Duplic rowns= ',db[dups])
db.drop_duplicates(inplace=True)
print(db.shape)




# Delete rown when Conuntry is null 
db.dropna(subset=[13], inplace=True)

######## TRY to set the (value that appears most often)#######################
## If no id of agent or company is null, just replace it with 0
db[[23,13,10]] = db[[23,13,10]].fillna(0.0)

## Missing values in the country column, replace it with mode (value that appears most often)
#db[13].fillna(db[13].mode().to_string(), inplace=True)
######################################################

# Delete the column agent that has no exprecive value for our classifocation
db.drop(columns=23, inplace=True, axis=1)


## for missing children value, replace it with rounded mean value
#db[10].fillna(round(db[10].mean()), inplace=True)


# mode= db[10].mean()
# db[10]= db[10].fillna(mode)

#criar coluna
#db['arrival_date'] = pd.to_datetime(db[3].astype(str) + '/' + db[5].astype(str) + '/' + db[4].astype(str))


#deletar colunas.
# db.drop(columns=["arrival_date_week_number", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"],
#            inplace=True, axis=1)

print(db.shape)
print(np.sum(db.isnull()))


#Get the X and Y for test and training

[r, c] = db.shape 
X = db.iloc[:, 1:c-1].values # external force
y = db.iloc[:, c-1:c].values # velocity of charged particles


# # reshape and resize - already done with df.iloc[].values
# X = np.reshape(X, (r, 1))
# y = np.reshape(y, (r, 1)) 

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)

# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
'''
# identify outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
'''
