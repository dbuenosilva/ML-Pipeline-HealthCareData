# -*- coding: utf-8 -*-

##########################################################################
# Project: COMP6004 - Machine learning pipeline for data analysis
# File: 02-pre-processing.py
# Author: Vanessa Gomes - v.gomes.da.silva.10@student.scu.edu.au 
# Date: 20/04/2021
# Description: 
#
##########################################################################
# Maintenance                            
# Author:                         
# Date:                                                         
# Description:         
#
##########################################################################>

#Libraries used to pre-processing
import pandas as pd 
import numpy as np
from numpy import unique
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from functions import openfile
from functions import savefile

#calling the function to Load database
db = openfile('data/hotel_bookings.csv')

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

# indicate if there is duplicate data with - True or false
print('Is there duplicate data?= ', dups.any())

# show the list of duplicate rows
print('Show duplicat rowns= ',db[dups])
db.drop_duplicates(inplace=True)
print(db.shape)

# Using the function .mode to insert into the 
db["country"].fillna(db["country"].mode().to_string(), inplace=True)

# Drop rown when Conuntry is null 
#db.dropna(subset=["country"], inplace=True)

######## TRY to set the (value that appears most often)#######################
## If no id of agent or company is null, just replace it with 0
#db[[23,13,10]] = db[[23,13,10]].fillna(0.0)
db[["agent","country","children"]] = db[["agent","country","children"]].fillna(0.0)

## Missing values in the country column, replace it with mode (value that appears most often)
#db[13].fillna(db[13].mode().to_string(), inplace=True)
######################################################

# Delete the column agent that has no exprecive value for our classifocation
#db.drop(columns=23, inplace=True, axis=1)
db.drop(columns="agent", inplace=True, axis=1)

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
print(db.head())

#Get the X and Y for test and training
[r, c] = db.shape 

def transform(dataframe):
    
    
    ## Import LabelEncoder from sklearn
    
    le = LabelEncoder()
    
    
    ## Select all categorcial features
    categorical_features = list(dataframe.columns[dataframe.dtypes == object])
    
    
    ## Apply Label Encoding on all categorical features
    return dataframe[categorical_features].apply(lambda x: le.fit_transform(x.astype(str)), axis=0, result_type='expand')

df = transform(db)

X = df.iloc[:,:-1].values 
y = df.iloc[:,-1].values 



print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

print(X_train.shape, y_train.shape)

# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)



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


## Saving current DF to CSV to step03
savefile(df,"data/step02.csv")




