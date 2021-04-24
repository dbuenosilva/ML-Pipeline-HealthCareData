# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 19:19:55 2021

@author: vanessa
"""

import pandas as pd 
from pandas import read_csv
import numpy as np
from numpy import unique
from sklearn.feature_selection import VarianceThreshold


# load database


db = read_csv("hotel_bookings.csv", encoding='unicode_escape', header=None)
#df = read_csv("sudeste.csv", encoding='unicode_escape', header=None)
#df = read_csv("time_series_covid_19_confirmed.csv", header=None)

[r, c] = db.shape 
X = db.iloc[:, 1:c-1].values # external force
y = db.iloc[:, c-1:c].values # velocity of charged particles


# reshape and resize - already done with df.iloc[].values
# X = np.reshape(X, (r, 1))
# y = np.reshape(y, (r, 1)) 


print(db.shape)
#print(df.head())

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

