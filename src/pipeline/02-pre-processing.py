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
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error

from functions import openfile
from functions import savefile
from functions import convert

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

# Indicates if there is duplicate data with - True or false
print('Is there duplicate data?= ', dups.any())

# show the list of duplicate rows
print('Show duplicat rowns= ',db[dups])
db.drop_duplicates(inplace=True)
print(db.shape)

# Using the function .mode to insert the most often values into the null rowns 
db["country"].fillna(db["country"].mode().to_string(), inplace=True)

# Drop rown when Conuntry is null 
#db.dropna(subset=["country"], inplace=True)



# Delete the column agent that has no exprecive value for our classification
#db.drop(columns=23, inplace=True, axis=1)
db.drop(columns="agent", inplace=True, axis=1)

#Insert rounded mean value to the null values rowns
db["children"].fillna(round(db["children"].mean()), inplace=True)


#criar coluna
#db['arrival_date'] = pd.to_datetime(db[3].astype(str) + '/' + db[5].astype(str) + '/' + db[4].astype(str))

#deletar colunas.
# db.drop(columns=["arrival_date_week_number", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"],
#            inplace=True, axis=1)

print(db.shape)
print(np.sum(db.isnull()))
print(db.head())

#Get the X and Y for test and training

df = convert(db) # Used function that converted the columns with string values to number 

X = df.iloc[:,:-1].values 
y = df.iloc[:,-1].values 


print(X.shape, y.shape)

# Split the data to 80% training and 20% teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

print(X_train.shape, y_train.shape)

# fit the model
my_model = LinearRegression()
my_model.fit(X_train, y_train)

# evaluate model
Y = my_model.predict(X_test)

# show the mean absolute error
mae = mean_absolute_error(y_test, Y)
print('mae %.3f' % mae)



# Into the training dataset indenfing the outliers 
Loc_OF = LocalOutlierFactor() 
Y = Loc_OF.fit_predict(X_train)

# select all rows that are not outliers
mask_out = Y != -1
X_train, y_train = X_train[mask_out, :], y_train[mask_out]

# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)

# fit the model
my_model = LinearRegression()
my_model.fit(X_train, y_train)

# evaluate the model
Y = my_model.predict(X_test)

# show the mean absolute error
mae = mean_absolute_error(y_test, Y)
print('MAE not Outliers: %.3f' % mae)


## Saving current DF to CSV to step03
savefile(df,"data/step02.csv")




