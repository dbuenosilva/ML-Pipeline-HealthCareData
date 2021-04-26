# -*- coding: utf-8 -*-

##########################################################################
# Project: COMP6004 - Machine learning pipeline for data analysis
# File: 02-pre-processing.py
# Author: Vanessa Gomes - v.gomes.da.silva.10@student.scu.edu.au 
# Date: 20/04/2021
# Description: Libraries used to pre-processing
#
##########################################################################
# Maintenance                            
# Author: Diego Bueno                         
# Date:  26/04/2021                                                       
# Description: Adding conditions to evaluate the action ( drop, insert, etc)      
#
##########################################################################>

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error

from functions import openfile
from functions import savefile
from functions import convert

#calling the function to Load data pre-reading on task 1
print("\nReading the step01 file\n")
db = openfile('data/step01.csv')

print("\nChecking the current shape of the data:")
rows, columns = db.shape
print( str(rows) + " rows and " + str(columns) + " columns")

print("\nBrief summary of data:\n")
print(db.head(5))

print("\nGetting database information:\n\n")
#db.info() ### try to gropup by datatype

###################################################
#
#  Starting the process of cleaning data
#
###################################################
print("\nGetting NULL values in each column:\n")
nullReport = np.sum(db.isnull())

# Print imformation about null data in the column
for index, value in nullReport.items():
    if value > 0:
        print("[" + str(index) + "] contains " + str(value) + " null registers (" + str( round(value/rows*100,3) )  + "%)" )    
   
    
print("\nRemoving columns with more than 60% of null values:\n")   

for v_col in db.columns:
    if np.sum(db[v_col].isnull())>(rows * 0.6):
        db.drop(columns=v_col, inplace=True, axis=1)
        print("Column [" + v_col + "] removed!")        
   
rows, columns = db.shape
print("\nThe new shape of the data: " + str(rows) + " rows and " + str(columns) + " columns")
              
print("\nChecking duplicated values:") 
dups = db.duplicated()

if not dups.any(): # Indicates if there is duplicate data with - True or false    
    print('\nThere is no duplicate information in the dataset!')
else:
    # show the list of duplicate rows
    print('\nShowing duplicated rowns:')
    print(db[dups])

    print('\nDropping duplicated rowns:')
    db.drop_duplicates(inplace=True)
    
    rows, columns = db.shape
    print("\nThe new shape of the data: " + str(rows) + " rows and " + str(columns) + " columns")
 

print("\nFilling often values into null registers of [country] column...")     
# Using the function .mode to insert the most often values into the null rowns 
db["country"].fillna(db["country"].mode().to_string(), inplace=True)

print("\nFilling rounded mean values into null registers of [children] column...")     
#Insert rounded mean value to the null values rowns
db["children"].fillna(round(db["children"].mean()), inplace=True)

print("\nReviewing NULL values in each column:\n")
nullReport = np.sum(db.isnull())

# Print imformation about null data in the column
for index, value in nullReport.items():
    if value > 0:
        print(str(index) + " contains " + str(value) + " null registers (" + str( round(value/rows*100,3) )  + "%)" )    
   


""" 

# Delete the column agent that has no exprecive value for our classification
#db.drop(columns=23, inplace=True, axis=1)
db.drop(columns="agent", inplace=True, axis=1)


#criar coluna
#db['arrival_date'] = pd.to_datetime(db[3].astype(str) + '/' + db[5].astype(str) + '/' + db[4].astype(str))

#deletar colunas.
# db.drop(columns=["arrival_date_week_number", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"],
#            inplace=True, axis=1)


""" 

## Saving current DF to CSV to step03
print("\nRecording step02.scv file...")
savefile(db,"data/step02.csv")
print("done! \n\nstep02.csv file is ready for feature extraction task\n")







