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
# Date:  18/04/2021                                                       
# Description: Adding conditions to evaluate the action ( drop, insert, etc)      
#
##########################################################################>
# Maintenance                            
# Author: Diego Bueno                         
# Date:  26/04/2021                                                       
# Description: Adding ML model to predic [agent] missing values     
#
##########################################################################>

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error

from functions import openfile
from functions import savefile
from functions import convert

from machineLearning import getMachineLearningModel

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
 

print("\nFilling in often values into null registers of [country] column...")     
# Using the function .mode to insert the most often values into the null rowns 
db["country"].fillna(db["country"].mode().to_string(), inplace=True)

print("\nFilling in rounded mean values into null registers of [children] column...")     
#Insert rounded mean value to the null values rowns
db["children"].fillna(round(db["children"].mean()), inplace=True)

print("\nReviewing NULL values in each column:\n")
nullReport = np.sum(db.isnull())

# Print imformation about null data in the column
for index, value in nullReport.items():
    if value > 0:
        print("[" + str(index) + "] contains " + str(value) + " null registers (" + str( round(value/rows*100,3) )  + "%)" )    
   
    
print("\nFilling in predicted values into null registers of [agent] column" 
          " according to the Country, Market Segment and Distribution Channel:")     

# Creating a dataFrame with 
#   [country] => db.iloc[:,14:15]
#   [market_segment]	  => db.iloc[:,15:16]
#   [distribution_channel]  => db.iloc[:,16:17]
#   [agent]   => db.iloc[:,24:25]

# Converting [country], [market_segment] and	[distribution_channel]
print("\n01 step => Converting categorical values to numerical classes...\n")
numeralClassesCountryDf = convert( db.iloc[:,14:17] ) 
onlyAgentsDF = db.iloc[ :,24:25]  #[agent] 

print("\n02 step => Creating a Training data frame with only [country] and [agent] columns...\n")
myTrainingDf = pd.concat( [ numeralClassesCountryDf,  onlyAgentsDF], axis=1 )  

print("\n03 step => Removing Null columns for [agent] on the Training data frame...\n")
myTrainingDf = myTrainingDf.dropna()

print("Checking the current shape of the training data:")
rows, columns = myTrainingDf.shape
print( str(rows) + " rows and " + str(columns) + " columns")

print("\n04 step => Summarizing [country], [market_segment], [distribution_channel] and [agent] on the Training data frame...\n")
myTrainingDf = myTrainingDf.reset_index(drop=True) #removing index column
myTrainingDf.drop_duplicates(inplace=True)

print("\nChecking the current shape of the training data:")
rows, columns = myTrainingDf.shape
print( str(rows) + " rows and " + str(columns) + " columns")

print("\n05 step => Final training DF for Machine Learning use on predicts [agent] :\n")
print(myTrainingDf)

print("\n06 step => Defining the best Machine Learning model...\n")
myModel = getMachineLearningModel(myTrainingDf, posX = 0, posY = 1)

print(myModel.coef_)

print("\n07 step => Infering agent values ...\n")

#country-market_segment-distribution_channel-of-null-agents
poly = PolynomialFeatures(degree = myModel.coef_).fit_transform()
X_poly = poly.fit_transform(numeralClassesCountryDf.values)

#X_poly = myModel.transform(numeralClassesCountryDf.values)
predicted_agents = myModel.predic(X_poly)

#print(predicted_agents)

#db["agent"].fillna(myModel.predict(db["agent"]), inplace=True)    



## Saving current DF to CSV to step03
#print("\nRecording step02.scv file...")
#savefile(db,"data/step02.csv")
#print("done! \n\nstep02.csv file is ready for feature extraction task\n")







