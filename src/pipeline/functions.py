#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################################################################
# Project: COMP6004 - Machine learning pipeline for data analysis
# File: functions.py
# Author: Diego Bueno - d.bueno.da.silva.10@student.scu.edu.au 
# Date: 20/04/2021
# Description: General functions for ML pipeline project.
#
##########################################################################
# Maintenance                            
# Author: Vanessa Silva -                             
# Date: 23/04/2021                                                            
# Description:  Added function           
#
##########################################################################>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import datetime
import logging
import sys
import pathlib
import re
# import pyodbc => does not work on MacOS
import pymssql
import sqlalchemy as sal
from sqlalchemy import create_engine
from sqlalchemy import exc
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut


## Importing especific functions used in this program 
path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)
import constants

## Logging
log_file = path + "/debug.log"

logging.basicConfig(filename=log_file, level=logging.DEBUG)
#logging.debug(datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S') + "|DEBUG|" + " " )
#logging.info(datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S') + "|INFO|" + " " )
#logging.warning(datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S') + "|WARNING|" + "  " )
#logging.error(datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S') + "|ERROR|" + "  " )


##########################################################################
# Function: DataFrameToSqlDb 
# Author: Diego Bueno - d.bueno.da.silva.10@student.scu.edu.au 
# Date: 25/04/2021
# Description: Save a Pandas Dataframe into a SQL Server Database.
# 
# Parameters: myDf - a Pandas dataframe
#             tableName - table name to be created/updated in SQL DB
# 
# Return: lError - True if error occurs, false instead of.
#
##########################################################################

def DataFrameToSqlDb(myDf,tableName):

    logging.info(datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S') + "|INFO| " + "Running DataFrameToSqlDb function... " )

    lError = False

    if not myDf.empty and tableName:
        try:
            # Create an in-memory SQLite database.
            engine = sal.create_engine(constants.DB_STR_CONNECTION)

            # Insert the dataframe into MSSQL       
            myDf.to_sql(tableName, con=engine, if_exists='replace', index=False)
            logging.info(datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S') + "|INFO|" + "Data frame inserted to SQL DB with table name " + tableName) 

        except exc.DBAPIError as err:
            logging.error(datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S') + "|ERROR|" + "Error to connect to database to insert/update dataframe " + tableName + "\n" + str(err) ) 
            lError = True
    else:
        logging.warning(datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S') + "|WARNING|" + "Data frame " + tableName + "is empty." ) 
        lError = True
    
    return lError



##########################################################################
# Function: openfile 
# Author: Vanessa Gomes - v.gomes.da.silva.10@student.scu.edu.au 
# Date: 25/04/2021
# Description: Save a CSV to Pandas Dataframe 
# 
# Parameters: myCsv - a CSV File
#             
# 
# Return: db_file - a DataFrame.
#
##########################################################################

#Function to open file
def openfile(myCsv):

    #print("Loading file " + path + myCsv)
    #print(path + myCsv)

    db_file = pd.read_csv(path + myCsv) #, encoding='unicode_escape', header=None)
    return db_file


##########################################################################
# Function: savefile 
# Author: Diego Bueno - d.bueno.da.silva.10@student.scu.edu.au 
# Date: 25/04/2021
# Description: Save a CSV to Pandas Dataframe 
# 
# Parameters: myNewCsv - a CSV File
#             
# 
# Return: db_file - a DataFrame.
#
##########################################################################

def savefile(df, myNewCsv):
        
    
    df.to_csv(path + myNewCsv) 
  
    return 


###########################################################################
#Function: convert

# Author: Vanessa Gomes - v.gomes.da.silva.10@student.scu.edu.au 
# Date: 25/04/2021

# Description: Convert the string columns to numrical to be used to Training and test
# 
# Parameters: database - database file
#             
# 
# Return: databse tranformed into columns with numbers 
###########################################################################
    
def convert(database):

    le = LabelEncoder()   
    
    ## Select all categorcial features
    c_features = list(database.columns[database.dtypes == object])

    ## Apply Label Encoding on all categorical features
    return database[c_features].apply(lambda x: le.fit_transform(x.astype(str)), axis=0, result_type='expand')


#########################################################################
# Function: removeSpecialCharacteres 
# Author: Diego Bueno - d.bueno.da.silva.10@student.scu.edu.au
# Date: 25/04/2021
# Description: Remove special characters from a string 
# 
# Parameters: originalString - the original string to remove special char.
#             specialChars - Special caracteres to be removed.
# 
# Return: newString - The new cleanned string.
#
##########################################################################

def removeSpecialCharacteres( originalString, specialChars="[!@#$%ˆ&*()_+=\][{}|<>??/.,'˜`]"):
    return re.sub(specialChars,'',originalString ).strip().title()

