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

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf # using Tensorflow 2.4
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
# import pyodbc => does not work on MacOS
import pymssql 
import sqlalchemy as sal
from sqlalchemy import create_engine
from sqlalchemy import exc
import datetime
import logging
from io import StringIO
import sys
import pathlib
from sklearn.preprocessing import LabelEncoder

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
        
    db_file = pd.read_csv(path + myCsv) #, encoding='unicode_escape', header=None)
    print(db_file)
    return db_file




##########################################################################
# Function: savefile 
# Author: Vanessa Gomes - v.gomes.da.silva.10@student.scu.edu.au 
# Date: 25/04/2021
# Description: Save a CSV to Pandas Dataframe 
# 
# Parameters: myNewCsv - a CSV File
#             
# 
# Return: db_file - a DataFrame.
#
##########################################################################

#Function to open file
def savefile(df, myNewCsv):
        
    
    df.to_csv(path + myNewCsv) 
  
    return 

###########################################################################
#Function> convert

# Author: Vanessa Gomes - v.gomes.da.silva.10@student.scu.edu.au 
# Date: 25/04/2021

# Description: Convert the string columns to numrical to be used to Training and test
# 
# Parameters: database - database file
#             
# 
# Return: databse tranformed into columns with numbers 
###########################################################################
#Functin to convert strings into numbers
    
def convert(database):

    le = LabelEncoder()
    
    
    ## Select all categorcial features
    c_features = list(database.columns[database.dtypes == object])
    
    
    ## Apply Label Encoding on all categorical features
    return database[c_features].apply(lambda x: le.fit_transform(x.astype(str)), axis=0, result_type='expand')


###########################################################################










""" Function read( file_name  )

    Read a pickle file format  
    and return a Python dictionary with its content.

    parameters: (String) file_name

    return: 
        dict: a dictionary with the content encoding in bytes
    
"""
def read(file_name):
    with open(file_name, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



""" Function getMyEarlyStop( myMonitor, myPatience,  myModelFile )

    Set callback functions to early stop training, save the best model  
    to disk and return it as array.

    parameters: (String) myMonitor - metric name chose for monitor
                   (int) myPatience - number of epochs to interrupt in case 
                                      there is no longer improvement
                (String) myModelFile - path including file name to save the 
                                      best model.

    return: 
        callbacks: array with callback functions
    
"""
def getMyEarlyStop( myMonitor = "", myPatience = 0, myModelFile = "" ):
    
    if not myMonitor or myPatience <= 0 or not myModelFile:
        print("Invalid parameters!")
        return []
    
    callbacks = [EarlyStopping(monitor=myMonitor, patience=myPatience, mode='auto'),
    ModelCheckpoint(filepath=myModelFile, monitor=myMonitor, save_best_only=True, verbose=1)]
    return callbacks                 



""" Function saveResultToFile(   )

    Save results
    
    parameters:

    return: 
        none
    
"""
def saveToFile(file , contend ):

    try:
        f = open(file, "a")
        f.write(contend)
        f.close()
    except:
        print("\nError to save contend " + contend + " to file " + file + "!" )



