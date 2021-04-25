
##########################################################################
# Project: COMP6004 - Machine learning pipeline for data analysis
# File: 01-readingData.py
# Author: Diego Bueno - d.bueno.da.silva.10@student.scu.edu.au 
# Date: 20/04/2021
# Description: Read a CSV file and import it to a SQL database.
#
##########################################################################
# Maintenance                            
# Author: Vanessa Silva -                             
# Date: 23/04/2021                                                            
# Description:  Added function           
#
##########################################################################>

import sys
import pathlib
from pandas import read_csv

path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)
from functions import DataFrameToSqlDb

print("Current local path:")
print(path)
print("\nReding CSV file...")
myCsv = path + 'data/hotel_bookings.csv'
myDf = read_csv(myCsv)

print("\nReding CSV file...")
DataFrameToSqlDb(myDf,'step01')

print("Recording raw data into step01")
print(myDf.head())