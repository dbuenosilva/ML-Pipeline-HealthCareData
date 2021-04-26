
##########################################################################
# Project: COMP6004 - Machine learning pipeline for data analysis
# File: 01-readingData.py
# Author: Diego Bueno - d.bueno.da.silva.10@student.scu.edu.au 
# Date: 20/04/2021
# Description: Read a CSV file or SQL database and prepare it for 
#              pre-processing tasks.
#
##########################################################################
# Maintenance                            
# Author: 
# Date: 
# Description:  
#
##########################################################################>

import sys
import pathlib
import pandas as pd

path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)
#from functions import DataFrameToSqlDb
from functions import removeSpecialCharacteres
from functions import savefile

print("Current local path:")
print(path)

myCsv = path + 'data/hotel_bookings.csv'
print("\nReading the raw file " + myCsv)

myDf = pd.read_csv(myCsv,index_col=False)

print("\nRemoving special chars from text columns...")
hotelColumn = myDf[["hotel"]].applymap(removeSpecialCharacteres) 
marketSegmentColumn = myDf[["market_segment"]].applymap(removeSpecialCharacteres) 
distributionChannelColumn = myDf[["distribution_channel"]].applymap(removeSpecialCharacteres) 
print("done!")

# Remainder columns between [hotel] and [market segment]
remainderColumns1Part = myDf.iloc[:,1:14]  

# Remainder columns from [distribution_channel]
remainderColumns2Part = myDf.iloc[:,16:31]  

# Concatenating as original layout
myDf = pd.concat( [hotelColumn, remainderColumns1Part, marketSegmentColumn, 
                        distributionChannelColumn, remainderColumns2Part] , axis=1)  

print("\nRecording step01.scv file... ")
savefile(myDf,"data/step01.csv")
print("done! \n\nstep01.csv file is ready for futher pre-processing task\n")