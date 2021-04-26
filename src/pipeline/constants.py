#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################################################################
# Project: COMP6004 - Machine learning pipeline for data analysis
# File: constants.py
# Author: Diego Bueno - d.bueno.da.silva.10@student.scu.edu.au 
# Date: 20/04/2021
# Description: Constants variables used on ML pipeline project.
#
##########################################################################
# Maintenance                            
# Author: Vanessa Silva -                             
# Date: 23/04/2021                                                            
# Description:  Added constant           
#
##########################################################################>


### SQL Server connection
serverInstance = 'gwayaanalytics.database.windows.net,1433/gwayaanalytics' 
testServer = 'gwayaanalytics.database.windows.net\\gwayaanalytics' 
server = 'gwayaanalytics.database.windows.net'  
database = 'COMP6004' 
username = 'scu' 
password = 'UaKiQpg%nZGDAyYkivUq6s59%#K5f^L9@@5RjNKpmkfu9cf!kM^SXxK' 
DB_CONNECTION = 'Driver={SQL Server};SERVER='+serverInstance+';DATABASE='+database+';UID='+username+';PWD='+ password+';Trusted_Connection=yes;'
DB_STR_CONNECTION = 'mssql+pymssql://' + username + ':' + password + '@' + server + '/'+database

