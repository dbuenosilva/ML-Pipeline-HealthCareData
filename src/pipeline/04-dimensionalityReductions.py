# -*- coding: utf-8 -*-

##########################################################################
# Project: COMP6004 - Machine learning pipeline for data analysis
# File: 04-dimensionalityReductions.py
# Author: Vanessa Gomes - v.gomes.da.silva.10@student.scu.edu.au 
# Date: 20/04/2021
# Description: Libraries used to dimensionality reduction process
#
##########################################################################
# Maintenance                            
# Author: 
# Date:  
# Description: 
#
##########################################################################>


import numpy as np
from functions import openfile
from numpy import set_printoptions
from functions import savefile
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier


#Name of columns for the database
#name:[arrival_date_month	meal	country	market_segment	distribution_channel	reserved_room_type	assigned_room_type	customer_type

# Open datavase saved with preprocessing alterations 

db = openfile('data/step03.csv')

print(db.shape)

# Get the features
array = db.values
X = array[:,0:8]
Y = array[:,8]


########################## PCA Method #####################################

# PCA feature extraction
pca = PCA(n_components=4)# Select the most important features
fit = pca.fit(X) 
# summarize components
print("\nExplained Variance Ratio : %s" % fit.explained_variance_ratio_)
print('\nFit components for PCA ;',fit.components_)
features = fit.transform(X)
print('\nFeature - Fit Transform for X -PCA : ',features)


########### Universal Statistical Tests ###################################
# Extraction
test = SelectKBest(score_func=f_classif, k=8) # the feature classification
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=4)
print('\nFit scores for UST ', fit.scores_)
features = fit.transform(X)
print('\nFeature of Fit Tranformatio for X UST= ',features)

# summarize selected features
print('\nFeature - Fit Transform for X -PCA : ',features[0,:])
print('\nPrint X = ',X)


######################## RFE ################################################
#RFE Straction

## Extraction
my_model = LogisticRegression(solver='liblinear')
rfe = RFE(my_model, 4)
fit = rfe.fit(X, Y)
print("\n\nQtd of Features: %d" % fit.n_features_)
print("\nSelected Features: %s" % fit.support_)
print("\nFeature Ranking: %s" % fit.ranking_)
print('\nFeature ;', features[0,:])
print('\nPrint X = ',X[0,:])

###################### Trees Classifier ################################
#Extra Trees Classifier

# Extraction
my_model = ExtraTreesClassifier(n_estimators=4) # The number of trees in the forest.
my_model.fit(X, Y)
print('\nFeature importance based o Trees Classif =', my_model.feature_importances_)
imfeat = np.argsort(my_model.feature_importances_)
print('\nImfeat - Trees Class =',imfeat)
features = my_model.n_outputs_
print('\nFeature ;', features)
print('\nPrint X = ',X)

# Bagged decision tree

# Extraction
my_model = ExtraTreesClassifier(n_estimators=4) # The number of trees in the forest.
my_model.fit(X, Y)
print('\nBagged decision tree =', my_model.feature_importances_)
features = my_model.n_outputs_
print('\nFeature ;', features)
print('\nPrint X = ',X)

# this useless column reached here!!!
db = db.drop(columns=["Unnamed: 0"])

print("Grouping by the more relevant feature: [arrival_date_month] ")
db = db.groupby(["arrival_date_month"]).sum()


print("\nThe new shape of the data:")
rows, columns = db.shape
print( str(rows) + " rows and " + str(columns) + " columns")

print("\nBrief summary of data:\n")
print(db.head(5))
## Saving current DF to CSV to step04
print("\nRecording step04.scv file...")
savefile(db,"data/step04.csv")
print("done! \n\nstep04.csv file is ready for Dimensionality Reduction task\n")


