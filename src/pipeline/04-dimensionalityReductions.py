# Libraryies used 

import numpy as np
from functions import openfile
from numpy import set_printoptions
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier


#Name of columns for the database
name:['hotel','is_canceled','lead_time','arrival_date_year','arrival_date_month','arrival_date_week_number','arrival_date_day_of_month','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies','meal','country','market_segment','distribution_channel','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','reserved_room_type','assigned_room_type','booking_changes','deposit_type','days_in_waiting_list','customer_type','adr','required_car_parking_spaces','total_of_special_requests','reservation_status','reservation_status_date']


# Open datavase saved with preprocessing alterations 

db = openfile('data/step02.csv')

print(db.shape)

# Get the features
array = db.values
X = array[:,0:11]
Y = array[:,11]


########################## PCA Method #####################################

# PCA feature extraction
pca = PCA(n_components=4)# Select the most important features
fit = pca.fit(X) 
# summarize components
print("Explained Variance Ratio : %s" % fit.explained_variance_ratio_)
print('Fit components for PCA ;',fit.components_)
features = fit.transform(X)
print('Feature - Fit Transform for X -PCA : ',features)


########### Universal Statistical Tests ###################################
# Extraction
test = SelectKBest(score_func=f_classif, k=10) # the feature classification
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=4)
print('Fit scores for UST ', fit.scores_)
features = fit.transform(X)
print('Feature of Fit Tranformatio for X UST= ',features)

# summarize selected features
print('Feature - Fit Transform for X -PCA : ',features[0,:])
print('Print X = ',X)


######################## RFE ################################################
#RFE Straction

## Extraction
my_model = LogisticRegression(solver='liblinear')
rfe = RFE(my_model, 4)
fit = rfe.fit(X, Y)
print("Qtd of Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
print('Feature ;', features[0,:])
print('Print X = ',X[0,:])

###################### Trees Classifier ################################
#Extra Trees Classifier

# Extraction
my_model = ExtraTreesClassifier(n_estimators=4) # The number of trees in the forest.
my_model.fit(X, Y)
print('Feature importance based o Trees Classif =', my_model.feature_importances_)
imfeat = np.argsort(my_model.feature_importances_)
print('Imfeat - Trees Class =',imfeat)
features = my_model.n_outputs_
print('Feature ;', features)
print('Print X = ',X)

# Bagged decision tree

# Extraction
my_model = ExtraTreesClassifier(n_estimators=4) # The number of trees in the forest.
my_model.fit(X, Y)
print('Bagged decision tree =', my_model.feature_importances_)
features = my_model.n_outputs_
print('Feature ;', features)
print('Print X = ',X)

