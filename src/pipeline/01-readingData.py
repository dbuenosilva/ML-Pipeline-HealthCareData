# Libraries used to read the data base

from pandas import read_csv

path= 'C:/users/vanessa/Documents/GitHub/ML-Pipeline-Hotel-booking-demand/data/hotel_bookings.csv'
db = read_csv(path,  encoding='unicode_escape', header=None)

print(db.shape)