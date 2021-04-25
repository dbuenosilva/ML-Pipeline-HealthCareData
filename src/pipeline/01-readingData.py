# Function to open data base

# Libraries used to read the data base

from pandas import read_csv

def openfile(path):
    db_file = read_csv(path,  encoding='unicode_escape', header=None)
    print(db_file)
    return db_file

