import os
# import kagglehub
# from kagglehub import KaggleDatasetAdapter
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
file_path = "netflix_titles.csv"
print("File path:", file_path)

df = pd.read_csv(file_path, sep=',', header=0, index_col=None, usecols=None, dtype=None, na_values=['', 'NA'], parse_dates=['date_added'])
print(df.columns)
print("First 5 records:", df.head(1))

# show_id type title director cast country date_added release_year rating duration listed_in description

