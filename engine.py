import os
# import kagglehub
# from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
file_path = "netflix_titles.csv"
print("File path:", file_path)

df = pd.read_csv(file_path, sep=',', header=0, index_col=None, usecols=None, dtype=None, na_values=['', 'NA'], parse_dates=['date_added'])
# df['year_added'] = pd.to_datetime(df['date_added'].apply(lambda x: x.strip() if isinstance(x, str) else x), format="%B %d, %Y").dt.year # Extract year from date_added
df['country'] = df['country'].fillna('United States') # Set United States as the default country
df = df.dropna(subset=['rating', 'description', 'duration']) # Remove columns where rating, description, or duration is null
df ['release_year'] = df['release_year'].astype(int) # Convert release_year to int
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")
print(f"Data types: {list(df['country'].unique())}")
print(f"Null value summary: {df.isnull().sum()}")
print(f"Unique genres: {df['listed_in'].unique()}")
# print(f"Unique release years: {df['year_added'].unique()}")
print("\n\n\n")
# print(df.head(1))
tfidf = TfidfVectorizer(stop_words=['the', 'is', 'and', 'in', 'to', 'of', 'a', 'for', 'on', 'that', 'with'])
tfidf_matrix = tfidf.fit_transform(df['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(id:str=None, title:str=None, cosine_sim=cosine_sim):
    if id:
        idx = df.index[df['show_id'] == id].tolist()[0]
    elif title:
        idx = df.index[df['title'] == title].tolist()[0]
    else:
        return None
    print(f"idx: {idx}, description: {df.iloc[idx]['description']}, listed_in: {df.iloc[idx]['listed_in']}")
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    for i in movie_indices:
        print(f"idx: {i}, description: {df.iloc[i]['description']}")
        print(f"similarity: {cosine_sim[idx][i]}")
        print("\n\n")
    return df['title'].iloc[movie_indices]
# return df['title'].iloc[movie_indices].tolist()
get_recommendations(title='Chicago Med')
# show_id type title director cast country date_added release_year rating duration listed_in description

# Perhaps use a model to obtain context of plot: Medical, sports, etc.