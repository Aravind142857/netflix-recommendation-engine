import os, re, requests
# import kagglehub
# from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
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
df['filtered_description'] = df['description'].apply(lambda x: re.sub(r'\d+', '', ' '.join([word for word in x.split() if len(word) > 3]))) # Filter out words with length less than 2
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")
print(f"Data types: {list(df['country'].unique())}")
print(f"Null value summary: {df.isnull().sum()}")
print(f"Unique genres: {df['listed_in'].unique()}")
# print(f"Unique release years: {df['year_added'].unique()}")
print("\n\n\n")
dramas:pd.DataFrame = df[df['listed_in'].str.contains('Dramas', case=False, na=False)]
# print(f"{dramas[dramas['description'].str.contains('medical|hospital', case=False, na=False)]['description']} \n###\n###\n")
def format_row(idx):
  s = ""
  for col in df.columns:
    s += f"{col}: {df.iloc[idx][col]}\n"
  return s

####### TFIDF #######
stop_1 = [
    'the', 'is', 'in', 'a', 'for', 'on', 'with', 'and', 'to', 'it', 
    'this', 'of', 'as', 'from', 'at', 'by', 'that', 'an', 'be', 'are', 
    'movie', 'show', 'film', 'series', 'netflix', 'watch', 'episodes', 'about'
]
stop_2 = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
stop_3_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stop_3 = set(stop_3_list.decode().splitlines()) 
# custom_stop_words = stop_1 + list(set(stop_2) - set(stop_1))
custom_stop_words = list(set(stop_1 + stop_2 + list(stop_3)))

tfidf = TfidfVectorizer(stop_words=custom_stop_words, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['filtered_description'])
with open('tfidf.txt', 'w') as f:
  f.write(f"Number of features: {len(tfidf.get_feature_names_out())}\n")
  for feature in tfidf.get_feature_names_out():
    f.write(f"{feature}\n")
###### TFIDF end #######
###### Sentence transformer begin ######
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process
model = SentenceTransformer('all-MiniLM-L6-v2')

sentence_embeddings = model.encode(df['filtered_description'].to_list())
cosine_ST = cosine_similarity(sentence_embeddings, sentence_embeddings)
def get_fuzzy_recommendations(title: str, cosine_sim=cosine_ST):
    idx = df.index[df['title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top 10 highest values
    top_10_indices = [i[0] for i in sim_scores[1:11]]  # Exclude the first one as it's the title itself
    top_10_rows = df.iloc[top_10_indices][['title', 'description']]
    # Change to obtain name, description
    print(top_10_rows)
    # top_10_descriptions = df['description'].iloc[top_10_indices].tolist()
    title_description_pairs = top_10_rows.set_index('description')['title'].to_dict()
    # Fuzzy matching to sort titles
    sorted_pairs = sorted(title_description_pairs.keys(), key=lambda x: process.extractOne(x, title_description_pairs.keys())[1], reverse=True)
    with open('sentence-transformers.txt', 'w') as f:
      for desc in sorted_pairs:
        f.write(f"{title_description_pairs[desc]}: {desc}\n\n")
    return sorted_pairs

### K Means clustering
num_clusters = 5
km = KMeans(n_clusters=num_clusters, random_state=42)
km.fit(tfidf_matrix)
df['cluster'] = km.labels_
###
### DBSCAN clustering
# dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
# df['dbscan-cluster'] = dbscan.fit_predict(tfidf_matrix)
###

# top_shows_by_cluster = []

# for cluster in range(num_clusters):
#   cluster_shows = df[df['cluster'] == cluster]
#   top_shows = cluster_shows.head(5)
#   top_shows_by_cluster.append(top_shows.index.to_list())

# with open('clusters.txt', 'w') as f:
#    for cluster in top_shows_by_cluster:
#       for show in cluster:
#          f.write(f"############\n{format_row(show)}############\n")
#       f.write("\n\n")
                     
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
  res = ""
  for i in movie_indices:
    res += format_row(i)
    res += f"similarity: {cosine_sim[idx][i]}\n\n"
  with open('output.txt', 'w') as f:
    f.write(res)
  return df['title'].iloc[movie_indices]

get_recommendations(title='Chicago Med')
get_fuzzy_recommendations(title='Chicago Med')
# show_id type title director cast country date_added release_year rating duration listed_in description

# Perhaps use a model to obtain context of plot: Medical, sports, etc.
# Filter to make sure same genre