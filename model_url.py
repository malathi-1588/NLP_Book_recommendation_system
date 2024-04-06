# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:54:30 2024

@author: vinod
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:49:43 2024

@author: vinod
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import process

#def make_clickable(val):
#    return '<a target="_blank" href="{}">Goodreads</a>'.format(val)


#%%
df = pd.read_csv(r"D:\Malathi\SEM_6\Book\goodreads_data.csv")


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Description'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#%%

def recommend_books_similar_to(book_name, n=5, cosine_sim_mat=cosine_sim):
    
    # Finding the closest match to the input book_name
    matches = process.extractOne(book_name, df['Book'])
    closest_book_name = matches[0]
    print(f"Closest match to '{book_name}' found: '{closest_book_name}'")
    
    books = pd.Series(df['Book'])
    input_idx = books[books == closest_book_name].index[0]
    print("Index of the closest match:", input_idx)
    
    top_n_books_idx = list(pd.Series(cosine_sim_mat[input_idx]).sort_values(ascending=False).iloc[1:n+1].index)
    
    recommended_books = [books[i] for i in top_n_books_idx]
    
    list_of_summary = []
    for i in top_n_books_idx:
        summary = ""
        try:
            text = df['Description'][i]
            summary = sentences_from_clusters(text)
            list_of_summary.append(summary)
        except:
            continue
        
    links = get_urls_for_books(recommended_books)    
            
    return recommended_books, list_of_summary, links

        

#%%
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
fasttext_model = api.load("fasttext-wiki-news-subwords-300")

import ast
def keyword_embedding(df, fasttext_model):
    # df['keywords1'] = df['keywords1'].apply(ast.literal_eval)
    df['embedding'] = np.nan
    arr = np.zeros((9923, 300))
    for i, row in enumerate(df['keywords1']):
        vect = np.zeros(300,  dtype=np.float32)
        for keyword in row:
            if keyword in fasttext_model:
                vect += fasttext_model[keyword]
        vect = vect / len(row)
        arr[i] = vect
    return arr

arr = keyword_embedding(df, fasttext_model)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def user_pref(ans1, ans2, ans3, ans4, ans5, fasttext_model=fasttext_model, arr=arr, df=df):
    user_response = [ans1, ans2, ans3, ans4, ans5]
    arr_user = np.zeros((5, 300))
    
    for i, ans in enumerate(user_response):
        vect = np.zeros(300, dtype=np.float32)  # Initialize a new vector for each response
        if ans in fasttext_model:
            vect += fasttext_model[ans]  # Add the vector representation of the word
        arr_user[i] = vect / len(user_response)  # Assign the normalized vector to arr_user
    arr[np.isnan(arr)] = 0
    arr_user[np.isnan(arr_user)] = 0
    cosine_sim = cosine_similarity(arr_user, arr)
    top_books_indices = []
    for i in range(cosine_sim.shape[0]):
        user_top_indices = np.argsort(cosine_sim[i])[-1:][::-1]  # Get indices of top 5 similar books
        top_books_indices.extend(user_top_indices)
    # Get unique top indices and their corresponding book names
    unique_top_indices = list(set(top_books_indices))
    top_books = df.iloc[unique_top_indices]['Book']
    top_books_list = list(top_books)
    list_of_summary = []
    for i in unique_top_indices:
        summary = ""
        try:
            text = df['Description'][i]
            summary = sentences_from_clusters(text)
            list_of_summary.append(summary)
        except:
            continue
        
    links = get_urls_for_books(top_books)
        
    return top_books_list, list_of_summary, links


#%%

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans
import numpy as np
import random

def sentences_from_clusters(text):
    result = ""
    sent = []
    sentences = sent_tokenize(text)
    tagged_data = [TaggedDocument(words=word_tokenize(sentence.lower()), tags=[str(i)]) for i, sentence in enumerate(sentences)]
    model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=100)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    embeddings = [model.infer_vector(word_tokenize(sentence.lower())) for sentence in sentences]
    X = np.array(embeddings)
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X)
    cluster_assignments = kmeans.labels_
    for cluster_id in range(kmeans.n_clusters):
        cluster_indices = np.where(cluster_assignments == cluster_id)[0]
        random_index = np.random.choice(cluster_indices)
        sent.append(sentences[random_index])
    
    result = " ".join(sent)
    return result

#%%
def get_urls_for_books(book_names):
    urls = []
    for book_name in book_names:
        # Search for the book in the DataFrame
        matches = process.extractOne(book_name, df['Book'])
        closest_book_name = matches[0]
        book_row = df[df['Book'] == closest_book_name]
        
        # Get the URL for the book
        book_url = book_row['URL'].values[0] if not book_row.empty else None
        urls.append(book_url)
    
    return urls




