# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:30:59 2021

@author: amit kumar
"""

import pandas as pd
import numpy as np
data=pd.read_csv(r'C:\Users\ayon_\Downloads\coaching\reco\Entertainment.csv')
data.Category
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words = "english")
data.Category.isnull().sum()
data.Category = data.Category.fillna(" ")
tfidf_matrix = tfidf.fit_transform(data.Category)  
tfidf_matrix.shape
from sklearn.metrics.pairwise import linear_kernel
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
index = pd.Series(data.index, index = data['Titles']).drop_duplicates()
data_id = index["Toy Story (1995)"]
data_id

def get_recommendations(Titles, topN):     
    data_id = index[Titles]
    cosine_scores = list(enumerate(cosine_sim_matrix[data_id]))
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    data_idx  =  [i[0] for i in cosine_scores_N]
    data_scores =  [i[1] for i in cosine_scores_N]
    
    
    data_similar_show = pd.DataFrame(columns=["Titles", "Score"])
    data_similar_show["Titles"] = data.loc[data_idx, "Titles"]
    data_similar_show["Score"] = data_scores
    data_similar_show.reset_index(inplace = True)  
    print (data_similar_show)
    
get_recommendations('GoldenEye (1995)', topN = 10)