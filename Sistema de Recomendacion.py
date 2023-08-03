import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Paso 1: Cargar el dataset
moviesdataset = pd.read_csv("C:/Users/Fede/Desktop/moviesdataset.csv")

# Paso 2: Análisis Exploratorio de Datos (EDA)

# Estadísticas descriptivas
print(moviesdataset.describe())

# Distribuciones de variables numéricas
moviesdataset.hist(figsize=(12, 10))

# Matriz de correlación
correlation_matrix = moviesdataset.corr()
print(correlation_matrix)

# Nube de palabras para los títulos de películas
from wordcloud import WordCloud
import matplotlib.pyplot as plt

title_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(moviesdataset['title']))
plt.figure(figsize=(10, 5))
plt.imshow(title_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Paso 3: Sistema de recomendación

# Preprocesamiento de datos
# Convertir columnas relevantes en una sola columna para calcular la similitud
moviesdataset['features'] = moviesdataset['belongs_to_collection'].fillna('') + ' ' + moviesdataset['popularity'].astype(str) + ' ' + moviesdataset['vote_average'].astype(str) + ' ' + moviesdataset['budget'].astype(str) + ' ' + moviesdataset['revenue'].astype(str)

# Cálculo de similitud
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(moviesdataset['features'])
cosine_sim = cosine_similarity(count_matrix)

# Función de recomendación
def recomendacion(titulo):
    index = moviesdataset[moviesdataset['title'] == titulo].index[0]
    similar_scores = list(enumerate(cosine_sim[index]))
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
    similar_movies = [moviesdataset.iloc[i[0]]['title'] for i in similar_scores[1:6]]
    return similar_movies
