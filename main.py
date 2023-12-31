from fastapi import FastAPI
import pandas as pd

# Crear instancia de la aplicación FastAPI
app = FastAPI()

# Funciones para los endpoints solicitados

@app.get('/peliculas_idioma/')
def peliculas_idioma(Idioma: str):
    count_peliculas = moviesdataset[moviesdataset['original_language'] == Idioma].shape[0]
    return f"{count_peliculas} películas fueron estrenadas en idioma {Idioma}"

@app.get('/peliculas_duracion/')
def peliculas_duracion(Pelicula: str):
    movie_data = moviesdataset[moviesdataset['original_title'] == Pelicula].iloc[0]
    return f"{Pelicula}. Duración: {movie_data['runtime']}. Año: {movie_data['release_date'][-4:]}"

@app.get('/franquicia/')
def franquicia(Franquicia: str):
    franquicia_data = moviesdataset[moviesdataset['belongs_to_collection'] == Franquicia]
    peliculas_count = franquicia_data.shape[0]
    ganancia_total = franquicia_data['revenue'].sum()
    ganancia_promedio = ganancia_total / peliculas_count
    return f"La franquicia {Franquicia} posee {peliculas_count} peliculas, una ganancia total de {ganancia_total} y una ganancia promedio de {ganancia_promedio}"

@app.get('/peliculas_pais/')
def peliculas_pais(Pais: str):
    count_peliculas = moviesdataset[moviesdataset['production_countries'] == Pais].shape[0]
    return f"Se produjeron {count_peliculas} películas en el país {Pais}"

@app.get('/productoras_exitosas/')
def productoras_exitosas(Productora: str):
    productora_data = moviesdataset[moviesdataset['production_companies'] == Productora]
    revenue_total = productora_data['revenue'].sum()
    peliculas_count = productora_data.shape[0]
    return f"La productora {Productora} ha tenido un revenue de {revenue_total} y ha realizado {peliculas_count} películas"

@app.get('/get_director/')
def get_director(nombre_director: str):
    resultado = []
    director_data = moviesdataset[moviesdataset['director'] == nombre_director]
    
    if not director_data.empty:
        for _, row in director_data.iterrows():
            pelicula_info = {
                "nombre_pelicula": row['nombre_pelicula'],
                "fecha_lanzamiento": row['fecha_lanzamiento'],
                "retorno_individual": row['retorno_individual'],
                "costo": row['costo'],
                "ganancia": row['ganancia']
            }
            resultado.append(pelicula_info)
        return resultado
    else:
        return {"message": "El director no se encuentra en el dataset"}
