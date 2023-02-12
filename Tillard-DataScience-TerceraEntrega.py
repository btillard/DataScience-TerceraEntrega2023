#LIBRERIAS
import pandas as pd
import numpy as np
from pandas import *
from datetime import datetime
from scipy.sparse import csr_matrix 
from lightfm import LightFM
from tqdm import tqdm


#DATOS
#Subo metadatos

col_names = ["asset_id", "content_id", "title", "reduced_title", "episode_title", "show_type", "released_year", "country_of_origin", "category", "keywords", "description", "reduced_desc","cast_first_name", "credits_first_name", "run_time_min", "audience", "made_for_tv", "close_caption", "sex_rating", "violence_rating", "language_rating", "dialog_rating", "fv_rating", "pay_per_view", "pack_premium_1", "pack_premium_2", "create_date", "modify_date", "start_vod_date", "end_vod_date"]
metadata = pd.read_csv("C:/Users/Usuario/Downloads/Entrega-Final/data/metadata.csv", sep=';', header=None, names=col_names)
 

#Elimino columnas que no me sirven
#//title, episode_title, released_year, country_of_origin, keywords, description, reduced_desc, cast_first_name, credits_first_name

metadata = metadata.drop(columns = [ "title", "episode_title", "released_year", "country_of_origin", "keywords", "description", "reduced_desc", "cast_first_name", "credits_first_name"] ,axis=1)
metadata

#Subo Train

train= pd.read_csv("C:/Users/Usuario/Downloads/Entrega-Final/data/train.csv")
train

#Agregamos una columna para caificar

train['rating'] = [1 if i ==0 else 5 for i in train['resume']] 
train

#Unifco los 2 sets  

df = pd.merge(train, metadata, left_on='asset_id', right_on='asset_id', how='left')


#Busco valores vacios y los elimino

df.isna().sum()
df = df.dropna()


#Cambio el formato de las fechas para operar con ellas

df['tunein'] = pd.to_datetime(df['tunein'], format='%Y-%m-%d %H:%M:%S')
df['tuneout'] = pd.to_datetime(df['tuneout'], format='%Y-%m-%d %H:%M:%S')
df['start_vod_date'] = pd.to_datetime(df['tuneout'], format='%Y-%m-%d %H:%M:%S').dt.tz_localize(None)
df['end_vod_date'] = pd.to_datetime(df['end_vod_date'], format='%Y-%m-%dT%H:%M:%S').dt.tz_localize(None)
df.head()

#Entreno y testeo
#Tomo los datos hasta el 1 de Marzo de 2021

train = df[df['tunein'] < datetime(year=2021, month=3, day=1)].copy()
test = df[df['tunein'] >= datetime(year=2021, month=3, day=1)].copy()

train.head(3)
test.head(3)

#Cold start
#Busco cuentas que etsan en TEST y no se entrenaron

test[~test.account_id.isin(train.account_id.unique())].account_id.nunique()

#Interaccion con la Matriz
#Construyo la matriz de interaccon con tabla dinamca de pandas

interactions = train[['account_id', 'content_id', 'rating']].copy()
interactions_matrix = pd.pivot_table(interactions, index='account_id', columns='content_id', values='rating') #pivot tables
interactions_matrix = interactions_matrix.fillna(0) # fill na interactions_matrix

interactions_matrix.head()
interactions_matrix.shape

#Transformo a Matriz csr

interactions_matrix_csr = csr_matrix(interactions_matrix.values)
interactions_matrix_csr

#Indice para cuentas

user_ids = list(interactions_matrix.index)
user_dict = {}
counter = 0
for i in user_ids:
        user_dict[i] = counter
        counter += 1
        
#Indice paacontenido

item_id = list(interactions_matrix.columns)
item_dict = {}
counter = 0 
for i in item_id:
    item_dict[i] = counter
    counter += 1
    
#El modelo
#Uso LFM (light fm) para entrenar nuestro modelo

model = LightFM(random_state=0,
                loss='warp',
                learning_rate=0.03,
                no_components=100)

model = model.fit(interactions_matrix_csr,
                  epochs=100,
                  num_threads=16, verbose=False)

# COLD START
#recomendar el contenido más popular

train.groupby("content_id", as_index=False).agg({"account_id":"nunique"})

popularity_df = train.groupby("content_id", as_index=False).agg({"account_id":"nunique"}).sort_values(by="account_id", ascending=False)
popularity_df.columns=["content_id", "popularity"]
popularity_df.head()

#Top10, contenido mas populor 

popular_content = popularity_df.content_id.values[:10]
popularity_df.head(10).content_id.values
popular_content

#Genero 20 recomendaciones para todos los usuarios (account_id)
# Filtrar lo que el usuario vio previamete
#Si el usauiro no esta entrenado le recomendamos los 20 mas populares

recomms_dict = {
    'account_id': [],
    'recomms': []
}

#Busco la cantidad de usaurios (account_id) y la cantidad de contenido (content_id)

n_users, n_items = interactions_matrix.shape
item_ids = np.arange(n_items)

#Generamos recomnedaciones para cada usuario

for user in tqdm(test.account_id.unique()):
    #Cheuqeo si el usuario (account_id) está en la matriz de interacciones (interactions_matrix.index)
    if user in list(interactions_matrix.index):
      
      user_x = user_dict[user]

      #Genro predicciones para el usuario(account_id) x
      preds = model.predict(user_ids=user_x, item_ids = item_ids)

      #Ordeno de menor a mayor
      scores = pd.Series(preds)
      scores.index = interactions_matrix.columns
      scores = list(pd.Series(scores.sort_values(ascending=False).index))[:50]

      #Lista de lo que vio previamente
      watched_contents = train[train.account_id == user].content_id.unique()

      recomms = [x for x in scores if x not in watched_contents][:20]

      #Almaceno las recomendaciones en el indice
      recomms_dict['account_id'].append(user)
      recomms_dict['recomms'].append(scores)
    
   
    else:
      recomms_dict['account_id'].append(user)
      # We recommend popular content
      recomms_dict['recomms'].append(popular_content)

recomms_df = pd.DataFrame(recomms_dict)
recomms_df

#Comparo las recomendaciones con lo que vieron los usuarios

ideal_recomms =  test.sort_values(by=["account_id", "rating"], ascending=False)\
                  .groupby(["account_id"], as_index=False)\
                  .agg({"content_id": "unique"})\
                  .head()
ideal_recomms.head()

#Control MAP
#Unir los recomendaciones

df_map = ideal_recomms.merge(recomms_df, how="left", left_on="account_id", right_on="account_id")[["account_id", "content_id", "recomms"]]
df_map.columns = ["account_id", "ideal", "recomms"]
df_map.head()

aps = [] 

for pred, label in df_map[["ideal", "recomms"]].values:
  n = len(pred) #Cuento los elementos recomendados)
  arange = np.arange(n, dtype=np.int32) + 1. #Indexa en base 1
  rel_k = np.in1d(pred[:n], label) 
  tp = np.ones(rel_k.sum(), dtype=np.int32).cumsum() 
  denom = arange[rel_k] 
  ap = (tp / denom).sum() / len(label)
  aps.append(ap)