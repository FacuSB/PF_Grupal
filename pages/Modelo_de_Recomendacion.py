import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

#streamlit run Desicion_Inversion.py
#python -m streamlit run Decisión_Inversion.py

df_combined = pd.read_parquet("pages\modeloml2.parquet") 
vectorizer = TfidfVectorizer()
business_categories_matrix = vectorizer.fit_transform(df_combined['categories'])

###############################################################################################################################

def obtener_recomendaciones_usuario(user_id, top_n=3):
    # Filtrar las críticas del usuario dado
    user_reviews = df_combined[df_combined['user_id'] == user_id]
    
    if user_reviews.empty:
        return {"Error": f"No se encontraron críticas para el usuario con id {user_id}"}
    
    # Calcular la similitud del coseno entre las categorías de negocios del usuario y todos los negocios
    user_categories_vector = vectorizer.transform(user_reviews['categories'])
    similarities = cosine_similarity(user_categories_vector, business_categories_matrix)
    
    # Obtener los índices de los negocios más similares
    similar_business_indices = similarities.argsort(axis=1)[:, ::-1]
    
    # Obtener las recomendaciones de negocios y filtrar duplicados
    recommendations = []
    seen = set()  # Para rastrear los negocios que ya hemos recomendado
    for indices in similar_business_indices:
        for index in indices:
            business_name = df_combined.iloc[index]['name_y']
            if business_name not in seen:
                categories = df_combined.iloc[index]['categories'].split(', ')
                recommendation = {
                    "restaurant": business_name,
                    "categories": categories
                }
                recommendations.append(recommendation)
                seen.add(business_name)
                if len(recommendations) == top_n:
                    return recommendations  # Salir si ya hemos obtenido suficientes recomendaciones
    
    # Devolver todas las recomendaciones encontradas si no hemos alcanzado top_n
    return recommendations

###############################################################################################################################
def obtener_info_usuario(user_id, top_n=3):
    # Filtrar las críticas del usuario dado
    user_reviews = df_combined[df_combined['user_id'] == user_id]
    
    if user_reviews.empty:
        return {"Error": f"No se encontraron críticas para el usuario con id {user_id}"}
    
    # Obtener el nombre del usuario
    user_name = user_reviews['name_x'].iloc[0]
    
    # Obtener todas las categorías de los restaurantes revisados por el usuario
    all_categories = ', '.join(user_reviews['categories']).split(', ')
    
    # Contar la frecuencia de cada categoría
    category_counts = {}
    for category in all_categories:
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Ordenar las categorías por frecuencia y obtener las top_n categorías más comunes
    favorite_categories = sorted(category_counts, key=category_counts.get, reverse=True)[:top_n]
    
    # Crear el diccionario JSON
    user_info = {
        "name": user_name,
        "favorite": favorite_categories
    }
    
    return user_info
###############################################################################################################################


# Obtener IDs disponibles en df_combined
available_ids = df_combined['user_id'].unique()

# Si df_combined es grande, puedes mostrar solo 100 IDs aleatorios para seleccionar
random.shuffle(available_ids)
available_ids = available_ids[:min(len(available_ids), 100)]

# Mostrar select box con los IDs de usuario disponibles



#####################################################################################################################################################################################

# Crear la interfaz de usuario con Streamlit
st.markdown("""
    <div style="text-align: center"> 
        <h1>Modelo de recomendacion de usuarios 🚀</h1>
    </div>
""", unsafe_allow_html=True)


selected_user = st.selectbox("Selecciona un usuario:", available_ids)

st.markdown("""---""")

boton = st.button('Iniciar')
if boton :
    user_info = obtener_info_usuario(selected_user)
    recommendations = obtener_recomendaciones_usuario(selected_user)

    st.markdown("""
    <div style="text-align: center"> 
        <h2>Información del usuario:</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding:10px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); border-radius:5px; text-align: center;">
        <h3><code style="color:black;">Usuario: {user_info["name"]}</code></h3>
        <p><code style="color:black;">Preferencias: {', '.join(user_info["favorite"])}</code></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center"> 
        <h2>Recomendacion para el usuario:</h2>
    </div>
    """, unsafe_allow_html=True)
    
    rcol1, rcol2, rcol3 = st.columns(3)

    # Iterar sobre las recomendaciones y asignarlas a las tarjetas
    for i, recommendation in enumerate(recommendations[:3]):
        if i == 0:
            tarjeta = rcol1
        elif i == 1:
            tarjeta = rcol2
        else:
            tarjeta = rcol3

        tarjeta.markdown(f"""
            <div style="padding:10px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); border-radius:5px; text-align: center;">
                <h3><code style="color:#E694FF;">{recommendation["restaurant"]}</code></h3>
                <p><code style="color:black;">Categorías: {', '.join(recommendation["categories"])}</code></p>
            </div>
            """, unsafe_allow_html=True)