from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import json
import pickle
import numpy as np

with open('../models/random_forest_classifier_2.sav', 'rb') as file:
    model = pickle.load(file)


with open('../data/processed/dic_dep.json','r', encoding='utf-8') as archivo:
    dic_dep = json.load(archivo)
with open('../data/processed/dic_sal.json','r', encoding='utf-8') as archivo:
    dic_sal = json.load(archivo)
with open('../data/processed/dic_Work_accident.json','r', encoding='utf-8') as archivo:
    dic_Work_accident = json.load(archivo)
with open('../data/processed/dic_promotion_last_5years.json','r', encoding='utf-8') as archivo:
    dic_promotion_last_5years = json.load(archivo)
with open('../data/processed/dic_left.json','r', encoding='utf-8') as archivo:
    dic_left = json.load(archivo)
# Configuración de la página y tema personalizados

st.set_page_config(
    page_title="The dropout decoder",
    page_icon="❌",
    layout="wide"
)

st.title("The dropout decoder")
st.subheader("Herramienta para predecir el abandono de empleados")
#st.write("Esta herramienta utiliza un modelo de Machine Learning para predecir si un empleado abandonará la empresa o no, basado en diferentes características del empleado.")
st.write("Por favor, complete los siguientes campos para realizar la predicción:")

col1, col2 = st.columns(2)

with col1:
    val1 = st.slider("Nivel de satisfacción del empleado (engagement)", min_value = 0.0, max_value = 1.0, step = 0.01)
    val2 = st.slider("Resultado última evaluación del empleado", min_value = 0.2, max_value = 1.0, step = 0.01)
    val3 = st.slider("Número de proyectos en los que ha contribuido/participado", min_value = 0, max_value = 8, step = 1)
    val4 = st.slider("Horas mensuales que pasa el trabajador en la empresa", min_value = 90.0, max_value = 320.0, step = 1.0)
    val5 = st.slider("Tiempo que lleva el empleado en la empresa [años]", min_value = 1.0, max_value = 6.0, step = 1.0)
with col2:
    val6 = st.selectbox(
        "Departamento del empleado",
        (dic_dep.keys()),
        index=None,
        placeholder="Selecciona el área del empleado....",
)
    val7 = st.selectbox(
        "Clasificación salarial del empleado",
        (dic_sal.keys()),
        index=None,
        placeholder="Selecciona nivel salarial del empleado....",
)
    val8 = st.selectbox(
        "¿El empleado ha tenido accidentes laborales?",
        (dic_Work_accident.keys()),
        index=None,
        placeholder="Selecciona si el empleado ha tenido algún accidente....",
)
    val9 = st.selectbox(
        "¿El empleado ha sido promovido en los últimos 5 años?",
        (dic_promotion_last_5years.keys()),
        index=None,
        placeholder="Selecciona si el empleado ha sido promovido en los últimos 5 años....",
)

if st.button("Predecir"):
    # convertir categoricas en numericas con diccionario
    dep_value = dic_dep[val6] if val6 is not None else None
    sal_value = dic_sal[val7] if val7 is not None else None
    accident_value = dic_Work_accident[val8] if val8 is not None else None
    promotion_value = dic_promotion_last_5years[val9] if val9 is not None else None
    
    # verificar
    if None in [val6, val7, val8, val9]:
        st.error("Por favor, complete todos los campos antes de predecir")
    else:
        # Crear el array de entrada y remodelar para la predicción
        X_pred = np.array([
            val1, val2, val3, val4, val5, 
            dep_value, sal_value, accident_value, promotion_value
        ]).reshape(1, -1)
        
        try:
            prediction = model.predict(X_pred)[0]
            result = "Sí" if prediction == 1 else "No"
            st.write(f"¿El empleado va a desertar? {result}")
        except Exception as e:
            st.error(f"Error al realizar la predicción: {str(e)}")
