# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:36:37 2024

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

# Función para generar datos aleatorios
def generar_datos_aleatorios(num_filas):
    fechas = pd.date_range('2023-01-01', periods=num_filas, freq='D')
    produccion = np.random.uniform(1000, 5000, size=num_filas)
    presion = np.random.uniform(100, 200, size=num_filas)
    temperatura = np.random.uniform(20, 40, size=num_filas)

    df = pd.DataFrame({
        'Fecha': fechas,
        'Producción': produccion,
        'Presión': presion,
        'Temperatura': temperatura
    })

    return df

# Función para convertir datetime64[ns] a date
def convertir_a_date(df, columna):
    df[columna] = df[columna].dt.date
    return df

# Función para prever la producción, presión y temperatura futura utilizando diferentes modelos
def predecir_variables(datos, modelo_prod, modelo_pres, modelo_temp, num_filas):
    X = np.array(range(len(datos))).reshape(-1, 1)  # Números de días como características
    y_prod = datos['Producción'].values  # Producción como variable objetivo
    y_pres = datos['Presión'].values  # Presión como variable objetivo
    y_temp = datos['Temperatura'].values  # Temperatura como variable objetivo

    # Crear y ajustar los modelos
    modelo_prod.fit(X, y_prod)
    modelo_pres.fit(X, y_pres)
    modelo_temp.fit(X, y_temp)

    # Predecir para el doble de días
    X_prediccion = np.array(range(len(datos) + num_filas)).reshape(-1, 1)

    y_pred_prod = modelo_prod.predict(X_prediccion)
    y_pred_pres = modelo_pres.predict(X_prediccion)
    y_pred_temp = modelo_temp.predict(X_prediccion)

    return y_pred_prod, y_pred_pres, y_pred_temp, modelo_prod, modelo_pres, modelo_temp

# Título de la aplicación
st.title('Predicción de Producción, Presión y Temperatura Futura para PEMEX')

# Generar datos aleatorios y mostrar tabla
num_filas = st.slider('Número de Filas a Generar', min_value=10, max_value=1000, value=100)
datos = generar_datos_aleatorios(num_filas)
st.subheader('Datos Generados')
st.write(datos)

# Convertir fechas a tipo date
datos = convertir_a_date(datos, 'Fecha')

# Análisis de Correlación
st.subheader('Análisis de Correlación')

# Calcular matriz de correlación
corr_matrix = datos[['Producción', 'Presión', 'Temperatura']].corr()

# Configurar el gráfico
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, annot_kws={"fontsize":10})

# Añadir título y etiquetas
ax.set_title('Matriz de Correlación')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
st.pyplot(fig)

# Filtrado de Datos
st.subheader('Filtrado de Datos')
fecha_inicio = st.date_input('Fecha de Inicio', value=date(2023, 1, 1))
fecha_fin = st.date_input('Fecha de Fin', value=date(2023, 1, 1) + pd.Timedelta(days=num_filas-1))
datos_filtrados = datos[(datos['Fecha'] >= fecha_inicio) & (datos['Fecha'] <= fecha_fin)]
st.write(datos_filtrados)

# Modelos a utilizar
modelos = {
    'Regresión Lineal': LinearRegression(),
    'Árbol de Decisión': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Regresión Ridge': Ridge(alpha=1.0)
}

# Selección de modelos por el usuario
modelos_seleccionados = st.multiselect('Seleccione los modelos para predicción', list(modelos.keys()), default=list(modelos.keys()))

# Parámetros de los modelos
params = {
    'Regresión Lineal': {'modelo': LinearRegression(), 'color':'red'},
    'Árbol de Decisión': {'modelo': DecisionTreeRegressor(), 'color':'blue'},
    'Random Forest': {'modelo': RandomForestRegressor(n_estimators=100), 'color':'green'},
    'Regresión Ridge': {'modelo': Ridge(alpha=1.0), 'color':'purple'}
}

# Crear DataFrame para almacenar las predicciones
fechas_totales = pd.date_range(start=datos['Fecha'].min(), periods=len(datos) + num_filas, freq='D')
predicciones_df = pd.DataFrame({'Fecha': fechas_totales})

# Añadir producción, presión y temperatura actual al DataFrame
produccion_actual_completa = np.concatenate([datos['Producción'].values, [np.nan] * num_filas])
presion_actual_completa = np.concatenate([datos['Presión'].values, [np.nan] * num_filas])
temperatura_actual_completa = np.concatenate([datos['Temperatura'].values, [np.nan] * num_filas])

predicciones_df['Producción Actual'] = produccion_actual_completa
predicciones_df['Presión Actual'] = presion_actual_completa
predicciones_df['Temperatura Actual'] = temperatura_actual_completa

# Prever la producción, presión y temperatura futura con cada modelo seleccionado y añadir al DataFrame
metricas_prod = []
metricas_pres = []
metricas_temp = []
for nombre_modelo in modelos_seleccionados:
    modelo = params[nombre_modelo]['modelo']
    color = params[nombre_modelo]['color']
    
    y_pred_prod, y_pred_pres, y_pred_temp, modelo_prod, modelo_pres, modelo_temp = predecir_variables(datos, modelo, modelo, modelo, num_filas)
    predicciones_df[f'Producción Predicha ({nombre_modelo})'] = y_pred_prod
    predicciones_df[f'Presión Predicha ({nombre_modelo})'] = y_pred_pres
    predicciones_df[f'Temperatura Predicha ({nombre_modelo})'] = y_pred_temp

    # Calcular métricas de rendimiento para producción
    y_true_prod = datos['Producción'].values
    y_pred_prod = modelo_prod.predict(np.array(range(len(datos))).reshape(-1, 1))
    mse_prod = mean_squared_error(y_true_prod, y_pred_prod)
    r2_prod = r2_score(y_true_prod, y_pred_prod)
    metricas_prod.append({'Modelo': nombre_modelo, 'MSE': mse_prod, 'R²': r2_prod})

    # Calcular métricas de rendimiento para presión
    y_true_pres = datos['Presión'].values
    y_pred_pres = modelo_pres.predict(np.array(range(len(datos))).reshape(-1, 1))
    mse_pres = mean_squared_error(y_true_pres, y_pred_pres)
    r2_pres = r2_score(y_true_pres, y_pred_pres)
    metricas_pres.append({'Modelo': nombre_modelo, 'MSE': mse_pres, 'R²': r2_pres})

    # Calcular métricas de rendimiento para temperatura
    y_true_temp = datos['Temperatura'].values
    y_pred_temp = modelo_temp.predict(np.array(range(len(datos))).reshape(-1, 1))
    mse_temp = mean_squared_error(y_true_temp, y_pred_temp)
    r2_temp = r2_score(y_true_temp, y_pred_temp)
    metricas_temp.append({'Modelo': nombre_modelo, 'MSE': mse_temp, 'R²': r2_temp})

# Mostrar DataFrame con predicciones
st.subheader('Producción, Presión y Temperatura Actual vs Predicha')
st.write(predicciones_df)

# Mostrar métricas de rendimiento para producción
st.subheader('Métricas de Rendimiento de los Modelos para Producción')
st.write(pd.DataFrame(metricas_prod))

# Mostrar métricas de rendimiento para presión
st.subheader('Métricas de Rendimiento de los Modelos para Presión')
st.write(pd.DataFrame(metricas_pres))

# Mostrar métricas de rendimiento para temperatura
st.subheader('Métricas de Rendimiento de los Modelos para Temperatura')
st.write(pd.DataFrame(metricas_temp))

# Visualización interactiva con Plotly para producción, presión y temperatura
fig_prod = go.Figure()
fig_pres = go.Figure()
fig_temp = go.Figure()

fig_prod.add_trace(go.Scatter(x=predicciones_df['Fecha'], y=predicciones_df['Producción Actual'], mode='lines', name='Producción Actual', line=dict(color='blue', width=2, dash='dash')))
fig_pres.add_trace(go.Scatter(x=predicciones_df['Fecha'], y=predicciones_df['Presión Actual'], mode='lines', name='Presión Actual', line=dict(color='red', width=2, dash='dash')))
fig_temp.add_trace(go.Scatter(x=predicciones_df['Fecha'], y=predicciones_df['Temperatura Actual'], mode='lines', name='Temperatura Actual', line=dict(color='green', width=2, dash='dash')))

for nombre_modelo in modelos_seleccionados:
    color = params[nombre_modelo]['color']
    
    fig_prod.add_trace(go.Scatter(x=predicciones_df['Fecha'], y=predicciones_df[f'Producción Predicha ({nombre_modelo})'], mode='lines', name=f'Producción Predicha ({nombre_modelo})', line=dict(color=color, width=2)))
    fig_pres.add_trace(go.Scatter(x=predicciones_df['Fecha'], y=predicciones_df[f'Presión Predicha ({nombre_modelo})'], mode='lines', name=f'Presión Predicha ({nombre_modelo})', line=dict(color=color, width=2)))
    fig_temp.add_trace(go.Scatter(x=predicciones_df['Fecha'], y=predicciones_df[f'Temperatura Predicha ({nombre_modelo})'], mode='lines', name=f'Temperatura Predicha ({nombre_modelo})', line=dict(color=color, width=2)))

fig_prod.update_layout(title='Predicción de Producción Futura', xaxis_title='Fecha', yaxis_title='Producción')
fig_pres.update_layout(title='Predicción de Presión Futura', xaxis_title='Fecha', yaxis_title='Presión')
fig_temp.update_layout(title='Predicción de Temperatura Futura', xaxis_title='Fecha', yaxis_title='Temperatura')

st.plotly_chart(fig_prod, use_container_width=True)
st.plotly_chart(fig_pres, use_container_width=True)
st.plotly_chart(fig_temp, use_container_width=True)