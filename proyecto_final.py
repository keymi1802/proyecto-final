#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = '/content/tortilla_prices.csv'
price_df = pd.read_csv(dataset)
price_df.head()

#%%
# Combina las columnas day, month, year en una nueva columna llamada fecha
price_df['fecha'] = pd.to_datetime(price_df[['Year', 'Month', 'Day']])
price_df.head()

#%%
# Eliminar entradas duplicadas
price_df_cleaned = price_df.drop_duplicates(subset=['Month', 'Year'], keep='first')

plt.figure(figsize=(13, 6))
price_heatmap = price_df_cleaned.pivot(index='Month', columns='Year', values='Price per kilogram')
sns.heatmap(price_heatmap, fmt=".2f", annot=True, cmap='Blues')
plt.title('Mapa de Calor de Precios de Tortillas')
plt.xlabel('Año')
plt.ylabel('Mes')
plt.show()

#%%
# Eliminar las columnas day, year y month
price_df.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)
price_df.head()

#%%
# estadísticas descriptivas de los precios de las tortillas
descripcion = price_df['Price per kilogram'].describe()
print(descripcion)

#%%
price_df['fecha'] = pd.to_datetime(price_df['fecha'])   #Converir la columna fecha en tipo datetime
price_df.set_index('fecha', inplace=True)               # Establecer la columna 'fecha' como el índice del DataFrame
#Calcular el precio mensual 
promedio_mensual = price_df['Price per kilogram'].resample('M').mean()
print(promedio_mensual)

#%%
# Calcular el precio promedio anual
promedio_anual = price_df['Price per kilogram'].resample('Y').mean()
print(promedio_anual)

#%%
plt.figure(figsize=(8, 5))
plt.hist(price_df['Price per kilogram'], bins=20, edgecolor='black')

plt.xlabel('Precio de las tortillas')
plt.ylabel('Frecuencia')
plt.title('Distribución de los precios de las tortillas')

plt.grid(True)
plt.show()

#%%
price_df['fecha'] = pd.to_datetime(price_df['fecha'])     #Converir la columna fecha en tipo datetime
mensual_df = price_df.resample('M', on='fecha')['Price per kilogram'].mean()    #Calcular el promedio mensual del precio 

plt.figure(figsize=(12, 6))
plt.plot(mensual_df, marker='o', linestyle='-')
plt.xlabel('Fecha')
plt.ylabel('Precio Promedio por Kilogramo')
plt.title('Tendencia de los Precios de las Tortillas a lo Largo del Tiempo')
plt.grid(True)
plt.show()

#%%
plt.figure(figsize=(10, 6))
sns.histplot(price_df['Price per kilogram'], bins=30, kde=True)
plt.xlabel('Precio por Kilo')
plt.ylabel('Frecuencia')
plt.title('Distribución de los Precios de las Tortillas')
plt.show()

#%%
plt.figure(figsize=(14, 6))
sns.boxplot(x='State', y='Price per kilogram', data=price_df)
plt.xticks(rotation=90)
plt.xlabel('Estado')
plt.ylabel('Precio por Kilogramo')
plt.title('Comparación de Precios de las Tortillas entre Estados')
plt.show()

#%%
plt.figure(figsize=(14, 6))
sns.boxplot(x='State', y='Price per kilogram', data=price_df)
plt.xticks(rotation=90)
plt.xlabel('Estado')
plt.ylabel('Precio por Kilogramo')
plt.title('Comparación de Precios de las Tortillas entre Estados')
plt.show()

#%%
plt.figure(figsize=(10, 6))
sns.boxplot(y=price_df['Price per kilogram'])
plt.title('Análisis de Valores Atípicos en los Precios de las Tortillas')
plt.show()

#%%
# Calcular el precio promedio por estado
promedio = price_df.groupby('State')['Price per kilogram'].mean().reset_index()

plt.figure(figsize=(14, 8))
sns.barplot(x='Price per kilogram', y='State', data=promedio, palette='viridis')

plt.xlabel('Precio Promedio por Kilogramo')
plt.ylabel('Estado')
plt.title('Precio Promedio de Tortillas por Estado')

plt.tight_layout()
plt.show()

#%%
def grafico_porcentaje_precios_tipo_tienda(df):
    """
    Crea un gráfico de pastel que muestra el porcentaje de precios por tipo de tienda.

    Args:
    df (DataFrame): DataFrame con los datos de precios de tortillas.

    Returns:
    None
    """
    plt.figure(figsize=(8, 5))
    
    tienda_tipo = df['Store type'].value_counts()   #Calcular valores unicos
    explode = [0.1] + [0] * (len(tienda_tipo) - 1)  #Separar las partes 

    #Crear grafica 
    tienda_tipo.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'), 
                           labels=None, explode=explode, shadow=True, startangle=140, textprops={'fontsize': 12})
    plt.title('Porcentaje de Precios por Tipo de Tienda', fontsize=18, fontweight='bold')
    plt.axis('equal') 
    plt.ylabel('')  
    #Agregar texto
    plt.legend(tienda_tipo.index, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    plt.text(0.5, -0.1, "Este gráfico de pastel muestra el porcentaje de precios por tipo de tienda.", 
             ha='center', fontsize=12, transform=plt.gca().transAxes)
    plt.show()

grafico_porcentaje_precios_tipo_tienda(price_df)

#%%
promedio_prices = {}    #Creardiccionario vacio para guardar el promedio de los precios 

for store_type in price_df['Store type'].unique():      #Iterar sobre los tipos de tiendas 
    store_prices = price_df[price_df['Store type'] == store_type]['Price per kilogram']   #Se tman los precios de las tortillas que correspondan al tipo de tienda 
    
    mean_price = store_prices.mean()      #Se calcula el promedio para ese tipo de tienda 
    
    promedio_prices[store_type] = mean_price    #Se guarda en el diccionario 

for store_type, mean_price in promedio_prices.items():      #Itera sobre el diccionario e imprime resultados 
    print(f"El precio promedio de las tortillas en tiendas de tipo '{store_type}' es: {mean_price:.2f}")

#%%
price_df['Year-Month'] = price_df['fecha'].dt.to_period('M')     #Crear columna que tenga mes y año 

promedio_mensual_prices_store = {}      #Diccionario vacio

for store_type in price_df['Store type'].unique():
    store_prices = price_df[price_df['Store type'] == store_type]     #Obtener filas del tipo de tienda
    
    promedio_mensual_prices = store_prices.groupby('Year-Month')['Price per kilogram'].mean()   # Calcular el precio promedio mensual agrupando por año-mes y calculando la media de los precios
    
    promedio_mensual_prices_store[store_type] = promedio_mensual_prices     #Guardar resultados en el diccionario 

for store_type, promedio_mensual_prices in promedio_mensual_prices_store.items():     #iterar soble el diccionario para imprimir resultados y graficar 
    print(f"\nPrecio promedio mensual de las tortillas en tiendas de tipo '{store_type}':")
    print(promedio_mensual_prices)
    promedio_mensual_prices.plot(label=store_type)

plt.title('Precio promedio mensual de las tortillas por tipo de tienda')
plt.xlabel('Año')
plt.ylabel('Precio por kilogramo')
plt.legend(title='Tipo de tienda')
plt.show()

#%%
q25 = price_df['Price per kilogram'].quantile(0.25)     #Calcula cuartiles 
q75 = price_df['Price per kilogram'].quantile(0.75)

def clasificar_precio(precio):      #Condicion para clasificar según lo scuartiles 
    if precio < q25:
        return 'Bajo'
    elif precio > q75:
        return 'Alto'
    else:
        return 'Medio'

price_df['Clasificación Precio'] = price_df['Price per kilogram'].apply(clasificar_precio)  #crear nueva columna y aplicar la clasificacion a la columna price

print(price_df.head())

clasificacion_counts = price_df['Clasificación Precio'].value_counts()    #Contar total de veces que aparece cada clasificación 

plt.figure(figsize=(10, 6))
clasificacion_counts.plot(kind='bar', color=['purple', 'pink', 'red'])
plt.title('Distribución de la Clasificación de Precios de Tortillas')
plt.xlabel('Clasificación')
plt.ylabel('Cantidad')
plt.xticks(rotation=0)
plt.show()
