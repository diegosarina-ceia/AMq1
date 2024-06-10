import msno
import missingno as msno
# Análisis de valores faltantes por regiones
missing_data_percentage = weather_original_df.groupby('Location').apply(lambda x: x.isnull().mean() * 100)

# Función para aplicar colores de fondo condicionales
def conditional_bg_color(val):
red = min(255, int(2.55 * val))
green = min(255, int(2.55 * (100 - val)))
return f'background-color: rgb({red}, {green}, 0); color: black;'

missing_data_percentage.style.applymap(conditional_bg_color)
#carga de los datos
weather_original_df = pd.read_csv("./dataset/weatherAUS.csv")

#cantidad de elementos en el dataset
print(weather_original_df.shape)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import scipy.stats as stats
import matplotlib.dates as mdates

#configuraciones generales
pd.set_option('display.max_columns', None)
#carga de los datos
weather_original_df = pd.read_csv("./dataset/weatherAUS.csv")

#cantidad de elementos en el dataset
print(weather_original_df.shape)
#carga de los datos
weather_original_df = pd.read_csv("../dataset/weatherAUS.csv")

#cantidad de elementos en el dataset
print(weather_original_df.shape)
missing_values = weather_original_df.isna().sum()
missing_percentage = (missing_values / len(weather_original_df)) * 100
missing_data = pd.DataFrame({'Feature': missing_values.index, 'MissingPercentage': missing_percentage})
missing_data = missing_data.sort_values(by='MissingPercentage', ascending=False)

plt.figure(figsize=(12, 8))
ax = sns.barplot(x='MissingPercentage', y='Feature', data=missing_data)
plt.title('Porcentaje de Valores Faltantes por Feature')
plt.xlabel('Porcentaje de Valores Faltantes')
plt.ylabel('Variables')

# Añadir el porcentaje encima de cada barra
for i in ax.containers:
ax.bar_label(i, fmt='%.1f%%', padding=5)

plt.show()
# Análisis de valores faltantes por regiones
missing_data_percentage = weather_original_df.groupby('Location').apply(lambda x: x.isnull().mean() * 100)

# Función para aplicar colores de fondo condicionales
def conditional_bg_color(val):
red = min(255, int(2.55 * val))
green = min(255, int(2.55 * (100 - val)))
return f'background-color: rgb({red}, {green}, 0); color: black;'

missing_data_percentage.style.applymap(conditional_bg_color)
weather_original_df["Date"] = pd.to_datetime(weather_original_df['Date'])
#weather_original_df['Year'] = dataset_clean['Date'].dt.year
weather_original_df['Month'] = dataset_clean['Date'].dt.month
#weather_original_df['Day'] = dataset_clean['Date'].dt.day
weather_original_df["Date"] = pd.to_datetime(weather_original_df['Date'])
#weather_original_df['Year'] = dataset_clean['Date'].dt.year
weather_original_df['Month'] = weather_original_df['Date'].dt.month
#weather_original_df['Day'] = dataset_clean['Date'].dt.day
# Análisis de valores faltantes por regiones
missing_data_percentage = weather_original_df.groupby('Month').apply(lambda x: x.isnull().mean() * 100)

# Función para aplicar colores de fondo condicionales
def conditional_bg_color(val):
red = min(255, int(2.55 * val))
green = min(255, int(2.55 * (100 - val)))
return f'background-color: rgb({red}, {green}, 0); color: black;'

missing_data_percentage.style.applymap(conditional_bg_color)
# Análisis de valores faltantes por regiones
missing_data_percentage = weather_original_df.groupby(['Month','Year']).apply(lambda x: x.isnull().mean() * 100)

# Función para aplicar colores de fondo condicionales
def conditional_bg_color(val):
red = min(255, int(2.55 * val))
green = min(255, int(2.55 * (100 - val)))
return f'background-color: rgb({red}, {green}, 0); color: black;'

missing_data_percentage.style.applymap(conditional_bg_color)
weather_original_df["Date"] = pd.to_datetime(weather_original_df['Date'])
weather_original_df['Year'] = dataset_clean['Date'].dt.year
weather_original_df['Month'] = weather_original_df['Date'].dt.month
#weather_original_df['Day'] = dataset_clean['Date'].dt.day
weather_original_df["Date"] = pd.to_datetime(weather_original_df['Date'])
weather_original_df['Year'] = weather_original_df['Date'].dt.year
weather_original_df['Month'] = weather_original_df['Date'].dt.month
#weather_original_df['Day'] = dataset_clean['Date'].dt.day
# Análisis de valores faltantes por regiones
missing_data_percentage = weather_original_df.groupby(['Month','Year']).apply(lambda x: x.isnull().mean() * 100)

# Función para aplicar colores de fondo condicionales
def conditional_bg_color(val):
red = min(255, int(2.55 * val))
green = min(255, int(2.55 * (100 - val)))
return f'background-color: rgb({red}, {green}, 0); color: black;'

missing_data_percentage.style.applymap(conditional_bg_color)
# Análisis de valores faltantes por regiones
missing_data_percentage = weather_original_df.groupby(['Month','Location']).apply(lambda x: x.isnull().mean() * 100)

# Función para aplicar colores de fondo condicionales
def conditional_bg_color(val):
red = min(255, int(2.55 * val))
green = min(255, int(2.55 * (100 - val)))
return f'background-color: rgb({red}, {green}, 0); color: black;'

missing_data_percentage.style.applymap(conditional_bg_color)
# creamos dataframe conde vamos a guardar las variables ya imputadas
weather_imputed_df = weather_original_df.copy()[['Date', 'Location']]
weather_imputed_df
# creacion del dataset imputado
df_final = weather_original_df.copy()
df_final
# creacion del dataset imputado
df_final = pd.read_csv("../dataset/weatherAUS.csv")
df_final = df_final.dropna(subset=["Rainfall", "RainToday", "RainTomorrow"])
print(f"Porcentaje de informacion eliminado: {((len(df) - len(df_final)) / len(df_final)) * 100:.2f}%")
# creacion del dataset imputado
df = pd.read_csv("../dataset/weatherAUS.csv")
df_final = pd.read_csv("../dataset/weatherAUS.csv")
df_final = df_final.dropna(subset=["Rainfall", "RainToday", "RainTomorrow"])
print(f"Porcentaje de informacion eliminado: {((len(df) - len(df_final)) / len(df_final)) * 100:.2f}%")
weather_df
#carga de los datos
weather_df = pd.read_csv("../dataset/weatherAUS.csv")

#cantidad de elementos en el dataset
print(weather_df.shape)
weather_df.corr()
weather_df.drop('Date', inplace=True).corr()
weather_df.drop('Date', inplace=True)
weather_df
weather_df.drop('Date')
weather_df.drop(['Date'], inplace=True).corr()
weather_df.drop(["Date"])
weather_df.dropna(subset=["Date"])
weather_df.drop(columns=["Date"]).corr()
numeric_cols = weather_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
weather_df.drop(columns=numeric_cols).corr()
numeric_cols = weather_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
weather_df.drop(columns=numeric_cols).corr()
categorical_cols = weather_df.select_dtypes(include=['object']).columns.tolist()
weather_df.drop(columns=categorical_cols).corr()
categorical_cols = weather_df.select_dtypes(include=['object']).columns.tolist()
corr_matrix = weather_df.drop(columns=categorical_cols).corr()
plt.figure(figsize=(16,12))
plt.title('Correlation Heatmap')

ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, fmt='.1f', linecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_xticklabels(), rotation=30)
def check_raintoday_condition(df):
condition = ((df['Rainfall'] >= 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] < 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
def check_raintoday_condition(df):
condition = ((df['Rainfall'] > 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] =< 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
def check_raintoday_condition(df):
condition = ((df['Rainfall'] > 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] <= 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
df_final['Rainfall'].value_counts()
# corrección de los 1747 datos
condition = ((df_final['Rainfall'] >= 1) & (df_final['RainToday'] == 'Yes')) | ((df_final['Rainfall'] < 1) & (df_final['RainToday'] == 'No'))

df_final[~condition]
# corrección de los 1747 datos
condition = ((df_final['Rainfall'] >= 1) & (df_final['RainToday'] == 'Yes')) | ((df_final['Rainfall'] < 1) & (df_final['RainToday'] == 'No'))

df_final[~condition]["Rainfall"]
# corrección de los 1747 datos
condition = ((df_final['Rainfall'] >= 1) & (df_final['RainToday'] == 'Yes')) | ((df_final['Rainfall'] < 1) & (df_final['RainToday'] == 'No'))

df_final[~condition]["Rainfall"].value_counts()
# corrección de los 1747 datos
condition = ((df_final['Rainfall'] >= 1) & (df_final['RainToday'] == 'Yes')) | ((df_final['Rainfall'] < 1) & (df_final['RainToday'] == 'No'))

condition =  ((df_final['Rainfall'] < 1) & (df_final['RainToday'] == 'No'))

df_final[~condition]["Rainfall"].value_counts()
# corrección de los 1747 datos
condition = ((df_final['Rainfall'] >= 1) & (df_final['RainToday'] == 'Yes')) | ((df_final['Rainfall'] < 1) & (df_final['RainToday'] == 'No'))

condition =  ((df_final['Rainfall'] < 1) & (df_final['RainToday'] == 'Yes'))

df_final[~condition]["Rainfall"].value_counts()
# corrección de los 1747 datos
condition = ((df_final['Rainfall'] >= 1) & (df_final['RainToday'] == 'Yes')) | ((df_final['Rainfall'] < 1) & (df_final['RainToday'] == 'No'))

condition =  ((df_final['Rainfall'] < 1) & (df_final['RainToday'] == 'Yes'))

df_final[condition]["Rainfall"].value_counts()
# Si hay valores de Rainfall, el flag Raintoday debe ser positivo.

def check_raintoday_condition(df):
condition = ((df['Rainfall'] >= 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] < 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
# probamos una variacion de la condicion

condition = ((df_final['Rainfall'] > 1) & (df_final['RainToday'] == 'Yes')) | ((df_final['Rainfall'] <= 1) & (df_final['RainToday'] == 'No'))

#condition =  ((df_final['Rainfall'] < 1) & (df_final['RainToday'] == 'Yes'))
count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
# probamos una variacion de la condicion

def check_raintoday_condition(df):
condition = ((df['Rainfall'] > 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] <= 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
df_final['Rainfall'].value_counts()
condition = ((df['Rainfall'] > 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] <= 1) & (df['RainToday'] == 'No'))
df_final[~condition].value_counts()
condition = ((df['Rainfall'] >= 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] < 1) & (df['RainToday'] == 'No'))
df_final[~condition].value_counts()
condition = ((df['Rainfall'] >= 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] < 1) & (df['RainToday'] == 'No'))
df_final[~condition]['Rainfall'].value_counts()
df_final['WindGustDir'].unique()
df_final['Location'].unique()
df_final = pd.get_dummies(df_final, columns=['Location'], dummy_na=True, drop_first=True)
display(df_final.head(5))
df_final.info()
# La nueva columna Location_nan parece no tener sentido, ya que se había analizado previamente que no había valores nulos para Location
len(df_final[df_final['Location_nan'] == True])
del df_final['Location_nan']
df_final['WindGustDir'].unique()
direccion_to_angulo = {
'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

def codificacion_wind_dir(df, columna, direccion_to_angulo):
# Se obtienen los ángulos a partir del mapeo
angulos = df[columna].map(direccion_to_angulo)

# Se convierten los ángulos a radianes
angulos_rad = np.deg2rad(angulos)

# Se crean las nuevas columnas con el seno y el coseno
df[f'{columna}_sin'] = np.sin(angulos_rad)
df[f'{columna}_cos'] = np.cos(angulos_rad)

# Se setea NaN en las nuevas columnas para aquellas direcciones que sean nulas.
df.loc[df[columna].isna(), [f'{columna}_sin', f'{columna}_cos']] = np.nan

# Se elimina la columna original
del df[columna]
#return df

codificacion_wind_dir(df_final, 'WindGustDir', direccion_to_angulo)
codificacion_wind_dir(df_final, 'WindDir9am', direccion_to_angulo)
codificacion_wind_dir(df_final, 'WindDir3pm', direccion_to_angulo)
df_final.head(5)
df_final.dtypes
df_final["Location_Canberra"].dtypes
df_final["Date"] = pd.to_datetime(df_final['Date'])
df_final['Year'] = df_final['Date'].dt.year
df_final['Month'] = df_final['Date'].dt.month
df_final['Day'] = df_final['Date'].dt.day
df_final["Date"].drop(inplace=True)
df_final["Date"] = pd.to_datetime(df_final['Date'])
df_final['Year'] = df_final['Date'].dt.year
df_final['Month'] = df_final['Date'].dt.month
df_final['Day'] = df_final['Date'].dt.day
df_final.drop(columns=["Date"],inplace=True)
df_final["Date"]
df_final["RainToday"] = df_final["RainToday"].apply(lambda x: 1 if x == "Yes" else 0)
df_final["RainTomorrow"] = df_final["RainTomorrow"].apply(lambda x: 1 if x == "Yes" else 0)
df_final
df_final["RainTomorrow"]
# Vamos a utilizar la moda
region_montly_mode = df_final.groupby(['Month', 'Location'])[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
#carga de los datos
weather_original_df = pd.read_csv("../dataset/weatherAUS.csv")

#cantidad de elementos en el dataset
print(weather_original_df.shape)
# creacion del dataset a imputar
df = pd.read_csv("../dataset/weatherAUS.csv")
df_final = pd.read_csv("../dataset/weatherAUS.csv")
df_final = df_final.dropna(subset=["Rainfall", "RainToday", "RainTomorrow"])
print(f"Porcentaje de informacion eliminado: {((len(df) - len(df_final)) / len(df_final)) * 100:.2f}%")
# Si hay valores de Rainfall, el flag Raintoday debe ser positivo.

def check_raintoday_condition(df):
condition = ((df['Rainfall'] >= 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] < 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
# probamos una variacion de la condicion

def check_raintoday_condition(df):
condition = ((df['Rainfall'] > 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] <= 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
# Vamos a utilizar la moda
region_montly_mode = df_final.groupby(['Month', 'Location'])[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
# Vamos a utilizar la moda por mes y locación
df_final["Date"] = pd.to_datetime(df_final['Date'])
df_final['Month'] = df_final['Date'].dt.month

region_montly_mode = df_final.groupby(['Month', 'Location'])[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
from sklearn.impute import SimpleImputer
imp_by_mean = SimpleImputer(missing_values=np.nan, strategy="most_frecuent")
imputed_data_by_mean = imp_by_mean.fit_transform(df_final[VARIABLES_CATEGORICAS])
df_final = pd.DataFrame(imputed_data_by_mean, columns=df_final[VARIABLES_CATEGORICAS].columns)
df_final
VARIABLES_NUMERICAS = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
VARIABLES_CATEGORICAS = ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
imp_by_mean = SimpleImputer(missing_values=np.nan, strategy="most_frecuent")
imputed_data_by_mean = imp_by_mean.fit_transform(df_final[VARIABLES_CATEGORICAS])
df_final = pd.DataFrame(imputed_data_by_mean, columns=df_final[VARIABLES_CATEGORICAS].columns)
df_final
imp_by_mean = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
imputed_data_by_mean = imp_by_mean.fit_transform(df_final[VARIABLES_CATEGORICAS])
df_final = pd.DataFrame(imputed_data_by_mean, columns=df_final[VARIABLES_CATEGORICAS].columns)
df_final
# creacion del dataset a imputar
df = pd.read_csv("../dataset/weatherAUS.csv")
df_final = pd.read_csv("../dataset/weatherAUS.csv")
df_final = df_final.dropna(subset=["Rainfall", "RainToday", "RainTomorrow"])
print(f"Porcentaje de informacion eliminado: {((len(df) - len(df_final)) / len(df_final)) * 100:.2f}%")
# Vamos a utilizar la moda por mes y locación
df_final["Date"] = pd.to_datetime(df_final['Date'])
df_final['Month'] = df_final['Date'].dt.month

region_montly_mode = df_final.groupby(['Month', 'Location'])[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
# Rellenar valores faltantes para columnas categóricas
df_final[categorical_cols] = df_final.apply(
lambda row: pd.Series(
    [region_montly_mode.loc[(row['Month'], row['Location']), col] if pd.isna(row[col]) else row[col] for col in categorical_cols],
    index=categorical_cols
),
axis=1
)
numeric_var_null = df_final.select_dtypes(include=['float64', 'int64']).isnull().mean() * 100
print("Porcentaje de valores perdidos para variables numericas: ")
print(numeric_var_null)

categorical_var_null = df_final.select_dtypes(include=['object']).isnull().mean() * 100
print("\nPorcentaje de valores perdidos para variables categoricas:")
print(categorical_var_null)
categorical_var_null = df_final.select_dtypes(include=['object']).isnull().mean() * 100
print("\nPorcentaje de valores perdidos para variables categoricas:")
print(categorical_var_null)
df_final[df_final["WindGustDir" is None]]
df_final[df_final["WindGustDir"==False]]
df_final[df_final["WindGustDir"].isnull()]
df_final[df_final["WindGustDir"].isnull()]["WindGustDir"]
df_final[df_final["WindGustDir"].isnull()]["WindGustDir"].count()
df_final[df_final["WindGustDir"].isnull()].count()
df_final[df_final["WindGustDir"].isnull()]
categorical_cols
df_final['Month'].dtype
categorical_cols = df_final.select_dtypes(include=['object','int32']).columns.tolist()
categorical_cols
# Vamos a utilizar la moda por mes y locación
df_final["Date"] = pd.to_datetime(df_final['Date'])
df_final['Month'] = df_final['Date'].dt.month

categorical_cols = df_final.select_dtypes(include=['object','int32']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
# Rellenar valores faltantes para columnas categóricas
df_final[categorical_cols] = df_final.apply(
lambda row: pd.Series(
    [region_montly_mode.loc[(row['Month'], row['Location']), col] if pd.isna(row[col]) else row[col] for col in categorical_cols],
    index=categorical_cols
),
axis=1
)
categorical_var_null = df_final.select_dtypes(include=['object']).isnull().mean() * 100
print("\nPorcentaje de valores nulos para variables categoricas:")
print(categorical_var_null)
categorical_var_null = df_final.select_dtypes(include=['object']).isnull().mean() * 100
print("\nPorcentaje de valores nulos para variables categoricas:")
print(categorical_var_null)
categorical_var_null = df_final.select_dtypes(include=['object','int32']).isnull().mean() * 100
print("\nPorcentaje de valores nulos para variables categoricas:")
print(categorical_var_null)
# Vamos a utilizar la moda por mes y locación
df_final["Date"] = pd.to_datetime(df_final['Date'])
df_final['Month'] = df_final['Date'].dt.month

categorical_cols = df_final.select_dtypes(include=['object','int32']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
categorical_cols
df_final.columns
df_final.groupby(['Month', 'Location'])
df_final.groupby(['Month', 'Location']).head()
# Vamos a utilizar la moda por mes y locación
df_final["Date"] = pd.to_datetime(df_final['Date'])
df_final['Month'] = df_final['Date'].dt.month

categorical_cols = df_final.select_dtypes(include=['object','int32']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
region_montly_mode.fillna(region_montly_mode.mode().iloc[0], inplace=True)
# Rellenar valores faltantes para columnas categóricas
df_final[categorical_cols] = df_final.apply(
lambda row: pd.Series(
    [region_montly_mode.loc[(row['Month'], row['Location']), col] if pd.isna(row[col]) else row[col] for col in categorical_cols],
    index=categorical_cols
),
axis=1
)
categorical_var_null = df_final.select_dtypes(include=['object','int32']).isnull().mean() * 100
print("\nPorcentaje de valores nulos para variables categoricas:")
print(categorical_var_null)
# Vamos a utilizar la moda por mes y locación
numerical_cols = df_final.select_dtypes(include=['float64']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
region_montly_mode.fillna(region_montly_mode.mode().iloc[0], inplace=True)
# Rellenar valores faltantes para columnas categóricas
df_final[categorical_cols] = df_final.apply(
lambda row: pd.Series(
    [region_montly_mode.loc[(row['Month'], row['Location']), col] if pd.isna(row[col]) else row[col] for col in categorical_cols],
    index=categorical_cols
),
axis=1
)
categorical_var_null = df_final.select_dtypes(include=['float64']).isnull().mean() * 100
print("\nPorcentaje de valores nulos para variables categoricas:")
print(categorical_var_null)
# Vamos a utilizar la moda por mes y locación
numerical_cols = df_final.select_dtypes(include=['float64']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[numerical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
region_montly_mode.fillna(region_montly_mode.mode().iloc[0], inplace=True)
# Rellenar valores faltantes para columnas categóricas
df_final[numerical_cols] = df_final.apply(
lambda row: pd.Series(
    [region_montly_mode.loc[(row['Month'], row['Location']), col] if pd.isna(row[col]) else row[col] for col in numerical_cols],
    index=numerical_cols
),
axis=1
)
numerical_var_null = df_final.select_dtypes(include=['float64']).isnull().mean() * 100
print("\nPorcentaje de valores nulos para variables categoricas:")
print(numerical_var_null)
# Vamos a utilizar la moda por mes y locación
numerical_cols = df_final.select_dtypes(include=['float64']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[numerical_cols].agg(lambda x: x.median().iloc[0] if not x.median().empty else None)
region_montly_mode.fillna(region_montly_mode.mode().iloc[0], inplace=True)
# Vamos a utilizar la moda por mes y locación
numerical_cols = df_final.select_dtypes(include=['float64']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[numerical_cols].agg(lambda x: x.median().iloc[0] if not x.median().empty else None)
#region_montly_mode.fillna(region_montly_mode.mode().iloc[0], inplace=True)
# creacion del dataset a imputar
df = pd.read_csv("../dataset/weatherAUS.csv")
df_final = pd.read_csv("../dataset/weatherAUS.csv")
df_final = df_final.dropna(subset=["Rainfall", "RainToday", "RainTomorrow"])
print(f"Porcentaje de informacion eliminado: {((len(df) - len(df_final)) / len(df_final)) * 100:.2f}%")
# Si hay valores de Rainfall, el flag Raintoday debe ser positivo.

def check_raintoday_condition(df):
condition = ((df['Rainfall'] >= 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] < 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
# Vamos a utilizar la moda por mes y locación
df_final["Date"] = pd.to_datetime(df_final['Date'])
df_final['Month'] = df_final['Date'].dt.month

categorical_cols = df_final.select_dtypes(include=['object','int32']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
region_montly_mode.fillna(region_montly_mode.mode().iloc[0], inplace=True)
# Rellenar valores faltantes para columnas categóricas
df_final[categorical_cols] = df_final.apply(
lambda row: pd.Series(
    [region_montly_mode.loc[(row['Month'], row['Location']), col] if pd.isna(row[col]) else row[col] for col in categorical_cols],
    index=categorical_cols
),
axis=1
)
categorical_var_null = df_final.select_dtypes(include=['object','int32']).isnull().mean() * 100
print("\nPorcentaje de valores nulos para variables categoricas:")
print(categorical_var_null)
# Vamos a utilizar la moda por mes y locación
numerical_cols = df_final.select_dtypes(include=['float64']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[numerical_cols].median()
region_montly_mode.fillna(region_montly_mode.median().iloc[0], inplace=True)
# Rellenar valores faltantes para columnas numericas
df_final[numerical_cols] = df_final.apply(
lambda row: pd.Series(
    [region_montly_mode.loc[(row['Month'], row['Location']), col] if pd.isna(row[col]) else row[col] for col in numerical_cols],
    index=numerical_cols
),
axis=1
)
numerical_var_null = df_final.select_dtypes(include=['float64']).isnull().mean() * 100
print("\nPorcentaje de valores nulos para variables categoricas:")
print(numerical_var_null)
# creacion del dataset a imputar
df = pd.read_csv("../dataset/weatherAUS.csv")
df_final = pd.read_csv("../dataset/weatherAUS.csv")
df_final = df_final.dropna(subset=["Rainfall", "RainToday", "RainTomorrow"])
print(f"Porcentaje de informacion eliminado: {((len(df) - len(df_final)) / len(df_final)) * 100:.2f}%")
# Si hay valores de Rainfall, el flag Raintoday debe ser positivo.

def check_raintoday_condition(df):
condition = ((df['Rainfall'] >= 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] < 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
# probamos una variacion de la condicion

def check_raintoday_condition(df):
condition = ((df['Rainfall'] > 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] <= 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
# Vamos a utilizar la moda por mes y locación
df_final["Date"] = pd.to_datetime(df_final['Date'])
df_final['Month'] = df_final['Date'].dt.month

categorical_cols = df_final.select_dtypes(include=['object','int32']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
region_montly_mode.fillna(region_montly_mode.mode().iloc[0], inplace=True)
# Rellenar valores faltantes para columnas categóricas
df_final[categorical_cols] = df_final.apply(
lambda row: pd.Series(
    [region_montly_mode.loc[(row['Month'], row['Location']), col] if pd.isna(row[col]) else row[col] for col in categorical_cols],
    index=categorical_cols
),
axis=1
)
categorical_var_null = df_final.select_dtypes(include=['object','int32']).isnull().mean() * 100
print("\nPorcentaje de valores nulos para variables categoricas:")
print(categorical_var_null)
# Vamos a utilizar la moda por mes y locación
numerical_cols = df_final.select_dtypes(include=['float64']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[numerical_cols].median()
region_montly_mode.fillna(region_montly_mode.median(), inplace=True)
# Rellenar valores faltantes para columnas numericas
df_final[numerical_cols] = df_final.apply(
lambda row: pd.Series(
    [region_montly_mode.loc[(row['Month'], row['Location']), col] if pd.isna(row[col]) else row[col] for col in numerical_cols],
    index=numerical_cols
),
axis=1
)
numerical_var_null = df_final.select_dtypes(include=['float64']).isnull().mean() * 100
print("\nPorcentaje de valores nulos para variables categoricas:")
print(numerical_var_null)
# drop de month ya que se define en la codificacion
df_final = df_final.drop(columns=["Month"],inplace=True)
poetry install xgboost
poetry add xgboost
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
df_sample = df_final.sample(10000)
df_sample = df_final.sample(10000, axis=0)
df_sample = df_final.sample(n=10000, axis=0)
df_final
df_final.head()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import scipy.stats as stats
import matplotlib.dates as mdates

#configuraciones generales
pd.set_option('display.max_columns', None)
# Vamos a utilizar la moda por mes y locación
df_final["Date"] = pd.to_datetime(df_final['Date'])
df_final['Month'] = df_final['Date'].dt.month

categorical_cols = df_final.select_dtypes(include=['object','int32']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
region_montly_mode.fillna(region_montly_mode.mode().iloc[0], inplace=True)
# creacion del dataset a imputar
df = pd.read_csv("../dataset/weatherAUS.csv")
df_final = pd.read_csv("../dataset/weatherAUS.csv")
# Vamos a utilizar la moda por mes y locación
df_final["Date"] = pd.to_datetime(df_final['Date'])
df_final['Month'] = df_final['Date'].dt.month

categorical_cols = df_final.select_dtypes(include=['object','int32']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
region_montly_mode.fillna(region_montly_mode.mode().iloc[0], inplace=True)
# Rellenar valores faltantes para columnas categóricas
df_final[categorical_cols] = df_final.apply(
lambda row: pd.Series(
    [region_montly_mode.loc[(row['Month'], row['Location']), col] if pd.isna(row[col]) else row[col] for col in categorical_cols],
    index=categorical_cols
),
axis=1
)
categorical_var_null = df_final.select_dtypes(include=['object','int32']).isnull().mean() * 100
print("\nPorcentaje de valores nulos para variables categoricas:")
print(categorical_var_null)
# Vamos a utilizar la moda por mes y locación
numerical_cols = df_final.select_dtypes(include=['float64']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[numerical_cols].median()
region_montly_mode.fillna(region_montly_mode.median(), inplace=True)
# Rellenar valores faltantes para columnas numericas
df_final[numerical_cols] = df_final.apply(
lambda row: pd.Series(
    [region_montly_mode.loc[(row['Month'], row['Location']), col] if pd.isna(row[col]) else row[col] for col in numerical_cols],
    index=numerical_cols
),
axis=1
)
# drop de month ya que se define en la codificacion
df_final = df_final.drop(columns=["Month"],inplace=True)
direccion_to_angulo = {
'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

def codificacion_wind_dir(df, columna, direccion_to_angulo):
# Se obtienen los ángulos a partir del mapeo
angulos = df[columna].map(direccion_to_angulo)

# Se convierten los ángulos a radianes
angulos_rad = np.deg2rad(angulos)

# Se crean las nuevas columnas con el seno y el coseno
df[f'{columna}_sin'] = np.sin(angulos_rad)
df[f'{columna}_cos'] = np.cos(angulos_rad)

# Se setea NaN en las nuevas columnas para aquellas direcciones que sean nulas.
df.loc[df[columna].isna(), [f'{columna}_sin', f'{columna}_cos']] = np.nan

# Se elimina la columna original
del df[columna]
#return df

codificacion_wind_dir(df_final, 'WindGustDir', direccion_to_angulo)
codificacion_wind_dir(df_final, 'WindDir9am', direccion_to_angulo)
codificacion_wind_dir(df_final, 'WindDir3pm', direccion_to_angulo)
# La nueva columna Location_nan parece no tener sentido, ya que se había analizado previamente que no había valores nulos para Location
len(df_final[df_final['Location_nan'] == True])
df_final['Location'].unique()
df_final
# creacion del dataset a imputar
#df = pd.read_csv("../dataset/weatherAUS.csv")
df_final = pd.read_csv("../dataset/weatherAUS.csv")
# creacion del dataset a imputar
df = pd.read_csv("../dataset/weatherAUS.csv")
df_final = pd.read_csv("../dataset/weatherAUS.csv")
df_final = df_final.dropna(subset=["Rainfall", "RainToday", "RainTomorrow"])
print(f"Porcentaje de informacion eliminado: {((len(df) - len(df_final)) / len(df_final)) * 100:.2f}%")
# Si hay valores de Rainfall, el flag Raintoday debe ser positivo.

def check_raintoday_condition(df):
condition = ((df['Rainfall'] >= 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] < 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
df_final
# Si hay valores de Rainfall, el flag Raintoday debe ser positivo.

def check_raintoday_condition(df):
condition = ((df['Rainfall'] >= 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] < 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
# probamos una variacion de la condicion

def check_raintoday_condition(df):
condition = ((df['Rainfall'] > 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] <= 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
# Vamos a utilizar la moda por mes y locación
df_final["Date"] = pd.to_datetime(df_final['Date'])
df_final['Month'] = df_final['Date'].dt.month

categorical_cols = df_final.select_dtypes(include=['object','int32']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
region_montly_mode.fillna(region_montly_mode.mode().iloc[0], inplace=True)
df_final
# Rellenar valores faltantes para columnas categóricas
df_final[categorical_cols] = df_final.apply(
lambda row: pd.Series(
    [region_montly_mode.loc[(row['Month'], row['Location']), col] if pd.isna(row[col]) else row[col] for col in categorical_cols],
    index=categorical_cols
),
axis=1
)
# Vamos a utilizar la moda por mes y locación
numerical_cols = df_final.select_dtypes(include=['float64']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[numerical_cols].median()
region_montly_mode.fillna(region_montly_mode.median(), inplace=True)
# Rellenar valores faltantes para columnas numericas
df_final[numerical_cols] = df_final.apply(
lambda row: pd.Series(
    [region_montly_mode.loc[(row['Month'], row['Location']), col] if pd.isna(row[col]) else row[col] for col in numerical_cols],
    index=numerical_cols
),
axis=1
)
df_final
# drop de month ya que se define en la codificacion
df_final = df_final.drop(columns=["Month"],inplace=True)
df_final
# creacion del dataset a imputar
df = pd.read_csv("../dataset/weatherAUS.csv")
df_final = pd.read_csv("../dataset/weatherAUS.csv")
df_final = df_final.dropna(subset=["Rainfall", "RainToday", "RainTomorrow"])
print(f"Porcentaje de informacion eliminado: {((len(df) - len(df_final)) / len(df_final)) * 100:.2f}%")
# Si hay valores de Rainfall, el flag Raintoday debe ser positivo.

def check_raintoday_condition(df):
condition = ((df['Rainfall'] >= 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] < 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
# probamos una variacion de la condicion

def check_raintoday_condition(df):
condition = ((df['Rainfall'] > 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] <= 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
# probamos una variacion de la condicion

def check_raintoday_condition(df):
condition = ((df['Rainfall'] > 1) & (df['RainToday'] == 'Yes')) | ((df['Rainfall'] <= 1) & (df['RainToday'] == 'No'))
count_true = condition.sum()
count_false = len(df) - count_true
return count_true, count_false

count_true, count_false = check_raintoday_condition(df_final)
print("Numero de filas que satisfacen la condicion:", count_true)
print("Numero de filas que no satisfacen la condicion:", count_false)
# Vamos a utilizar la moda por mes y locación
df_final["Date"] = pd.to_datetime(df_final['Date'])
df_final['Month'] = df_final['Date'].dt.month

categorical_cols = df_final.select_dtypes(include=['object','int32']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
region_montly_mode.fillna(region_montly_mode.mode().iloc[0], inplace=True)
# Rellenar valores faltantes para columnas categóricas
df_final[categorical_cols] = df_final.apply(
lambda row: pd.Series(
    [region_montly_mode.loc[(row['Month'], row['Location']), col] if pd.isna(row[col]) else row[col] for col in categorical_cols],
    index=categorical_cols
),
axis=1
)
# Vamos a utilizar la moda por mes y locación
numerical_cols = df_final.select_dtypes(include=['float64']).columns.tolist() # agregamos el mes (unico feature int32)

region_montly_mode = df_final.groupby(['Month', 'Location'])[numerical_cols].median()
region_montly_mode.fillna(region_montly_mode.median(), inplace=True)
# Rellenar valores faltantes para columnas numericas
df_final[numerical_cols] = df_final.apply(
lambda row: pd.Series(
    [region_montly_mode.loc[(row['Month'], row['Location']), col] if pd.isna(row[col]) else row[col] for col in numerical_cols],
    index=numerical_cols
),
axis=1
)
# drop de month ya que se define en la codificacion
df_final.drop(columns=["Month"],inplace=True)
df_final
df_final['Location'].unique()
# La nueva columna Location_nan parece no tener sentido, ya que se había analizado previamente que no había valores nulos para Location
len(df_final[df_final['Location_nan'] == True])
df_final = pd.get_dummies(df_final, columns=['Location'], dummy_na=True, drop_first=True)
display(df_final.head(5))
# La nueva columna Location_nan parece no tener sentido, ya que se había analizado previamente que no había valores nulos para Location
len(df_final[df_final['Location_nan'] == True])
direccion_to_angulo = {
'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

def codificacion_wind_dir(df, columna, direccion_to_angulo):
# Se obtienen los ángulos a partir del mapeo
angulos = df[columna].map(direccion_to_angulo)

# Se convierten los ángulos a radianes
angulos_rad = np.deg2rad(angulos)

# Se crean las nuevas columnas con el seno y el coseno
df[f'{columna}_sin'] = np.sin(angulos_rad)
df[f'{columna}_cos'] = np.cos(angulos_rad)

# Se setea NaN en las nuevas columnas para aquellas direcciones que sean nulas.
df.loc[df[columna].isna(), [f'{columna}_sin', f'{columna}_cos']] = np.nan

# Se elimina la columna original
del df[columna]
#return df

codificacion_wind_dir(df_final, 'WindGustDir', direccion_to_angulo)
codificacion_wind_dir(df_final, 'WindDir9am', direccion_to_angulo)
codificacion_wind_dir(df_final, 'WindDir3pm', direccion_to_angulo)
df_final["Date"] = pd.to_datetime(df_final['Date'])
df_final['Year'] = df_final['Date'].dt.year
df_final['Month'] = df_final['Date'].dt.month
df_final['Day'] = df_final['Date'].dt.day
df_final.drop(columns=["Date"],inplace=True)
df_final["RainToday"] = df_final["RainToday"].apply(lambda x: 1 if x == "Yes" else 0)
df_final["RainTomorrow"] = df_final["RainTomorrow"].apply(lambda x: 1 if x == "Yes" else 0)
df_final.head()
df_sample = df_final.sample(n=10000, axis=0)
df_sample
df_sample = df_final.sample(n=10000, axis=0)
df_sample.shape()
df_sample = df_final.sample(n=10000, axis=0)
df_sample.shape
model_target = "RainTomorrow"
y = df[model_target]
X = df.drop(columns = model_target)

selector = SelectFromModel(estimator=xgb.XGBClassifier(),max_features=30)
X
y
df_final["RainToday"] = df_final["RainToday"].apply(lambda x: 1 if x == "Yes" else 0)
df_final["RainTomorrow"] = df_final["RainTomorrow"].apply(lambda x: 1 if x == "Yes" else 0)
model_target = "RainTomorrow"
y = df_sample[model_target]
X = df_sample.drop(columns = model_target)

selector = SelectFromModel(estimator=xgb.XGBClassifier(),max_features=30)
y
model_target = "RainTomorrow"
y = df_sample[model_target]
X = df_sample.drop(columns = model_target)

selector = SelectFromModel(estimator=xgb.XGBClassifier(),max_features=30)

selector.fit(X, y)
selector
selector.get_params
selector.feature_names_in_
selector.importance_getter
selector.get_feature_names_out()
selector.feature_importances_()
selector.feature_importances_
model_target = "RainTomorrow"
y = df_sample[model_target]
X = df_sample.drop(columns = model_target)

# primer arbol
selector = SelectFromModel(estimator=xgb.XGBClassifier(),max_features=30)
selector.fit(X, y)
relevant_features = list(selector.get_feature_names_out())

#selector real
estimator = xgb.XGBClassifier()
estimator.fit(X[relevant_features], y)
feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importances_,X[relevant_features].columns)), columns=['Value','Feature'])
feature_imp
model_target = "RainTomorrow"
y = df_sample[model_target]
X = df_sample.drop(columns = model_target)

# primer arbol
selector = SelectFromModel(estimator=xgb.XGBClassifier(),max_features=30)
selector.fit(X, y)
relevant_features = list(selector.get_feature_names_out())

#selector real
estimator = xgb.XGBClassifier()
estimator.fit(X[relevant_features], y)
feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importances_,X[relevant_features].columns)), columns=['Value','Feature'])
feature_imp.sort("Value")
model_target = "RainTomorrow"
y = df_sample[model_target]
X = df_sample.drop(columns = model_target)

# primer arbol
selector = SelectFromModel(estimator=xgb.XGBClassifier(),max_features=30)
selector.fit(X, y)
relevant_features = list(selector.get_feature_names_out())

#selector real
estimator = xgb.XGBClassifier()
estimator.fit(X[relevant_features], y)
feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importances_,X[relevant_features].columns)), columns=['Value','Feature'])
feature_imp
feature_imp.sort_values(by="Value")
feature_imp.sort_values(by="Value",ascending=False)
df_sample = df_final.sample(n=50000, axis=0)
df_sample.shape
model_target = "RainTomorrow"
y = df_sample[model_target]
X = df_sample.drop(columns = model_target)

# primer arbol
selector = SelectFromModel(estimator=xgb.XGBClassifier(),max_features=76)
selector.fit(X, y)
relevant_features = list(selector.get_feature_names_out())

#selector real
estimator = xgb.XGBClassifier()
estimator.fit(X[relevant_features], y)
feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importances_,X[relevant_features].columns)), columns=['Value','Feature'])
feature_imp
model_target = "RainTomorrow"
y = df_sample[model_target]
X = df_sample.drop(columns = model_target)

# primer arbol
selector = SelectFromModel(estimator=xgb.XGBClassifier(),max_features=74)
selector.fit(X, y)
relevant_features = list(selector.get_feature_names_out())

#selector real
estimator = xgb.XGBClassifier()
estimator.fit(X[relevant_features], y)
feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importances_,X[relevant_features].columns)), columns=['Value','Feature'])
feature_imp
feature_imp.sort_values(by="Value",ascending=False)
df_sample = df_final.sample(n=20000, axis=0)
df_sample.shape
model_target = "RainTomorrow"
y = df_sample[model_target]
X = df_sample.drop(columns = model_target)

# primer arbol
selector = SelectFromModel(estimator=xgb.XGBClassifier(),max_features=74)
selector.fit(X, y)
relevant_features = list(selector.get_feature_names_out())

#selector real
estimator = xgb.XGBClassifier()
estimator.fit(X[relevant_features], y)
feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importances_,X[relevant_features].columns)), columns=['Value','Feature'])
feature_imp
df_sample = df_final.sample(n=10000, axis=0)
df_sample.shape
model_target = "RainTomorrow"
y = df_sample[model_target]
X = df_sample.drop(columns = model_target)

# primer arbol
selector = SelectFromModel(estimator=xgb.XGBClassifier(),max_features=30)
selector.fit(X, y)
relevant_features = list(selector.get_feature_names_out())

#selector real
estimator = xgb.XGBClassifier()
estimator.fit(X[relevant_features], y)
feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importances_,X[relevant_features].columns)), columns=['Value','Feature'])
feature_imp
df_sample = df_final.sample(n=10000, axis=0)
df_sample.shape
model_target = "RainTomorrow"
y = df_sample[model_target]
X = df_sample.drop(columns = model_target)

# primer arbol
selector = SelectFromModel(estimator=xgb.XGBClassifier(),max_features=30)
selector.fit(X, y)
relevant_features = list(selector.get_feature_names_out())

#selector real
estimator = xgb.XGBClassifier()
estimator.fit(X[relevant_features], y)
feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importances_,X[relevant_features].columns)), columns=['Value','Feature'])
feature_imp
df_sample = df_final.sample(n=10000, axis=0)
df_sample.shape
model_target = "RainTomorrow"
y = df_sample[model_target]
X = df_sample.drop(columns = model_target)

# primer arbol
selector = SelectFromModel(estimator=xgb.XGBClassifier(),max_features=30)
selector.fit(X, y)
relevant_features = list(selector.get_feature_names_out())

#selector real
estimator = xgb.XGBClassifier()
estimator.fit(X[relevant_features], y)
feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importances_,X[relevant_features].columns)), columns=['Value','Feature'])
feature_imp