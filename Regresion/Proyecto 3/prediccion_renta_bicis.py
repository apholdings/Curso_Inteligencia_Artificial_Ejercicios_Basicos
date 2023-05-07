import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Importando Set de Datos
bike = pd.read_csv("rentas_bicis.csv")

#LIMPIEZA DE DATOS
sns.heatmap(bike.isnull())

#Limpiando Columnas
bike = bike.drop(labels = ['instant'], axis= 1)

bike = bike.drop(labels = ['casual', 'registered'], axis= 1)

#Formato de Fecha
bike.dteday = pd.to_datetime(bike.dteday, format = '%m/%d/%Y')

#DateTime como Index
bike.index = pd.DatetimeIndex(bike.dteday)

bike = bike.drop(labels = ['dteday'], axis= 1)


#VISUALIZACION

bike['cnt'].asfreq('W').plot(linewidth = 3)
plt.title('Uso de Bicis por Semana (W)')
plt.xlabel('Semana')
plt.ylabel('Renta de Bicis')


bike['cnt'].asfreq('M').plot(linewidth = 3)
plt.title('Uso de Bicis por Mes (M)')
plt.xlabel('Mes')
plt.ylabel('Renta de Bicis')


bike['cnt'].asfreq('Q').plot(linewidth = 3)
plt.title('Uso de Bicis por Cuarto de Año (Q)')
plt.xlabel('Cuarto de Año')
plt.ylabel('Renta de Bicis')


#Datos Categoricos y Numericos
#Datos Numericos
X_numerical = bike[['temp', 'hum', 'windspeed', 'cnt']]

#Encontrando Correlacion en Datos Numericos
sns.heatmap(X_numerical.corr(), annot = True)


#Datos Categoricos
X_cat = bike[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']]


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()


X_cat = pd.DataFrame(X_cat)


X_numerical = X_numerical.reset_index()

X_all = pd.concat([X_cat, X_numerical], axis = 1)
X_all = X_all.drop('dteday', axis = 1)

X = X_all.iloc[:, :-1].values
y = X_all.iloc[:, -1:].values


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y = scaler.fit_transform(y)


#Sepaarar set de datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



#Definiendo Modelo
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(35, )))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))


model.summary()


#Compilado
model.compile(optimizer='Adam', loss='mean_squared_error')


#Entrenamiento
epochs_hist = model.fit(X_train, y_train, epochs = 20, batch_size = 50, validation_split = 0.2)



#Prediccion
epochs_hist.history.keys()


#Grafico
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Progreso de entrenamiento del modelo')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])


#Prediccion
y_predict = model.predict(X_test)
plt.plot(y_test, y_predict, "^", color = 'r')
plt.xlabel('Prediccion del Modelo')
plt.ylabel('Valores Verdaderos')



























