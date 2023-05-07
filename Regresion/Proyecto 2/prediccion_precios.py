import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Importando Datos
sales_df = pd.read_csv("datos_de_ventas.csv")

#Visualizacion
sns.scatterplot(sales_df['Temperature'], sales_df['Revenue'])

#Creando set de entrenamiento
X_train = sales_df['Temperature']
y_train = sales_df['Revenue']

#Creando Modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 1, input_shape = [1]))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss = 'mean_squared_error')

#Entrenamiento
epochs_hist = model.fit(X_train, y_train, epochs = 1000)


keys = epochs_hist.history.keys()


# Grafico de Entrenamiento del Modelo
plt.plot(epochs_hist.history['loss'])
plt.title('Progreso de Perdida durante Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])


weights = model.get_weights()


# Prediccion
Temp = 5
Revenue = model.predict([Temp])
print('La Ganancia segun la Red Neuronal sera de: ', Revenue)

#Grafico de Prediccion
plt.scatter(X_train, y_train, color='gray')
plt.plot(X_train, model.predict(X_train), color = 'red')
plt.ylabel('Ganancia [Dolares]')
plt.xlabel('Temperatura [gCelsius]')
plt.title('Ganancia Generada vs. Temperatura @Empresa de Helados')









