import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

#Cargar Set de Datos
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#Normalizacion de Imagenes
X_train = X_train / 255.0
X_test = X_test / 255.0

#Reformar Datos
X_train = X_train.reshape(-1, 28*28)
X_train.shape

X_test = X_test.reshape(-1, 28*28)


# CREANDO RED NEURONAL
#Definir Modelo
model = tf.keras.models.Sequential()

#Primera Capa
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784, )))

#Segunda Capa
model.add(tf.keras.layers.Dense(units=64, activation='relu'))

#Capa Desercion
model.add(tf.keras.layers.Dropout(0.2))

#Capa OutPut
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

#COMPILADO
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

#ENTRENANDO MODELO
model.fit(X_train, y_train, epochs=10)

#EVALUANDO MODELO
test_loss, test_accuracy = model.evaluate(X_test, y_test)
