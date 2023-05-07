import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

#Pre Procesamiento Datos
#Nombre de Clases por Identificar
class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Cargando Dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#Normalizacion de Imagenes
X_train = X_train / 255.0
X_train.shape

X_test = X_test / 255.0

#CREANDO RED NEURONAL CONVOLUCIONAL
model = tf.keras.models.Sequential()

#Primera Capa
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32,32,3]))

#Segunda Capa
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#Tercera Capa
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

#Cuarta Capa
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#Aplanado
model.add(tf.keras.layers.Flatten())

#Primera Capa Dense
model.add(tf.keras.layers.Dense(units=128, activation='relu'))

#Segunda Capa Dense (Output)
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))


#Compilado
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["sparse_categorical_accuracy"])

#Entrenando
model.fit(X_train, y_train, epochs=15)


#Evaluando Modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)







































