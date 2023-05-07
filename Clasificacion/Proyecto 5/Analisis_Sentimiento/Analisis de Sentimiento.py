import numpy as np
import math
import re
import pandas as pd
from bs4 import BeautifulSoup

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

#Set de Datos
cols = ["sentiment", "id", "date", "query", "user", "text"]
train_data = pd.read_csv("train.csv", 
                         header=None, 
                         names = cols,
                         engine="python",
                         encoding="latin1")

test_data = pd.read_csv("test.csv", 
                         header=None, 
                         names = cols,
                         engine="python",
                         encoding="latin1")

data = train_data

#Limpieza
data.drop(["id","date","query","user"],
          axis=1,
          inplace = True)

def clean_tweet(tweet):
    tweet = BeautifulSoup(tweet, "lxml").get_text()
    # Remover el @
    tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
    # Remover links URL
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
    # Quedandono solo con letras
    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
    # Removiendo espacios adicionales
    tweet = re.sub(r" +", ' ', tweet)
    return tweet   
    
data_clean = [clean_tweet(tweet) for tweet in data.text]
    
data_labels = data.sentiment.values
data_labels[data_labels == 4] = 1

#Tokenizacion
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    data_clean, target_vocab_size=2**16
)

data_inputs = [tokenizer.encode(sentence) for sentence in data_clean]

#CREANDO INPUTS
#Padding
MAX_LEN = max([len(sentence) for sentence in data_inputs])
data_inputs = tf.keras.preprocessing.sequence.pad_sequences(data_inputs,
                                                            value=0,
                                                            padding="post",
                                                            maxlen=MAX_LEN)

#Sets de Entrenamiento y Prueba
test_idx = np.random.randint(0, 800000, 8000)
test_idx = np.concatenate((test_idx, test_idx+800000))

test_inputs = data_inputs[test_idx]
test_labels = data_labels[test_idx]
train_inputs = np.delete(data_inputs, test_idx, axis=0)
train_labels = np.delete(data_labels, test_idx)

#Creando Red Neuronal
class DCNN(tf.keras.Model):
    
    def __init__(self,
                 vocab_size,
                 emb_dim=128,
                 nb_filters=50,
                 FFN_units=512,
                 nb_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="dcnn"):
        super(DCNN, self).__init__(name=name)
        
        self.embedding = layers.Embedding(vocab_size,
                                          emb_dim)
        self.bigram = layers.Conv1D(filters=nb_filters,
                                    kernel_size=2,
                                    padding="valid",
                                    activation="relu")
        self.trigram = layers.Conv1D(filters=nb_filters,
                                     kernel_size=3,
                                     padding="valid",
                                     activation="relu")
        self.fourgram = layers.Conv1D(filters=nb_filters,
                                      kernel_size=4,
                                      padding="valid",
                                      activation="relu")
        self.pool = layers.GlobalMaxPool1D() 
        
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=nb_classes,
                                           activation="softmax")
    
    def call(self, inputs, training):
        x = self.embedding(inputs)
        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool(x_3)
        
        merged = tf.concat([x_1, x_2, x_3], axis=-1)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)
        
        return output

#Configuracion del Modelo
VOCAB_SIZE = tokenizer.vocab_size
EMB_DIM = 200
NB_FILTERS = 100
FFN_UNITS = 256
NB_CLASSES = len(set(train_labels))

DROPOUT_RATE = 0.2

BATCH_SIZE = 32
NB_EPOCHS = 5

#Entrenamiento
Dcnn = DCNN(vocab_size=VOCAB_SIZE,
            emb_dim=EMB_DIM,
            nb_filters=NB_FILTERS,
            FFN_units=FFN_UNITS,
            nb_classes=NB_CLASSES,
            dropout_rate=DROPOUT_RATE)

if NB_CLASSES == 2:
    Dcnn.compile(loss="binary_crossentropy",
                 optimizer="adam",
                 metrics=["accuracy"])
else:
    Dcnn.compile(loss="sparse_categorical_crossentropy",
                 optimizer="adam",
                 metrics=["sparse_categorical_accuracy"])

#Checkpoints
checkpoint_path = "ckpt/"

ckpt = tf.train.Checkpoint(Dcnn=Dcnn)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.lates_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Ulltimo Checkpoint Cargado")
    
Dcnn.fit(train_inputs,
         train_labels,
         batch_size=BATCH_SIZE,
         epochs=NB_EPOCHS)
ckpt_manager.save()


#Evaluacion
results = Dcnn.evaaluate(test_inputs, test_labels, batch_size=BATCH_SIZE)
print(results)

