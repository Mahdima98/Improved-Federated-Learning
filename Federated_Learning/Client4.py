import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))





####################### for NMF Reduced Data 8 ###################
data = pd.read_csv('Data4\data4_NMF_train.csv',low_memory=False)

X_train=data[['nmf1', 'nmf2', 'nmf3', 'nmf4', 'nmf5','nmf6','nmf7','nmf8']]
y_train=data[['category']]

test = pd.read_csv('Data4\data4_NMF_test.csv')

X_test=test[['nmf1', 'nmf2', 'nmf3', 'nmf4', 'nmf5','nmf6','nmf7','nmf8']]
y_test=test[['category']]


# ####################### for ICA Reduced Data 8 ###################
# data = pd.read_csv('Data4\data4_ICA_train.csv',low_memory=False)

# X_train=data[['ICA1', 'ICA2', 'ICA3', 'ICA4', 'ICA5','ICA6','ICA7','ICA8']]
# y_train=data[['category']]

# test = pd.read_csv('Data4\data4_ICA_test.csv')

# X_test=test[['ICA1', 'ICA2', 'ICA3', 'ICA4', 'ICA5','ICA6','ICA7','ICA8']]
# y_test=test[['category']]


# Scaling the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train= tf.keras.utils.to_categorical(y_train, num_classes=5)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=5)

X_train=np.reshape(X_train, (X_train.shape[0], 8, 1))



# Load and compile Keras model
model = keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(8,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')

])
model.compile("adam", loss=tf.keras.losses.categorical_crossentropy ,
               metrics=["accuracy",f1_m,precision_m, recall_m])

# # Load dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
# dist = [4000, 4000, 4000, 3000, 10, 10, 10, 10, 4000, 10]
# x_train, y_train = getData(dist, x_train, y_train)
# getDist(y_train)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(X_train, y_train, epochs=5,batch_size=128, validation_data=(X_test, y_test), verbose=0)
        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        print("Eval Precision : ", precision)
        print("Eval Recall : ", recall)
        print("Eval f1_score : ", f1_score)
        return loss, len(X_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        # grpc_max_message_length = 1024*1024*1024
)