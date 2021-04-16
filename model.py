import tensorflow as tf

"""
Model weights are initialized using Xavier
Relu is utilized as the activation layer for convolutions 
Dropout is added to mitigate overfitting
The last output layer has sigmoid as its activation function since we are performing binary classification
"""

IMAGE_SIZE   = 40
WEIGHTS_INIT = tf.keras.initializers.GlorotUniform()

def create_model():
    inputs = tf.keras.layers.Input(shape=[IMAGE_SIZE,IMAGE_SIZE,3])
    conv1 = tf.keras.layers.Conv2D(12, 3, kernel_initializer=WEIGHTS_INIT, activation='relu', padding='valid')(inputs)
    max_pooling_1 = tf.keras.layers.MaxPooling2D()(conv1)
    dropout1  = tf.keras.layers.Dropout(0.2)(max_pooling_1)
    conv2 = tf.keras.layers.Conv2D(32, 3, kernel_initializer=WEIGHTS_INIT, activation='relu', padding='valid')(dropout1)
    max_pooling_2 = tf.keras.layers.MaxPooling2D()(conv2)
    dropout2 = tf.keras.layers.Dropout(0.2)(max_pooling_2)
    flatten = tf.keras.layers.Flatten()(dropout2)
    dense = tf.keras.layers.Dense(128,activation='relu')(flatten)
    output = tf.keras.layers.Dense(1,activation='sigmoid')(dense)
    return tf.keras.Model(inputs=inputs, outputs=output)
