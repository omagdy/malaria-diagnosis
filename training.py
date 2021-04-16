import os
import time
import datetime
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from logger import log
from plotting import plot_evaluations
from model_checkpoints import get_model, save_model
from data_processing import get_image_batch, get_label_batch, data_preprocessing


@tf.function
def train_step(images, labels):
    with tf.GradientTape(persistent=True) as tape:
        output = model(images, training=True)
        loss = tf.keras.losses.binary_crossentropy(labels, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    model_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def training_loop(x_train, y_train, N_TRAINING_DATA, BATCH_SIZE):
    for i in range(0, N_TRAINING_DATA, BATCH_SIZE):
        r = np.random.randint(0,2,2)
        batch_image = get_image_batch(x_train, i, BATCH_SIZE, r[0], r[1])
        batch_label = get_label_batch(y_train, i, BATCH_SIZE)
        model_loss = train_step(batch_image, batch_label).numpy().mean()
    return model_loss

@tf.function
def evaluation_loop(x_data, y_data):
    output = model(x_data, training=False)
    loss = tf.keras.losses.binary_crossentropy(y_data, output)
    return loss

def main_loop(EPOCHS, BATCH_SIZE, LR):

    training_start = time.time()

    malaria_images    = glob.glob("dataset/malaria/*.jpg")
    no_malaria_images = glob.glob("dataset/no malaria/*.jpg")

    data, labels = data_preprocessing(malaria_images, no_malaria_images)

    x_train, x_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.2, random_state=42)
    x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    
    data_splits = [[y_train, x_train], [y_validation, x_validation], [y_test, x_test]]
    clipped_data_splits = []
    for i in range(len(data_splits)):
        data_split = data_splits[i]
        y_data = data_split[0]
        x_data = data_split[1]
        n_data  = y_data.shape[0]
        if n_data%BATCH_SIZE != 0:
            n_data = n_data - n_data%BATCH_SIZE
            x_data = x_data[0:n_data]
            y_data = y_data[0:n_data]
        clipped_data_splits.append([y_data, x_data])
    (y_train, x_train),(y_validation, x_validation),(y_test, x_test) = clipped_data_splits

    N_TRAINING_DATA   = x_train.shape[0]
    N_VALIDATION_DATA = x_validation.shape[0]
    N_TESTING_DATA    = x_test.shape[0]
    
    no_data_log = "Number of Training Data: {}, Number of Validation Data: {}, Number of Testing Data: {}".format(N_TRAINING_DATA, N_VALIDATION_DATA, N_TESTING_DATA)
    log(no_data_log)
                
    global model, model_optimizer
    model, model_optimizer = get_model(LR_G)
        
    epochs_plot = []
    training_loss_plot = []
    validation_loss_plot = []

    for epoch in range(EPOCHS):

        epoch_s_log = "Began epoch {} at {}".format(epoch, time.ctime())
        log(epoch_s_log)
        epoch_start = time.time()
        
        #TRAINING
        try:
            model_loss = training_loop(x_train, y_train, N_TRAINING_DATA, BATCH_SIZE)
        except tf.errors.ResourceExhaustedError:
            oom_error_log = "Encountered OOM Error at {} !".format(time.ctime())
            log(oom_error_log)
            return

        training_loss_plot.append(model_loss)
        y_train, x_train = shuffle(y_train, x_train)
        
        #Validation
        validation_error = evaluation_loop(x_validation, y_validation.reshape(-1,1)).numpy().mean()
        validation_loss_plot.append(validation_error)
        
        #Epoch Logging
        epochs_plot.append(epoch)
        epoch_e_log = "Finished epoch {} at {}.".format(epoch, time.ctime())
        log(epoch_e_log)
        epoch_seconds = time.time() - epoch_start
        epoch_t_log = "Epoch took {}".format(datetime.timedelta(seconds=epoch_seconds))
        log(epoch_t_log)
        
        if epoch == 40:
            freezing_log = "Training only head layers from no on."
            log(freezing_log)
            for layer in model.layers[:-2]:
                layer.trainable = False
            
    plot_evaluations(epochs_plot, training_loss_plot, validation_loss_plot)
    #Testing
    test_error = evaluation_loop(x_test, y_test.reshape(-1,1)).numpy().mean()
    evaluation_log = "After training: Error = {}".format(test_error)
    log(evaluation_log)
    training_e_log = "Finished training at {}".format(time.ctime())
    log(training_e_log)
    training_seconds = time.time() - training_start
    training_t_log = "Training took {}".format(datetime.timedelta(seconds=training_seconds))
    log(training_t_log)
