import tensorflow as tf
from logger import log
from model import create_model

def get_model(LR_G):
    model           = create_model()
    model_optimizer = tf.keras.optimizers.Adam(LR_G)
    path = "tensor_checkpoints/"
    ckpt = tf.train.Checkpoint(model=model,
                               model_optimizer=model_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=3)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        log('Latest checkpoint restored!!')
    else:
        log("No checkpoint found! Staring from scratch!")
                
    return model, model_optimizer, ckpt_manager

def save_model(ckpt_manager, epoch):
    ckpt_save_path = ckpt_manager.save()
    log('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))
