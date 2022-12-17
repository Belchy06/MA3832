import sys
import time
import itertools
import subprocess
import numpy as np
import argparse, os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.models import Model

# AWS script mode doesn't support requirements.txt
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


if __name__ == '__main__':
    # Keras-metrics brings additional metrics: precision, recall, f1
    # AWS is missing keras metrics
    install('keras-metrics')
    import keras_metrics
    
    # Hyperparameters
    ## all set at particular value here. we will learn how to tune parameters without setting a default
    parser = argparse.ArgumentParser() 
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()

    epochs = args.epochs
   
    gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.training
    validation_dir = args.validation

    
    # input image dimensions
    img_rows, img_cols = 224, 224
    # create data generator
    datagen = ImageDataGenerator(featurewise_center=True)
    # specify imagenet mean values for centering
    datagen.mean = [123.68, 116.779, 103.939]
    
    train_batches = datagen.flow_from_directory(training_dir , class_mode='categorical',
                                            target_size=(img_rows, img_cols))
    val_batches = datagen.flow_from_directory(validation_dir , class_mode='categorical',
                                              target_size=(img_rows, img_cols))
    
    
    base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))
    
    for layer in base_model.layers:
        layer.trainable = False
        
    x = Flatten()(base_model.output)
    x = Dense(1000, activation='relu',)(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.inputs, outputs=predictions)
    # Take a look at the model summary
    print(model.summary())
    
    
    # Use multiple gpus if our instance has more than 1 gpu
    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
        
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
     
    # fit model
    model.fit(train_batches, steps_per_epoch=len(train_batches),validation_data=val_batches, 
                    validation_steps=len(val_batches), epochs=10, verbose=1, callbacks=[ callback ])
    
    score = model.evaluate(val_batches, verbose=0)
    print('val_loss    :', score[0])
    print('val_accuracy:', score[1])

    # save Keras model for Tensorflow Serving
    model.save(os.path.join(model_dir, '1'))