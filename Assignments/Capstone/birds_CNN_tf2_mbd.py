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
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dense-layer', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()

    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    dense_layer = args.dense_layer
    dropout = args.dropout
   
    gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.training
    validation_dir = args.validation

    
    # input image dimensions
    img_rows, img_cols = 224, 224
    # rescale RGB values to be between 0 and 1
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_batches = datagen.flow_from_directory(training_dir , class_mode='categorical',
                                            target_size=(img_rows, img_cols))
    val_batches = datagen.flow_from_directory(validation_dir , class_mode='categorical',
                                              target_size=(img_rows, img_cols))
    
    
    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(img_rows, img_cols, 3)))
    #### 1st convolution layer - defining our feature extract head
    ######## Convolutional layer with 32 filters - n of filters to extract
    model.add(MaxPooling2D(pool_size=2))
    #### downsample using a maxpooling operations
    #### feed this into the next set of convolutional layers
    ## 2nd convolultion layer
    ## convolutional layer with 32 filters
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    # Flatten and classify
    ## Flatten spatial information into a vector, and learn the final probability distribution for each class
    model.add(Flatten())
    model.add(Dense(dense_layer, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation="softmax"))
    # Take a look at the model summary
    print(model.summary())
    
    # Use multiple gpus if our instance has more than 1 gpu
    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
        
    model.compile(optimizer=SGD(lr=lr, decay=0.1, momentum=0.1, nesterov=False), loss='categorical_crossentropy', metrics=["accuracy"])
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.fit(train_batches, validation_data=val_batches, epochs=epochs, steps_per_epoch=len(train_batches),
                        validation_steps=len(val_batches), verbose=1, callbacks=[ callback ])
    
    score = model.evaluate(val_batches, verbose=0)
    print('val_loss    :', score[0])
    print('val_accuracy:', score[1])

    # save Keras model for Tensorflow Serving
    model.save(os.path.join(model_dir, '1'))