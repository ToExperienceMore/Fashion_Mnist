import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score
import csv 

print(tf.version.VERSION)

#EPOCH = 50
BATCH_SIZE = 32

#checkpoint_path = "training_dropout/cp-{epoch:04d}.ckpt"
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
model_name = 'fashion_mnist_.h5'


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*BATCH_SIZE)

# Load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    # Horizontal flip
    horizontal_flip=True,
    # Shift images horizontally and vertically by up to 2 pixels
    # random_crop=True,
    # Add random noise
    # 0-15 degree
    rotation_range=15,
    zoom_range=0.1,
)

def train_model(EPOCH, hidden_num, data_argu):
    global x_train
    global y_train
    global x_test
    global y_test

    # Build the model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    """

    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(hidden_num, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    #restore checkpoint
    """
    if os.path.exists(checkpoint_dir):
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
    """

    # Train the model
    # Preprocess the data
    if (data_argu==False):
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        #history = model.fit(x_train, y_train, epochs=EPOCH, validation_data=(x_test, y_test), callbacks=[cp_callback], learning_rate=0.001)
        history = model.fit(x_train, y_train, epochs=EPOCH, validation_data=(x_test, y_test), callbacks=[cp_callback])
    else:
        # Apply data augmentation during training
        train_datagen = ImageDataGenerator(rescale=1./255)
        x_train = x_train.reshape(60000, 28, 28, 1)
        x_test = x_test.reshape(10000, 28, 28, 1)
        train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

        validation_datagen = ImageDataGenerator(rescale=1./255)
        validation_generator = validation_datagen.flow(x_test, y_test, batch_size=BATCH_SIZE)

        #history = model.fit(train_generator, epochs=EPOCH, validation_data=(x_test, y_test), callbacks=[cp_callback])
        history = model.fit(train_generator, epochs=EPOCH, validation_data=validation_generator, callbacks=[cp_callback])

    model.save(model_name)

    acc = [0.] + history.history['accuracy']
    val_acc = [0.] + history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Evaluate the model
    res = model.evaluate(x_test, y_test)
    test_acc = res[-1]
    print("evaluate res:", res[-1])


    # Extract the accuracy from the model's history
    accuracy = history.history['accuracy']

    train_acc = accuracy[-1]
    print("train res:", accuracy[-1])

    # Create a pandas DataFrame
    df = pd.DataFrame({'accuracy': accuracy})

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy of Fashion-MNIST FCN Model')
    plt.legend()
    #plt.show()
    # Save the plot as an image file
    plt.savefig('training_accuracy.png')

    return train_acc, test_acc


EPOCHS = [50]
#hidden_nums = [100, 256, 512]
#hidden_nums = [100, 256]
hidden_nums = [100]
data_argus = [False, True]

current_dir = "./"
writeFile = open('{}/stats.csv'.format(current_dir), 'a')
writer = csv.writer(writeFile)
#writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy'])
writer.writerow(['Epoch', 'hidden num', 'Data Argu', 'train Acc', 'test Acc'])

# for param
for epoch in EPOCHS:
    for hidden_num in hidden_nums:
        for data_argu in data_argus:
            train_acc, test_acc = train_model(epoch, hidden_num, data_argu)
            # Write to csv file
            writer.writerow([epoch, hidden_num, data_argu, train_acc, test_acc])