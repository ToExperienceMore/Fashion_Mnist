import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import os
import time
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.version.VERSION)

EPOCHS = 1000
BATCH_SIZE = 128

checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
model_name = 'fashion_mnist_cnn.h5'


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=10*BATCH_SIZE)
# Load and preprocess the Fashion-MNIST dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
model.add(layers.Flatten())
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

class AccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}, Training Accuracy: {logs['accuracy']:.4f}, Validation Accuracy: {logs['val_accuracy']:.4f}")

#restore checkpoint
if os.path.exists(checkpoint_dir):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

"""
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    steps_per_epoch=len(train_images) // 32,
                    epochs=10,
                    validation_data=(test_images, test_labels))
"""

# Train the model
start = time.perf_counter()
#history = model.fit(train_images, train_labels, epochs= EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2,callbacks=[cp_callback, AccuracyCallback()])
history = model.fit(datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE), steps_per_epoch=len(train_images) // BATCH_SIZE, epochs= EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_images, test_labels), validation_split=0.1,callbacks=[cp_callback])

end = time.perf_counter()
print('Train time: %s Seconds'%(end-start))
model.save(model_name)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

model.evaluate(test_images, test_labels)


# Extract the accuracy from the model's history
accuracy = history.history['accuracy']

# Create a pandas DataFrame
df = pd.DataFrame({'accuracy': accuracy})

# Plot the accuracy
#plt.plot(df['accuracy'])
#plt.plot(df['val_accuracy'])
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy of Fashion-MNIST FCN Model')
plt.legend()
#plt.show()
# Save the plot as an image file
plt.savefig('training_accuracy.png')
