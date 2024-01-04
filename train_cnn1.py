import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score
print(tf.version.VERSION)

EPOCHS = 100
BATCH_SIZE = 32

checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
model_name = 'fashion_mnist_cnn.h5'


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

# Preprocess the data
#x_train = x_train / 255.0
#x_test = x_test / 255.0

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

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#restore checkpoint
if os.path.exists(checkpoint_dir):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

# Train the model
#history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=[cp_callback], learning_rate=0.001)

# Apply data augmentation during training
train_datagen = ImageDataGenerator(rescale=1./255)
#train_datagen.fit(x_train.reshape(60000, 28, 28, 1))
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
train_generator = train_datagen.flow(x_train, y_train, batch_size=32)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow(x_test, y_test, batch_size=32)

#history = model.fit(train_generator, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=[cp_callback])
history = model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator, callbacks=[cp_callback])
model.save(model_name)

# Evaluate the model
#model.evaluate(x_test, y_test)


acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


"""
pred = model.predict(x_test, y_test)

labels_array = np.array([])
pred_array = np.array([])

for x, y in x_test, y_test:
    pred_prob = model.predict(x)
    labels_array = np.concatenate([labels_array, y])
    pred_class = np.argmax(pred_prob, axis=1)
    pred_array = np.concatenate([pred_array, pred_class])    

print("Accuracy On Test Dataset: ", accuracy_score(labels_array, pred_array))
"""

model.evaluate(x_test, y_test)


# Extract the accuracy from the model's history
accuracy = history.history['accuracy']

# Create a pandas DataFrame
df = pd.DataFrame({'accuracy': accuracy})

# Plot the accuracy
plt.plot(df['accuracy'])
#plt.plot(df['train_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy of Fashion-MNIST FCN Model')
#plt.show()
# Save the plot as an image file
plt.savefig('training_accuracy.png')