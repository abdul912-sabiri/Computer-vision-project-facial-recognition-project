#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
training_images = np.load("C:/Users/hp/Downloads/project  Expression Classification from Facial Images/sir/train_images.npy")
training_labels = np.load("C:/Users/hp/Downloads/project  Expression Classification from Facial Images/sir/train_labels.npy")
testing_images = np.load ("C:/Users/hp/Downloads/project  Expression Classification from Facial Images/sir/test_images.npy")
test_labels = np.load("C:/Users/hp/Downloads/project  Expression Classification from Facial Images/sir/test_labels.npy")


# In[2]:


print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(test_labels.shape)


# In[3]:


label_names = ['','angry','disgust','fear','happy','neutral','sad','surprise']
print(label_names)


# In[4]:


import matplotlib.pyplot as plt
plt.imshow(training_images[26000])
print("label = ",label_names[training_labels[26000]])


# In[5]:


plt.imshow(training_images[20100])
print("label = ",label_names[training_labels[20100]])


# In[6]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPool2D


# In[7]:


model=Sequential()


# In[8]:


model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(48,48,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
#model.add(BatchNormalization())
#model.add(Dropout(rate=0.50))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(156,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))
model.summary()


# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the data
training_images = np.load("C:/Users/hp/Downloads/project  Expression Classification from Facial Images/sir/train_images.npy")
training_labels = np.load("C:/Users/hp/Downloads/project  Expression Classification from Facial Images/sir/train_labels.npy")
testing_images = np.load("C:/Users/hp/Downloads/project  Expression Classification from Facial Images/sir/test_images.npy")
test_labels = np.load("C:/Users/hp/Downloads/project  Expression Classification from Facial Images/sir/test_labels.npy")

# Normalize the images to values between 0 and 1
training_images = training_images / 255.0
testing_images = testing_images / 255.0

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
   # layers.BatchNormalization(),
    #layers.Dropout(rate=0.50),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256,(3,3),padding='same',activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Conv2D(512,(3,3),padding='same',activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 classes for the emotions
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Reshape the training and testing data to include a single channel (grayscale)
training_images = training_images.reshape(training_images.shape[0], 48, 48, 1)
testing_images = testing_images.reshape(testing_images.shape[0], 48, 48, 1)

# Ensure that the label values are within the valid range [0, 6]
training_labels = np.clip(training_labels, 0, 6)
test_labels = np.clip(test_labels, 0, 6)

# Train the model
model.fit(training_images, training_labels, epochs=20, batch_size=32, validation_split=0.4)

# Evaluate the model on the test set
#test_loss, test_accuracy = model.evaluate(testing_images, test_labels)
#print("Test accuracy:", test_accuracy)


# In[13]:


model.save(r"C:/Users/hp/Downloads/project  Expression Classification from Facial Images/sir/emotion_detection_model.h5")
# Print a message indicating that the model has been saved
print("Model saved successfully.")


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the test data (if not already loaded)
testing_images = np.load("C:/Users/hp/Downloads/project  Expression Classification from Facial Images/sir/test_images.npy")
test_labels = np.load("C:/Users/hp/Downloads/project  Expression Classification from Facial Images/sir/test_labels.npy")

label_names = [ '','angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# In[170]:


#plt.imshow(testing_images[2])
#print("label = ",label_names[testing_images[2]])


# In[22]:


testing_images = testing_images / 255.0

# Reshape the testing data to include a single channel (grayscale)
testing_images = testing_images.reshape(testing_images.shape[0], 48, 48, 1)

# Ensure that the label values are within the valid range [0, 6]
test_labels = np.clip(test_labels, 0, 5)

# Load the trained model
loaded_model = load_model(r"C:/Users/hp/Downloads/project  Expression Classification from Facial Images/sir/New folder/emotion_detection_model.h5")


# Evaluate the model on the test set
test_loss, test_accuracy = loaded_model.evaluate(testing_images, test_labels)
print("Test Accuracy:", test_accuracy)

# Make predictions on the test set
predicted_labels = loaded_model.predict(testing_images)
predicted_classes = np.argmax(predicted_labels, axis=1)

# Label names for emotions
label_names = ['', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
#data=pd.read_csv('C:/Users/hp/Downloads/project  Expression Classification from Facial Images/sir/67.jpg')
# Visualize some predictions
num_samples_to_visualize = 36
plt.figure(figsize=(18, 18))
for i in range(num_samples_to_visualize):
    plt.subplot(6, 6, i + 1)
    plt.imshow(testing_images[i].reshape(48, 48), cmap='gray')
    predicted_label = label_names[predicted_classes[i]]
    actual_label = label_names[test_labels[i]]
   # plt.title(f"Predicted: {predicted_label}")
    plt.title(f"Predicted: {predicted_label}\nActual: {actual_label}")
    plt.axis('off')

plt.show()


# In[ ]:




