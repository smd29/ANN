#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network

# ### Importing the libraries

# In[18]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras.utils as image


# In[2]:


tf.__version__


# ## Part 1 - Data Preprocessing

# ### Preprocessing the Training set

# In[2]:


train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip=True)


# In[3]:


training_dataset = train_datagen.flow_from_directory('dataset/training_set', target_size=(64,64), batch_size = 32, class_mode='binary')


# ### Preprocessing the Test set

# In[4]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[5]:


test_dataset = test_datagen.flow_from_directory('dataset/test_set',target_size=(64,64),batch_size=32,class_mode='binary')


# ## Part 2 - Building the CNN

# ### Initialising the CNN

# In[6]:


cnn = tf.keras.models.Sequential()


# ### Step 1 - Convolution

# In[7]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size = 3, activation='relu', input_shape =[64,64,3]))


# ### Step 2 - Pooling

# In[8]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# ### Adding a second convolutional layer & a pooling layer

# In[9]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size = 3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# ### Step 3 - Flattening

# In[10]:


cnn.add(tf.keras.layers.Flatten())


# ### Step 4 - Full Connection

# In[11]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# ### Step 5 - Output Layer

# In[12]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# ## Part 3 - Training the CNN

# ### Compiling the CNN

# In[13]:


cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# ### Training the CNN on the Training set and evaluating it on the Test set

# In[14]:


cnn.fit(x = training_dataset, validation_data = test_dataset, epochs = 25)


# ## Part 4 - Making a single prediction

# In[19]:


test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image/255.0)
training_dataset.class_indices
if result[0][0] > 0.5:
  prediction = 'dog'
else:
  prediction = 'cat'


# In[20]:


print(prediction)


# In[ ]:




