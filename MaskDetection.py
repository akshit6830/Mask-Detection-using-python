#!/usr/bin/env python
# coding: utf-8

# In[14]:


import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import os

Dataset = 'data'
Data_Dir = os.listdir(Dataset)
print(Data_Dir)

img_rows , img_cols = 112 ,112

images = []
labels = []

for category in Data_Dir :
    folder_path = os.path.join(Dataset , category)
    for img in os.listdir(folder_path) :
        img_path = os.path.join(folder_path , img)
        img = cv2.imread(img_path)

        try :
            gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

            resized_img = cv2.resize(gray , (img_rows,img_cols))
            images.append(resized_img)
            labels.append(category)
        except Exception as e :
            print('Exception : ' , e )


# In[15]:



images = np.array(images)/255.0
images = np.reshape(images , (images.shape[0],img_rows,img_cols,1))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
labels = np.array(labels)

(train_X , test_X , train_y , test_y ) = train_test_split(images , labels , test_size=0.25 , random_state=0)


# In[16]:


from keras.models import Sequential 
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D , MaxPooling2D

num_classes = 2
batch_size = 32 

model = Sequential()

model.add(Conv2D(64 , (3,3) , input_shape=(img_rows , img_cols , 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128 , (3,3) ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(64 , activation='relu'))
model.add(Dense(num_classes , activation='softmax'))

model.summary()


# In[17]:


from keras.utils.vis_utils import plot_model

plot_model(model , to_file='data/mask_architecture.png')


# In[18]:


from keras.optimizers import Adam

epochs = 100

model.compile(loss='categorical_crossentropy' ,
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

fitted_model = model.fit(
            train_X , 
            train_y ,
            epochs = epochs , 
            validation_split = 0.25)


# In[19]:


model.save('model_100.h5')


# In[22]:


from matplotlib import pyplot as plt 

plt.plot(fitted_model.history['loss'] , 'r' , label="training_loss")
plt.plot(fitted_model.history['val_loss'] , label="validation_loss")
plt.xlabel('Number of Epochs')
plt.ylabel('Loss Value')
plt.legend()
plt.show()

plt.plot(fitted_model.history['accuracy'] , 'r' , label="training_accuracy")
plt.plot(fitted_model.history['val_accuracy'] , label="validation_accuracy")
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy Value')
plt.legend()
plt.show()


# In[ ]:




