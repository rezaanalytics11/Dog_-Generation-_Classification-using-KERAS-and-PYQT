import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import layers,models
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras import layers, models, optimizers
from keras import backend as K
import numpy as np
from keras import utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,\
     Dropout,Flatten,Dense,Activation,\
     BatchNormalization


train_dataset_path=r'C:\Users\Ariya Rayaneh\Desktop\dogs'
labels=os.listdir(r'C:\Users\Ariya Rayaneh\Desktop\dogs')

Image_Width=224
Image_Height=224
Image_Size=(Image_Width,Image_Height)
Image_Channels=3
batch_size=15



filenames=os.listdir(r"C:\Users\Ariya Rayaneh\Desktop\dogs\train")
categories=[]
for f_name in filenames:
    category=f_name.split('_')[0]
    if category=='n02085620':
        categories.append(0)
    elif category=='n02093647':
        categories.append(1)
    else:
        categories.append(2)

df=pd.DataFrame({
    'filename':filenames,
    'category':categories
})

# model=Sequential()
# model.add(Conv2D(32,(3,3),activation='relu',input_shape=(Image_Width,Image_Height,Image_Channels)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(128,(3,3),activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(512,activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(3,activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',
#   optimizer='rmsprop',metrics=['accuracy'])
# model.summary()
#
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# earlystop = EarlyStopping(patience = 10)
# learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',patience = 2,verbose = 1,factor = 0.5,min_lr = 0.00001)
# callbacks = [earlystop,learning_rate_reduction]
#
# df["category"] = df["category"].replace({0:'Chihuahua',1:'Bedlington_terrier',2:'Mexican_hairless'})
# train_df,validate_df = train_test_split(df,test_size=0.20,
#   random_state=42)
#
# train_df = train_df.reset_index(drop=True)
# validate_df = validate_df.reset_index(drop=True)
# total_train=train_df.shape[0]
# total_validate=validate_df.shape[0]
#
#
#
# train_datagen = ImageDataGenerator(rotation_range=15,
#                                 rescale=1./255,
#                                 shear_range=0.1,
#                                 zoom_range=0.2,
#                                 horizontal_flip=True,
#                                 width_shift_range=0.1,
#                                 height_shift_range=0.1
#                                 )
#
# train_generator = train_datagen.flow_from_dataframe(train_df,
#                                                  r"C:\Users\Ariya Rayaneh\Desktop\dogs\train",x_col='filename',y_col='category',
#                                                  target_size=Image_Size,
#                                                  class_mode='categorical',
#                                                  batch_size=batch_size)
#
# validation_datagen = ImageDataGenerator(rescale=1./255)
# validation_generator = validation_datagen.flow_from_dataframe(
#     validate_df,
#     r"C:\Users\Ariya Rayaneh\Desktop\dogs\train",
#     x_col='filename',
#     y_col='category',
#     target_size=Image_Size,
#     class_mode='categorical',
#     batch_size=batch_size
# )
#
# test_datagen = ImageDataGenerator(rotation_range=15,
#                                 rescale=1./255)
#
# # test_generator = test_datagen.flow_from_dataframe(train_df,
# #                                                  r'C:\Users\Ariya Rayaneh\Desktop\dogs\test',x_col='filename',y_col='category',
# #                                                  target_size=Image_Size,
# #                                                  class_mode='categorical',
# #                                                  batch_size=batch_size)
#
#
# epochs=20
# history = model.fit_generator(
#     train_generator,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=total_validate//batch_size,
#     steps_per_epoch=total_train//batch_size,
#     callbacks=callbacks
# )
#
# model.save(r"C:\Users\Ariya Rayaneh\Desktop\dog_model.h5")

model = keras.models.load_model(r"C:\Users\Ariya Rayaneh\Desktop\dog_model.h5")

results={
    0:'Chihuahua',
    1:'Bedlington_terrier',
    2:'Mexican_hairless'
}

from PIL import Image
import numpy as np
im=cv2.imread(r'C:\Users\Ariya Rayaneh\Desktop\n02113978_2054.jpg')
im=cv2.resize(im,Image_Size)
print(im,5*'\n')
im=np.expand_dims(im,axis=0)
print(im)
im=np.array(im)
im=im/255.0
pred=np.argmax(model.predict([im])[0])
#pred=model.predict_classes([im])[0]
print(pred,results[pred])