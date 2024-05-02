#!/usr/bin/env python
# coding: utf-8

# # Semantic Segmentation of Flood Events using U-Net for Real-Time Flood Hazard Assessment(Google Girl's Hackathon Project)
# 1. Submitted by-Swastika Satya
# 2. GOC ID - 954790390322

# # A.Import

# In[1]:


# Common
import os
import cv2 as cv
from keras.metrics import MeanIoU
from tqdm import tqdm

# data 
import numpy as np 
from keras.preprocessing.image import load_img, img_to_array

# viz
import matplotlib.pyplot as plt

# Model
import keras 
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization, Input, Conv2DTranspose, concatenate, GlobalAveragePooling2D, Dense
from keras import Sequential 
from keras.callbacks import EarlyStopping, ModelCheckpoint


# # **B.Data**

# In[2]:


def show_image(image, cmap=None, title=None):
  plt.imshow(image, cmap=cmap)
  if title is not None: plt.title(title)
  plt.axis('off')


# In[3]:


images = []
mask = []

image_path = '../input/satellite-images-of-water-bodies/Water Bodies Dataset/Images/'
mask_path = '../input/satellite-images-of-water-bodies/Water Bodies Dataset/Masks/'

image_names = sorted(next(os.walk(image_path))[-1])
mask_names = sorted(next(os.walk(mask_path))[-1])

if image_names == mask_names:
  print('Image and Mask are corretly Placed!!')


# In[4]:


SIZE = 128
images = np.zeros(shape=(len(image_names),SIZE, SIZE, 3))
masks = np.zeros(shape=(len(image_names),SIZE, SIZE, 1))

for id in tqdm(range(len(image_names)), desc="Images"):
  path = image_path + image_names[id]
  img = img_to_array(load_img(path)).astype('float')/255.
  img = cv.resize(img, (SIZE,SIZE), cv.INTER_AREA)
  images[id] = img

for id in tqdm(range(len(mask_names)), desc="Mask"):
  path = mask_path + mask_names[id]
  mask = img_to_array(load_img(path)).astype('float')/255.
  mask = cv.resize(mask, (SIZE,SIZE), cv.INTER_AREA)
  masks[id] = mask[:,:,:1]


# In[5]:


plt.figure(figsize=(15,15))
for i in range(1,21):
  plt.subplot(5,4,i)

  if i%2!=0:
    id = np.random.randint(len(images))
    show_image(images[id], title="Orginal Image")
  elif i%2==0:
    show_image(masks[id].reshape(128,128), title="Mask Image", cmap='gray')

plt.tight_layout()
plt.show()


# # **C.Model-U-Net**

# In[6]:


X, y = images[:int(len(images)*0.9)], masks[:int(len(images)*0.9)]
test_X, test_y = images[int(len(images)*0.9):], masks[int(len(images)*0.9):]


# # Encoder

# In[7]:


# Contraction 
class EncoderBlock(keras.layers.Layer):
  
  def __init__(self, filters, rate=None, pooling=True):
    super(EncoderBlock,self).__init__()
    self.filters = filters
    self.rate = rate
    self.pooling = pooling
    self.conv1 = Conv2D(self.filters,kernel_size=3,strides=1,padding='same',activation='relu',kernel_initializer='he_normal')
    self.conv2 = Conv2D(self.filters,kernel_size=3,strides=1,padding='same',activation='relu',kernel_initializer='he_normal')
    if self.pooling: self.pool = MaxPool2D(pool_size=(2,2))
    if self.rate is not None: self.drop = Dropout(rate)
    
  def call(self, inputs):
    x = self.conv1(inputs)
    if self.rate is not None: x = self.drop(x)
    x = self.conv2(x)
    if self.pooling: 
      y = self.pool(x)
      return y, x
    else:
      return x
  
  def get_config(self):
    base_config = super().get_config()
    return {
        **base_config, 
        "filters":self.filters,
        "rate":self.rate,
        "pooling":self.pooling
    }


# # Decoder

# In[8]:


# Expansion
class DecoderBlock(keras.layers.Layer):
  
  def __init__(self, filters, rate=None, axis=-1):
    super(DecoderBlock,self).__init__()
    self.filters = filters
    self.rate = rate
    self.axis = axis
    self.convT = Conv2DTranspose(self.filters,kernel_size=3,strides=2,padding='same')
    self.conv1 = Conv2D(self.filters, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding='same')
    if rate is not None: self.drop = Dropout(self.rate)
    self.conv2 = Conv2D(self.filters, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding='same')
    
  def call(self, inputs):
    X, short_X = inputs
    ct = self.convT(X)
    c_ = concatenate([ct, short_X], axis=self.axis)
    x = self.conv1(c_)
    if self.rate is not None: x = self.drop(x)
    y = self.conv2(x)
    return y
  
  def get_config(self):
    base_config = super().get_config()
    return {
        **base_config, 
        "filters":self.filters,
        "rate":self.rate,
        "axis":self.axis,
    }


# In[9]:


# Callback 
class ShowProgress(keras.callbacks.Callback):
  def __init__(self, save=False):
    self.save = save
  def on_epoch_end(self, epoch, logs=None):
    id = np.random.randint(len(images))
    real_img = images[id][np.newaxis,...]
    pred_mask = self.model.predict(real_img).reshape(128,128)
    proc_mask1 = post_process(pred_mask)
    proc_mask2 = post_process(pred_mask, threshold=0.5)
    proc_mask3 = post_process(pred_mask, threshold=0.9)
    mask = masks[id].reshape(128,128)

    plt.figure(figsize=(10,5))

    plt.subplot(1,6,1)
    show_image(real_img[0], title="Orginal Image")

    plt.subplot(1,6,2)
    show_image(pred_mask, title="Predicted Mask", cmap='gray')
    
    plt.subplot(1,6,3)
    show_image(mask, title="Orginal Mask", cmap='gray')

    plt.subplot(1,6,4)
    show_image(proc_mask1, title="Processed@0.4", cmap='gray')

    plt.subplot(1,6,5)
    show_image(proc_mask2, title="Processed@0.5", cmap='gray')

    plt.subplot(1,6,6)
    show_image(proc_mask3, title="Processed@0.9", cmap='gray')

    plt.tight_layout()
    if self.save: plt.savefig("Progress-{}.png".format(epoch+1))
    plt.show()


# In[10]:


# Post Process
def post_process(image,threshold=0.4):
  return image>threshold


# In[11]:


inputs= Input(shape=(SIZE,SIZE,3))

# Contraction 
p1, c1 = EncoderBlock(16,0.1)(inputs)
p2, c2 = EncoderBlock(32,0.1)(p1)
p3, c3 = EncoderBlock(64,0.2)(p2)
p4, c4 = EncoderBlock(128,0.2)(p3)

# Encoding Layer
c5 = EncoderBlock(256,rate=0.3,pooling=False)(p4) 

# Expansion
d1 = DecoderBlock(128,0.2)([c5,c4]) # [current_input, skip_connection]
d2 = DecoderBlock(64,0.2)([d1,c3])
d3 = DecoderBlock(32,0.1)([d2,c2])
d4 = DecoderBlock(16,0.1, axis=3)([d3,c1])

# Outputs 
outputs = Conv2D(1,1,activation='sigmoid')(d4)

unet = keras.models.Model(
    inputs=[inputs],
    outputs=[outputs],
)


# # Adam Optimizer

# In[12]:


unet.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)


# In[13]:


callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint('UNet-01.h5',save_best_only=True),
    ShowProgress(save=True)
]


# In[14]:


unet.summary()


# In[15]:


get_ipython().run_cell_magic('time', '', 'with tf.device("/GPU:0"):\n  results = unet.fit(\n      X, y,\n      epochs=100,\n      callbacks=callbacks,\n      validation_split=0.1,\n      batch_size=16\n  )\n')


# * After looking at the produced images, it could be said that the model is actually able to identify the regions of water but its confused between water and forest when the color difference between them is low.
# 
# * It could also be noticed that the model is actually preserving some of these spatial features which we actually don't want in the final image. That's why I have post processed it.
# 
# * Overall, the model is great. It's actually able to identify the regions correctly. Well, not exactly up to the point, but roughly correct enough to be convincing.

# # **D.Adaptive Threshold UNET**

# In[16]:


# Callback 
class ShowProgress(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    id = np.random.randint(len(images))
    real_img = images[id][np.newaxis,...]
    pred_mask = self.model.predict(real_img).reshape(128,128)
    proc_mask1 = post_process(pred_mask, threshold=0.5)
    thresh = float(np.mean(pred_mask) + np.min(pred_mask))
    proc_mask2 = post_process(pred_mask, threshold=thresh)
    mask = masks[id].reshape(128,128)

    plt.figure(figsize=(15,6))

    plt.subplot(1,5,1)
    show_image(real_img[0], title="Orginal Image")

    plt.subplot(1,5,2)
    show_image(pred_mask, title="Predicted Mask", cmap='gray')
    
    plt.subplot(1,5,3)
    show_image(mask, title="Orginal Mask", cmap='gray')

    plt.subplot(1,5,4)
    show_image(proc_mask1, title="Processed@0.4", cmap='gray')

    plt.subplot(1,5,5)
    show_image(proc_mask2, title="Processed@{:.2}".format(thresh), cmap='gray')

    plt.tight_layout()
    plt.show()

# Post Process
def post_process(image,threshold): return image>threshold


# In[17]:


inputs= Input(shape=(SIZE,SIZE,3))

# Contraction 
p1, c1 = EncoderBlock(16,0.1)(inputs)
p2, c2 = EncoderBlock(32,0.1)(p1)
p3, c3 = EncoderBlock(64,0.2)(p2)
p4, c4 = EncoderBlock(128,0.2)(p3)

# Encoding Layer
c5 = EncoderBlock(256,rate=0.3,pooling=False)(p4) 

# Expansion
d1 = DecoderBlock(128,0.2)([c5,c4]) # [current_input, skip_connection]
d2 = DecoderBlock(64,0.2)([d1,c3])
d3 = DecoderBlock(32,0.1)([d2,c2])
d4 = DecoderBlock(16,0.1, axis=3)([d3,c1])

# Outputs
outputs = Conv2D(1,1,activation='sigmoid')(d4)

unet = keras.models.Model(
    inputs=[inputs],
    outputs=[outputs],
)


# In[18]:


unet.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)


# In[19]:


callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint('Adaptive-UNet.h5',save_best_only=True),
    ShowProgress()
]


# In[20]:


get_ipython().run_cell_magic('time', '', 'with tf.device("/GPU:0"):\n  results = unet.fit(\n      X, y,\n      epochs=100,\n      callbacks=callbacks,\n      validation_split=0.1,\n      batch_size=32\n  )\n')


# You can decide what type of threshold you're willing to use. I kind of like loved the idea of using the adaptive threshold. Now, if you somehow find the best threshold value, then you can actually get the best masking, because the model is able to predict the mask correctly almost 99% of the time, but some of the regions are blurry and somehow, you will have to tackle them. My technique was to add this Adaptive Threshold your technique could be something else.

# # **E.Testing**

# In[21]:


plt.figure(figsize=(5,28))
n=0
for i in range(1,31):
    plt.subplot(10,3,i)
    
    if n==0:
        id = np.random.randint(len(images))
        real_img = images[id][np.newaxis,...]
        pred_mask = unet.predict(real_img).reshape(128,128)
        mask = masks[id].reshape(128,128)
        show_image(real_img[0], title="Real Image")
        n+=1
    elif n==1:
        show_image(pred_mask, title="Predicted Mask", cmap='gray')
        n+=1
    elif n==2:
        show_image(mask, title="Original Mask", cmap='gray')
        n=0


# The **high clarity** that you are able to find in the **original mask** is probably because they have used a **threshold** and that makes it **highly clear**. But as our model is **producing that image**, it's pretty good that it's able to **produce images with mask**, even if they are a **little bit of blurry**. 

# In[22]:


def process_single_image(image_path, SIZE):
    image = np.zeros(shape=(1, SIZE, SIZE, 3))

    img = img_to_array(load_img(image_path)).astype('float') / 255.
    img = cv.resize(img, (SIZE, SIZE), cv.INTER_AREA)
    image[0] = img

    return image


# # F.Test on Pre and Post Flood Images

# In[24]:


# Assuming process_single_image and unet.predict functions are already defined

new_SIZE = 128

# Get a list of all files in the folder

# Iterate over each file
for i in range(1,3):
    input_before="/kaggle/input/flood-detection-test/before/"+"{:03d}".format(i) +".png"
    input_after="/kaggle/input/flood-detection-test/after/" + "{:03d}".format(i) +".png" 

        # Process the image
    new_image_before = process_single_image(input_before, new_SIZE)
        
        # Predict mask for the new_image
    pred_mask_before = unet.predict(new_image_before).reshape(new_SIZE, new_SIZE)
        
         # Process the image
    new_image_after = process_single_image(input_after, new_SIZE)
        
        # Predict mask for the new_image
    pred_mask_after = unet.predict(new_image_after).reshape(new_SIZE, new_SIZE)
        
        
        
        # Visualize the new_image and the predicted mask
    plt.figure(figsize=(12, 28))  # Adjust the figure size as needed
        
        # Display the new_image
    plt.subplot(1, 4, 1)
    plt.imshow(new_image_before[0])  # Assuming new_image is a single image array
    plt.title("New Image")
    plt.axis('off')
        
        # Display the predicted mask
    plt.subplot(1, 4, 2)
    plt.imshow(pred_mask_before, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')
        
    plt.subplot(1, 4, 3)
    plt.imshow(new_image_after[0])  # Assuming new_image is a single image array
    plt.title("New Image")
    plt.axis('off')
        
        # Display the predicted mask
    plt.subplot(1, 4, 4)
    plt.imshow(pred_mask_after, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')
        
    plt.tight_layout()
    plt.show()


# In[25]:


# Assuming process_single_image and unet.predict functions are already defined

new_SIZE = 128

# Iterate over each file
for i in range(1,3):
    input_before="/kaggle/input/flood-detection-test/before/"+"{:03d}".format(i) +".png"
    input_after="/kaggle/input/flood-detection-test/after/" + "{:03d}".format(i) +".png" 

    # Process the image before the event
    new_image_before = process_single_image(input_before, new_SIZE)
    
    # Predict mask for the image before the event
    pred_mask_before = unet.predict(new_image_before).reshape(new_SIZE, new_SIZE)
    
    # Process the image after the event
    new_image_after = process_single_image(input_after, new_SIZE)
    
    # Predict mask for the image after the event
    pred_mask_after = unet.predict(new_image_after).reshape(new_SIZE, new_SIZE)
    
    # Compute the absolute difference between the two masks
    mask_difference = abs(pred_mask_before - pred_mask_after)
    
    # Threshold the mask difference to identify significant changes
    threshold = 0.3  # You can adjust this threshold as needed
    significant_changes = mask_difference > threshold
    
    # Check if flood is present by determining if there are significant changes in the masks
    if significant_changes.any():
        print("Flood is detected in image", i)
    else:
        print("No flood detected in image", i)


# In[ ]:




