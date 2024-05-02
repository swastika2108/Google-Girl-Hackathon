import keras
from keras.layers import Conv2D, MaxPool2D, Dropout

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