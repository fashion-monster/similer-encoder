
# coding: utf-8

# # Auto Encoderモデルの定義など

# In[ ]:


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(604,604,3))  # adapt this if using `channels_first` image data format
# input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

def ae_model(img):
    print(img)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(img)
    print(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    print(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    print(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    print(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    print(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='encoded')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    print(x)
    x = UpSampling2D((2, 2))(x)
    print(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    print(x)
    x = UpSampling2D((2, 2))(x)
    print(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    print(x)
    x = UpSampling2D((2, 2))(x)
    print(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    print(decoded)
    return encoded, decoded

encoded, decoded = ae_model(input_img)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# # 画像入力のヘルパ

# In[ ]:


def load_img(path):
    from PIL import Image
    image = Image.open(path)
    return image

def reshape(image):
    """
    Making gotten images regular size.
    Arg:
        image: 3-D Tensor

    Return:
        reshaped: reshaped images
    """
    import tensorflow as tf
    import tensorflow.contrib.eager as tfe

    print(tf.__version__)
    tfe.enable_eager_execution()
    max_size = tf.reduce_max(tf.shape(image))
    new_height = 604
    new_width = 604

    reshaped = tf.image.resize_images(tf.image.resize_image_with_crop_or_pad(image, max_size, max_size),[new_height, new_width])
    return reshaped

def size_decision(image):
    """
    Helper function.
    Returning longer edge size.

    Arg:
        image: 3-D Tensor
    Return:
        size: longer edge size
    """
    return tf.reduce_max(tf.shape(image))


def helper():
    import os
    import numpy as np
    from keras.preprocessing.image import  img_to_array, array_to_img
    train = os.listdir('train')
    valid = os.listdir('valid')
    
    t_img_path = ['train/'+x for x in train]
    v_img_path = ['valid/'+x for x in valid]
    
    t_img = []
    for x in t_img_path:
        x = load_img(x)
        x = img_to_array(x)
        x = reshape(x)
        t_img.append(x)
        
    v_img = []
    for x in v_img_path:
        x = load_img(x)
        x = img_to_array(x)
        x = reshape(x)
        v_img.append(x)
        
    return np.asarray(t_img), np.asarray(v_img)


# # 画像などの処理

# In[ ]:


from keras.datasets import mnist
from keras.preprocessing.image import  img_to_array, array_to_img

import numpy as np

# (x_train, _), (x_test, _) = mnist.load_data()

x_train, x_test = helper()
for x in x_train:
    print x.shape
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
# x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

x_train = np.reshape(x_train, (len(x_train), 604, 604, 3))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 604, 604, 3))  # adapt this if using `channels_first` image data format


# In[ ]:


from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=2000,
                batch_size=12,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


# # Validationしてみたり

# In[ ]:


from keras.preprocessing.image import load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

decoded_imgs = autoencoder.predict(x_test)

# image = array_to_img(decoded_imgs[0].reshape(604, 604, 3))
# plt.imshow(np.asarray(image))
# plt.show()

n = len(decoded_imgs)
plt.figure(figsize=(16, 8))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(604, 604, 3))
    plt.colors()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n +1)
    plt.imshow(decoded_imgs[i].reshape(604, 604, 3))
    plt.colors()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# # 中間層取り出しと類似度

# In[ ]:


import numpy as np
layer_name = 'encoded'
intermediate_layer_model = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.get_layer(layer_name).output)

encoded_train = intermediate_layer_model.predict(x_train)
encoded_valid = intermediate_layer_model.predict(x_test)

def get_nearest_value(nplist, num):
    idx = np.abs(nplist - num).argmin()
    return idx

def euclid(y,x):
    '''
    Euclid distance
    '''
    return np.linalg.norm(y-x)

def cosie(y,x):
    '''
    cosine simmiler(?)
    '''
    import scipy.spatial.distance as dis
    
    return dis.cosine(y.flatten(), x.flatten())

print 'encoded_train'
# counter = 0
# for x in encoded_valid:
#     diff = [np.linalg.norm(y-x) for y in encoded_train]
#     t_image = x_train[int(np.asarray(diff).argmin())]
#     v_image = x_test[counter]
#     load_img(array_to_img(t_image.reshape(604, 604, 3)))
#     load_img(array_to_img(v_image.reshape(604, 604, 3)))
    
#     counter+=1
    
    
n = len(encoded_valid)
plt.figure(figsize=(16, 8))
for i in range(n):
    diff = [cosie(y, encoded_valid[i]) for y in encoded_train]
    t_image = x_train[int(np.asarray(diff).argmin())]
    v_image = x_test[i]
    
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(v_image.reshape(604, 604, 3))
    plt.colors()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(t_image.reshape(604, 604, 3))
    print(i+1, i+n+1)
    plt.colors()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# # ラムダ式使うとこうなるっぽい

# In[ ]:


a = 1
b = 2
c = 3
def add():
    return lambda x,y: x+y*c

def subtract():
    return lambda x,y: x-y
add()(add()(a,b),c)


# In[ ]:




