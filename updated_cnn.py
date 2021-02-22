import tensorflow_addons as tfa
import keras 
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
import numpy as np
import cv2

#img  = cv2.imread("/content/9.jpg", 0)
#img = np.pad(img, ((3,3), (3,3)), mode='constant', constant_values=0)
#img = cv2.resize(img, (28, 28))
#plt.imshow(img)

"""
#img = np.array(img).reshape(28, 28, 1)
#img = img.astype('float32')
#img = img/255
#print(img.shape)
#dumb = []
#dumb.append(img)
#dumb = np.array(dumb)
#print(dumb.shape)
y = 4
y = keras.utils.to_categorical(y, 10)
y = y.reshape(1,10)
print(y)
"""
#print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28,28,1)

# dum = x_train[0].reshape(28,28)
# plt.imshow(dum)
# dum = dum.reshape(1,28,28,1)
# dum = dum.astype('float32')
# dum = dum/255
x_train = x_train.astype('float32')
x_train = x_train/255
x_test = x_test.astype('float32')
x_test = x_test/255

#print(y_train[0])

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#print(y_train[0])

model = keras.models.Sequential()
model.add(tfa.layers.WeightNormalization(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = input_shape )))
model.add(tfa.layers.WeightNormalization(keras.layers.Conv2D(64, (3,3), activation='relu')))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tfa.layers.WeightNormalization(keras.layers.Conv2D(128, (3,3), activation='relu')))
model.add(keras.layers.Flatten())
model.add(tfa.layers.WeightNormalization(keras.layers.Dense(128, activation='relu')))
model.add(tfa.layers.WeightNormalization(keras.layers.Dense(10, activation='softmax')))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=15, validation_data=(x_test, y_test))
model.fit(train, ytrain, batch_size=1, epochs=5, validation_data=(x_test, y_test))

"""
#model.save('SolveSudoku-master\models\model1')

 model_json = model.to_json()
 with open("SolveSudoku-master\\models\\new_model.json", "w") as json_file:
     json_file.write(model_json)
 # serialize weights to HDF5
 model.save_weights("SolveSudoku-master\\models\\new_model.h5")
 print("Saved model to disk")

#print(img[0])
#print(x_train[0])

#print(model.predict(dumb))

"""