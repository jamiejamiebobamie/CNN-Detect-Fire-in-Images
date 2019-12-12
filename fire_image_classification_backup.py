"""
    - Fire/noFire image data classification
    - The data is unbalance

    - Without data augmentation
        - Change the image input size
        - Avoid model overfitting -> track both training accuracy and
            also more importantly test (validation) accuracy
        - Report confusion matrix

    - Apply data augmentation for fire images
        - Avoid model overfitting -> track both training accuracy and
            also more importantly test (validation) accuracy
        - Report confusion matrix
"""

import glob
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image


# 111 images
files = glob.glob("Fire-Detection-Image-Dataset/Fire_images/*")
ls_fire = []
for i in files:
    ls_fire.append(['Fire-Detection-Image-Dataset/Fire_images',
        i.split("/")[-1], '1'])

# 542 images
files = glob.glob("Fire-Detection-Image-Dataset/No_Fire_images/*")
ls_no_fire = []
for i in files:
    ls_no_fire.append(['Fire-Detection-Image-Dataset/No_Fire_images',
        i.split("/")[-1], '0'])

# there are 4.88x more "no_fire" images than "fire" images.
df_fire = pd.DataFrame(ls_fire, columns=['Folder', 'filename', 'label'])
df_no_fire = pd.DataFrame(ls_no_fire, columns=['Folder', 'filename', 'label'])

# if we perform a very simple data augmentation of rotating the fire images 90
# degrees three times we'll have a total of 444 fire images and 981 total images.
# a more balanced dataset :).
target_dataframe_size = len(df_fire)*4 + len(df_no_fire)
target_dataframe_size = target_dataframe_size

frames = [df_fire, df_no_fire]

df_total = pd.concat(frames)

# split the data into train and test
train, test = train_test_split(df_total, test_size=0.2)

input_shape = (128, 128, 3)
num_classes = 1
batch_size = 9
epochs = 3


def data_gen(df, batch_size):
    while True:
        x_batch = np.zeros((batch_size, 128, 128, 3))
        y_batch = np.zeros((batch_size, 1))
        for j in range(int(target_dataframe_size/batch_size)):
            b = 0
            for m, k, n in zip(df['filename'].values[j*batch_size:(j+1)*batch_size],
                            df['label'].values[j*batch_size:(j+1)*batch_size],
                            df['Folder'].values[j*batch_size:(j+1)*batch_size]):
                img = Image.open('{}/{}'.format(n, m)).convert("RGB")
                image_red = img.resize((128, 128))
                X = img_to_array(image_red)
                X = np.array(X)
                X = X.astype('float32')
                X /= 255
                x_batch[b] = X
                if k == '1':
                    y_batch[b] = 1.0
                else:
                    y_batch[b] = 0.0
                b += 1
            yield (x_batch, y_batch)

def data_gen_with_aug(df, batch_size):
    while True:
        x_batch = np.zeros((batch_size, 128, 128, 3))
        y_batch = np.zeros((batch_size, 1))
        for j in range(int(target_dataframe_size/batch_size)):
            b = 0
            for m, k, n in zip(df['filename'].values[j*batch_size:(j+1)*batch_size],
                            df['label'].values[j*batch_size:(j+1)*batch_size],
                            df['Folder'].values[j*batch_size:(j+1)*batch_size]):
                if b == batch_size:
                    b = 0
                    yield (x_batch, y_batch)
                img = Image.open('{}/{}'.format(n, m)).convert("RGB")
                image_red = img.resize((128, 128))
                X = img_to_array(image_red)
                X = np.array(X)
                X = X.astype('float32')
                X /= 255
                x_batch[b] = X
                if k == '1':
                    y_batch[b] = 1.0
                    b += 1
                    # image augmentation
                    for _ in range(3):
                        if b == batch_size:
                            b = 0
                            yield (x_batch, y_batch)
                        rotate90(X)
                        x_batch[b] = X
                        y_batch[b] = 1.0
                        b += 1
                else:
                    y_batch[b] = 0.0
                    b += 1
            yield (x_batch, y_batch)

# https://www.geeksforgeeks.org/rotate-matrix-90-degree-without-using-extra-space-set-2/
# Function to anticlockwise rotate matrix
# by 90 degree
def rotate90(arr):
    if len(arr) < 0:
        return
    # pretty sure this breaks if the array isn't 2D...
    if len(arr[0]) < 0:
        return
    R = len(arr)
    C = len(arr[0])

    # transpose of matrix
    for i in range(R):
        for j in range(i, C):
            t = arr[i][j]
            arr[i][j] = arr[j][i]
            arr[j][i] = t

    # reverseColumns
    for i in range(C):
        j = 0
        k = C-1
        while j < k:
            t = arr[j][i]
            arr[j][i] = arr[k][i]
            arr[k][i] = t
            j += 1
            k -= 1


# building the model.
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

# compiling the model.
optimizer = keras.optimizers.Adadelta()
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

# fitting the model.
steps = len(train) // batch_size
model.fit_generator(generator=data_gen(train, batch_size=batch_size),
                    steps_per_epoch=steps,
                    epochs=epochs, verbose=1)

steps = len(test) // 1 # 131 is the test dataset's length and is a prime number
validation_generator = data_gen(test, batch_size=1)
Y_pred = model.predict_generator(validation_generator, steps)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
predictions = []
for i, val in enumerate(y_pred):
    predictions.append([val])
predictions = np.asarray(predictions)

y_true = test[['label']].to_numpy()
for arr in y_true:
    arr[0] = int(arr[0])
print(confusion_matrix(y_true, predictions)) # BROKEN

# pickling the model.
# joblib.dump(model, 'fire_no_fire_classifire.pkl')
