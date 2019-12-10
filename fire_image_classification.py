import glob
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from PIL import Image
from keras.preprocessing.image import img_to_array

files = glob.glob("Fire-Detection-Image-Dataset/Fire images/*")
ls_fire = []
for i in files:
    ls_fire.append(['Fire-Detection-Image-Dataset/Fire images', i.split("/")[-1], '1'])

files = glob.glob("Fire-Detection-Image-Dataset/No_Fire images/*")
ls_no_fire = []
for i in files:
    ls_no_fire.append(['Fire-Detection-Image-Dataset/No_Fire images',
        i.split("/")[-1], '0'])

# there are 4.88x more no_fire images than fire images.
df_fire = pd.DataFrame(ls_fire, columns=['Folder', 'filename', 'label'])
df_no_fire = pd.DataFrame(ls_no_fire, columns=['Folder', 'filename', 'label'])

# if we perform a very simple data augmentation of rotating the fire images 90
# degrees four times we'll have 333 more images for a total of 444 fire images.
# and 981 total images. a more balanced dataset :)
target_dataframe_size = len(df_fire)*4 + len(df_no_fire)
target_dataframe_size = target_dataframe_size

frames = [df_fire, df_no_fire]

df_total = pd.concat(frames)

input_shape = (128, 128, 3)
num_classes = 1
batch_size = 9
epochs = 5

def data_gen(df, batch_size):
    while True:
        x_batch = np.zeros((batch_size, 128, 128, 3))
        y_batch = np.zeros((batch_size, 1))
        # for j in range(int(len(df)/batch_size)):
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
                # print(b)
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

# sample code for data augmentation from the class repo.
# I couldn't figure out how to use the generators alongside or in place of the
# current generator so I decided to "write" custom code for data augmentation.
# - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
#
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(
#         'data/train',
#         target_size=(150, 150),
#         batch_size=32,
#         class_mode='binary')
#
# validation_generator = test_datagen.flow_from_directory(
#         'data/validation',
#         target_size=(150, 150),
#         batch_size=32,
#         class_mode='binary')
#
# model.fit_generator(
#         train_generator,
#         steps_per_epoch=2000,
#         epochs=50,
#         validation_data=validation_generator,
#         validation_steps=800)
# - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# fitting the model.
model.fit_generator(generator=data_gen(df_total, batch_size=batch_size),
                    steps_per_epoch=len(df_total) // batch_size,
                    epochs=epochs, verbose=1)

# pickling the model.

# testing the model on a single image.
