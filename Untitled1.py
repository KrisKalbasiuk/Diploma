
# coding: utf-8

# In[137]:

import glob, os, numpy as np
import theano as th, tensorflow as tf
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split


# In[161]:

path = "/home/kristina/Diploma/DataSet/Data/"
category = []
for i in glob.glob(path + "*"):
    category.append(i[len(path):])


# In[170]:

def get_pixels(image):
    size = image.size
    lst = np.array(image).flatten()
    lst = list(lst)
    for i in range(len(lst)):
        if lst[i] == False:
            lst[i] = 0
        else:
            lst[i] = 1     
    lst = np.array(lst).reshape(size[1], size[0])         
    return lst


# In[171]:

def load_images(path):
    images, answers = [], []
    size = 32, 32
    group = 0
    for item in category:
        for infile in glob.glob(path + item + "/*.png"):
            im = Image.open(infile)
            im = im.resize(size)
            images.append(get_pixels(im))
            answers.append(group)
        group += 1   
    images = np.array(images)
    return (images, answers)


# In[256]:

data = load_images(path)
X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size = 0.25, random_state = 42)


# In[257]:

nb_classes = len(category)       # Количество классов изображений                     
nb_epoch = 2                     # Количество эпох для обучения
img_rows, img_cols = 32, 32      # Размер изображений
batch_size = 32                  
# Размер мини-выборки


# In[259]:

for i in range(nb_classes):
    print(str(i) + ' - ' + str(y_test.count(i)) + ' - ' + str(y_train.count(i)))


# In[260]:

# Преобразуем метки в категории
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# In[262]:

# Создаем последовательную модель
model = Sequential()
# Первый сверточный слой
model.add(Convolution2D(32, 3, 3, border_mode = 'same', input_shape = (1, img_rows, img_cols), activation = 'relu'))
# Второй сверточный слой
model.add(Convolution2D(32, 1, 1, activation = 'relu'))
# Первый слой подвыборки
# model.add(MaxPooling2D(pool_size = (2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))


# In[263]:

# Третий сверточный слой
model.add(Convolution2D(64, 1,1, border_mode = 'same', activation = 'relu'))
# Четвертый сверточный слой
model.add(Convolution2D(64, 1,1, activation = 'relu'))
# Второй слой подвыборки
#model.add(MaxPooling2D(pool_size = (2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))
# Слой преобразования данных из 2D представления в плоское
model.add(Flatten())
# Полносвязный слой для классификации
model.add(Dense(512, activation = 'relu'))
# Слой регуляризации Dropout
model.add(Dropout(0.5))
# Выходной полносвязный слой
model.add(Dense(nb_classes))


# In[264]:

# Задаем параметры оптимизации
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
# Обучаем модель
X_train=X_train.reshape(990,1,32,32)
model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = nb_epoch, validation_split = 0.1, shuffle = True)
# Оцениваем качество обучения модели на тестовых данных
scores = model.evaluate(X_test.reshape(330,1,32,32), Y_test, verbose = 0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))


# In[ ]:



