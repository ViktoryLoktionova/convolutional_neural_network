import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

x_train = np.expand_dims(x_train, axis=3) #добавляем еще одну размерность для количества каналов
x_test = np.expand_dims(x_test, axis=3) #добавляем еще одну размерность для количества каналов

print( x_train.shape )

model = keras.Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)), #первый сверточный слой
    MaxPooling2D((2, 2), strides=2), #карта признаков уменьшилась в два раза
    Conv2D(64, (3,3), padding='same', activation='relu'), #второй сверточный слой, исп предыдущую карту признаков
    MaxPooling2D((2, 2), strides=2),#карта признаков стала 7*7
    Flatten(), #превращает тензор 7*7*64 в единый вектор, который подает дальше на полносвязную НС
    Dense(128, activation='relu'), #слой из 128 нейронов
    Dense(10,  activation='softmax') #выходной слой НС
])

# print(model.summary())      # вывод структуры НС в консоль
#компиллируем НС
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


his = model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2) #обучаем НС

model.evaluate(x_test, y_test_cat) #пропускаем через НС тестовую выборку