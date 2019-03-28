import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD


class VGG16:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        model = Sequential()

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2), padding='valid', strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2), padding='valid', strides=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2), padding='valid', strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2), padding='valid', strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2), padding='valid', strides=(2, 2)))
        # output (7, 7, 512)
        model.add(Flatten())
        # 变成一维向量
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        # 以0.5的概率抛弃一些神经元
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))

        sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        # model.summary()
        self.model = model

    def predict(self):
        return self.model.predict(self.x_test)


'''
# 生成虚拟数据
# 随机生成训练数据
# np.random.random(size=None) return [0.0, 1.0)
x_train = np.random.random((100, 224, 224, 3))
# 随机生成训练数据分类标签，利用to_categorical转换成one-hot表示
# numpy.random.randint(low, high=None, size=None, dtype='l')
# If high is None (the default), then results are from [0, low).

# keras工具类，keras.utils.to_categorical(y, num_classes=None, dtype='float32')
# Converts a class vector (integers) to binary class matrix.(one-hot representation)
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

# 随机生成测试数据
x_test = np.random.random((20, 224, 224, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

# 建立顺序模型
model = Sequential()
# 输入 （224， 224， 3）张量
# 使用32个大小为3x3的卷积滤波器
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
# output (222,222,32)
model.add(Conv2D(32, (3, 3), activation='relu'))
# output (220,220,32)
# MaxPooling, stride 默认是 pool_size
model.add(MaxPooling2D(pool_size=(2, 2)))
# output (110, 110, 32)
# 随机去神经元
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
# output (108, 108, 64)
model.add(Conv2D(64, (3, 3), activation='relu'))
# output (106, 106, 64)
model.add(MaxPooling2D(pool_size=(2, 2)))
# output (53, 53, 64)
model.add(Dropout(0.25))
# 随机去神经元

# 平整化，即变成一维向量
model.add(Flatten())
# 全连接层
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
# 分类层，10类
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# 损失函数，优化器
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)

'''
