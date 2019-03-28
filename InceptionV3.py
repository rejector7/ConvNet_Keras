import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras import regularizers, optimizers


class InceptionV3:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # input (299, 299, 3)

    # combine Conv, BN, Activation into one func
    def conv_block(self, x, nb_filter, nb_row, nb_col):
        x = Conv2D(nb_filter, nb_row, nb_col)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    # four pathway
    def inception_module1(self, x, params):
        (branch1, branch2, branch3, branch4) = params
        # 1x1 #strides = 1 by default
        pathway1 = Conv2D(filters=branch1[0], kernel_size=(1, 1), strides=1, padding='same')(x)
        # 先归一化 再激活
        pathway1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway1))
        # 1x1 -> 3x3
        pathway2 = Conv2D(filters=branch2[0], kernel_size=(1, 1), padding='same')(x)
        pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
        pathway2 = Conv2D(filters=branch2[1], kernel_size=(3, 3), padding='same')(pathway2)
        pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
        # 1x1 -> 3x3 -> 3x3 (change from 5x5)
        pathway3 = Conv2D(filters=branch3[0], kernel_size=(1, 1), padding='same')(x)
        pathway3 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3))
        pathway3 = Conv2D(filters=branch3[1], kernel_size=(3, 3), padding='same')(pathway3)
        pathway3 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3))
        pathway3 = Conv2D(filters=branch3[1], kernel_size=(3, 3), padding='same')(pathway3)
        pathway3 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3))
        # average pool 3x3 -> 1x1
        pathway4 = AveragePooling2D((3, 3), padding='same')(x)
        pathway4 = Conv2D(filters=branch4[0], kernel_size=(1, 1), padding='same')(pathway4)
        pathway4 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway4))

        # concatenate
        return concatenate([pathway1, pathway2, pathway3, pathway4], axis=-1)

    # reduce grid size
    def inception_reduce1(self, x, params):
        (branch1, branch2) = params
        # 1x1 -> 3x3 strides=2
        # pathway1 = Conv2D(filters=branch1[0], kernel_size=(1, 1), padding='same')(x)
        # pathway1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway1))
        # 3x3 strides=2
        pathway1 = Conv2D(filters=branch1[0], kernel_size=(3, 3), strides=(2, 2))(x)
        pathway1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway1))

        # 1x1 -> 3x3 strides=1 -> 3x3 strides=2
        pathway2 = Conv2D(filters=branch2[0], kernel_size=(1, 1), padding='same')(x)
        pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
        pathway2 = Conv2D(filters=branch2[1], kernel_size=(3, 3), padding='same')(pathway2)
        pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
        pathway2 = Conv2D(filters=branch2[2], kernel_size=(3, 3), strides=(2, 2))(pathway2)
        pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))

        # max pool
        pathway3 = MaxPooling2D((3, 3), strides=(2, 2))(x)

        return concatenate([pathway1, pathway2, pathway3], axis=-1)

    def inception_module2(self, x, params):
        (branch1, branch2, branch3, branch4) = params
        # 1x1
        pathway1 = Conv2D(filters=branch1[0], kernel_size=(1, 1), strides=1, padding='same')(x)
        pathway1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway1))
        # 1x1->1x7->7x1
        pathway2 = Conv2D(filters=branch2[0], kernel_size=(1, 1), strides=1, padding='same')(x)
        pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
        pathway2 = self.conv_block(pathway2, branch2[1], 1, 7)
        pathway2 = self.conv_block(pathway2, branch2[2], 7, 1)
        # 1x1->7x1->1x7->7x1->1x7
        pathway3 = Conv2D(filters=branch3[0], kernel_size=(1, 1), strides=1, padding='same')(x)
        pathway3 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3))
        pathway3 = self.conv_block(pathway3, branch3[1], 7, 1)
        pathway3 = self.conv_block(pathway3, branch3[2], 1, 7)
        pathway3 = self.conv_block(pathway3, branch3[3], 7, 1)
        pathway3 = self.conv_block(pathway3, branch3[4], 1, 7)
        # pool 3x3 strides = 1 ->1x1
        pathway4 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
        pathway4 = Conv2D(filters=branch4[0], kernel_size=(1, 1), strides=1, padding='same')(pathway4)
        pathway4 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway4))
        return concatenate([pathway1, pathway2, pathway3, pathway4], axis=-1)

    def inception_reduce2(self, x, params):
        """
        pathway 1：1×1（1）→3×3（2）
        pathway 2：1×1（1）→1×7（1）→7×1（1）→3×3（2）
        pathway 3：max pooling 3×3（2）
        """
        (branch1, branch2) = params
        # 1x1->3x3
        pathway1 = Conv2D(filters=branch1[0], kernel_size=(1, 1), strides=1, padding='same')(x)
        pathway1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway1))
        pathway1 = Conv2D(filters=branch1[1], kernel_size=(3, 3), strides=2)(pathway1)
        pathway1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway1))
        # 1x1->1x7->7x1->3x3
        pathway2 = Conv2D(filters=branch2[0], kernel_size=(1, 1), strides=1, padding='same')(x)
        pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
        pathway2 = self.conv_block(pathway2, branch2[1], 1, 7)
        pathway2 = self.conv_block(pathway2, branch2[2], 7, 1)
        pathway2 = Conv2D(filters=branch2[3], kernel_size=(3, 3), strides=2)(pathway2)
        pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
        # 3x3->1x1
        pathway3 = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        return concatenate([pathway1, pathway2, pathway3], axis=-1)

    def inception_module3(self, x, params):
        (branch1, branch2, branch3, branch4) = params
        # 1x1
        pathway1 = Conv2D(filters=branch1[0], kernel_size=(1, 1), strides=1, padding='same')(x)
        pathway1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway1))
        # 1x1->1x3+3x1（并列）
        pathway2 = Conv2D(filters=branch2[0], kernel_size=(1, 1), strides=1, padding='same')(x)
        pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
        pathway2_1 = self.conv_block(pathway2, branch2[1], 1, 3)
        pathway2_2 = self.conv_block(pathway2, branch2[2], 3, 1)

        # 1x1->3x3->1x3+3x1
        pathway3 = Conv2D(filters=branch3[0], kernel_size=(1, 1), strides=1, padding='same')(x)
        pathway3 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3))
        pathway3 = Conv2D(filters=branch3[1], kernel_size=(3, 3), strides=1, padding='same')(pathway3)
        pathway3 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3))
        pathway3_1 = self.conv_block(pathway3, branch3[2], 1, 3)
        pathway3_2 = self.conv_block(pathway3, branch3[3], 3, 1)
        # 3x3->1x1
        pathway4 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
        pathway4 = Conv2D(filters=branch4[0], kernel_size=(1, 1), strides=1, padding='same')(pathway4)
        pathway4 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway4))
        return concatenate([pathway1, pathway2_1, pathway2_2, pathway3_1, pathway3_2, pathway4], axis=-1)

    # construct the InceptionV3 Net
    def cons_model(self, img_input, weight_decay, DATA_FORMAT, DROPOUT, num_classes):
        x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
                   kernel_initializer="he_normal")(img_input)
        x = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                   kernel_initializer="he_normal")(x)
        x = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_initializer="he_normal")(x)
        x = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        x = Conv2D(80, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        x = Conv2D(192, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', data_format=DATA_FORMAT)(x)
        # 3 x module1
        x = self.inception_module1(x, params=[(64,), (48, 64), (64, 96), (32,)])  # 3a 256
        x = self.inception_module1(x, params=[(64,), (48, 64), (64, 96), (64,)])  # 3b 288
        x = self.inception_module1(x, params=[(64,), (48, 64), (64, 96), (64,)])  # 3c 288
        x = self.inception_reduce1(x, params=[(384,), (64, 96)])  # 768
        # 5 x module2
        x = self.inception_module2(x, params=[(192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)])  # 4a 768
        x = self.inception_module2(x, params=[(192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)])  # 4b 768
        x = self.inception_module2(x, params=[(192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)])  # 4c 768
        x = self.inception_module2(x, params=[(192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)])  # 4d 768
        x = self.inception_module2(x, params=[(192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,)])  # 4e 768
        x = self.inception_reduce2(x, params=[(192, 320), (192, 192, 192, 192)])  # 1280
        # 2 x module3
        x = self.inception_module3(x, params=[(320,), (384, 384, 384), (448, 384, 384, 384), (192,)])  # 4e 2048
        x = self.inception_module3(x, params=[(320,), (384, 384, 384), (448, 384, 384, 384), (192,)])  # 4e 2048

        x = GlobalAveragePooling2D()(x)
        x = Dropout(DROPOUT)(x)
        x = Dense(num_classes, activation='softmax', kernel_initializer="he_normal",
                  kernel_regularizer=regularizers.l2(weight_decay))(x)
        return x

    def create_model(self):
        img_input = Input(shape=(32, 32, 3))
        output = self.cons_model(img_input)
        model = Model(img_input, output)
        model.summary()
