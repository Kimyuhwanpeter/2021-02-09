# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# StarGAN V2 (https://arxiv.org/pdf/1912.01865.pdf)
l2 = tf.keras.regularizers.l2

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def adaIN(inputs, style, epsilon=1e-5):

    in_mean, in_var = tf.nn.moments(inputs, axes=[1,2], keepdims=True)
    st_mean, st_var = tf.nn.moments(style, axes=[1,2], keepdims=True)
    in_std, st_std = tf.sqrt(in_var + epsilon), tf.sqrt(st_var + epsilon)

    return st_std * (style - in_mean) / in_std + st_mean

class Conitional_IN(tf.keras.layers.Layer):
    def __init__(self, alpha = 1., num_class = 2):
        super(Conitional_IN, self).__init__()
        self.alpha = alpha
        self.num_class = num_class

    def build(self, input):

        self.y1 = self.add_weight(
            name = "y1",
            shape = [1, self.num_class],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True)

        self.y2 = self.add_weight(
            name = "y2",
            shape = [1, self.num_class],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True)

        self.beta = self.add_weight(
            name = "beta",
            shape=[self.num_class, input[0][-1]],
            initializer=tf.constant_initializer([0.]), 
            trainable=True)
        self.gamma = self.add_weight(
            name = "gamma",
            shape=[self.num_class, input[0][-1]],
            initializer=tf.constant_initializer([1.]),
            trainable=True)

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs[0], [1,2], keepdims=True)
        
        beta1 = tf.matmul(self.y1, self.beta)
        gamma1 = tf.matmul(self.y1, self.gamma)
        beta2 = tf.matmul(self.y2, self.beta)
        gamma2 = tf.matmul(self.y2, self.gamma)
        beta = inputs[1] * beta1 + (1. - inputs[1]) * beta2
        gamma = inputs[1] * gamma1 + (1. - inputs[1]) * gamma2

        return tf.nn.batch_normalization(inputs[0], mean, var, beta, gamma, 1e-10)


def generator_networks(input_shape=(256, 256, 3),
                       style_shape=(256, 256, 3),
                       weight_decay=0.000005,
                       num_classes=24,
                       age_factor=(1)):

    h = inputs = tf.keras.Input(input_shape)
    s = style_inputs = tf.keras.Input(style_shape)
    age = age_scale = tf.keras.Input(age_factor)

    ##########################################################################################
    # Conditional Instance normalization을 중간 layer에 쓰자 --> 이건 Adaptive instance normalization과 비슷하게 스타일을 따로따로 만들어 낼때 사용하였다. 
    # 공간정보를 유지하면서 donwsampling을 하는것이 제일 중요하다.
    s1 = tf.keras.layers.ZeroPadding2D((1,1))(s)
    s1 = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(s1)
    s1 = Conitional_IN(num_class=num_classes)([s1, age])
    s1 = tf.keras.layers.ReLU()(s1) # [256, 256, 64]

    s2 = tf.keras.layers.ZeroPadding2D((1,1))(s1)
    s2  = tf.keras.layers.Conv2D(filters=128,
                                 kernel_size=3,
                                 strides=2,
                                 padding="valid",
                                 use_bias=False,
                                 kernel_regularizer=l2(weight_decay))(s2)
    s2 = Conitional_IN(num_class=num_classes)([s2, age])
    s2 = tf.keras.layers.ReLU()(s2)  # [128, 128, 128]

    s3 = tf.keras.layers.ZeroPadding2D((1,1))(s2)
    s3 = tf.keras.layers.Conv2D(filters=256,
                                kernel_size=3,
                                strides=2,
                                padding="valid",
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay))(s3)
    s3 = Conitional_IN(num_class=num_classes)([s3, age])
    s3 = tf.keras.layers.ReLU()(s3) # [64, 64, 256]

    ##########################################################################################

    def residual_block(inputs, style):

        h = tf.keras.layers.ZeroPadding2D((1,1))(inputs)
        h = tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=3,
                                   strides=1,
                                   padding="valid",
                                   use_bias=False,
                                   kernel_regularizer=l2(weight_decay),
                                   groups=32)(h)
        h = adaIN(h, style)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.ZeroPadding2D((1,1))(h)
        h = tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=3,
                                   strides=1,
                                   padding="valid",
                                   use_bias=False,
                                   kernel_regularizer=l2(weight_decay),
                                   groups=32)(h)
        h = adaIN(h, style)

        return tf.keras.layers.ReLU()(h + inputs)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = adaIN(h, s1)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 64]

    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=1,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = adaIN(h, s1)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.ZeroPadding2D((1,1))(h)
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h_1 = h
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.ZeroPadding2D((1,1))(h)
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=2,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h_2 = h
    h = adaIN(h, s2)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 128]

    h = tf.keras.layers.ZeroPadding2D((1,1))(h)
    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=2,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h_3 = h
    h = adaIN(h, s3)
    h = tf.keras.layers.ReLU()(h)   # [64, 64, 256]

    for _ in range(8):
        h = residual_block(h, s3)   # [64, 64, 256]

    h = tf.keras.layers.ZeroPadding2D((1,1))(h) # 그런데 만일... decode 부분에 bottleneck 구조가 들어가려면?? --> 기존 bottleneck 구조와는 다르게 접근해야한다.
    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = adaIN(h + h_3, s3)
    h = tf.keras.layers.ReLU()(h)   # [64, 64, 256]

    h = tf.keras.layers.Conv2DTranspose(filters=128,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=l2(weight_decay))(h)
    h = adaIN(h + h_2, s2)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 128]

    h = tf.keras.layers.Conv2DTranspose(filters=64,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=l2(weight_decay))(h)
    h = adaIN(h + h_1, s1)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 64]

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=3,
                               kernel_size=7,
                               strides=1,
                               padding="valid")(h)
    h = tf.nn.tanh(h)   # [256, 256, 3]

    return tf.keras.Model(inputs=[inputs, style_inputs, age_scale], outputs=h)

def style_encoder(input_shape=(256, 256, 3),
                  style_shape=(256, 256, 3),
                  weight_decay=0.000005,
                  num_classes=24,
                  age_factor=(1)):

    h = inputs = tf.keras.Input(input_shape)
    s = style_inputs = tf.keras.Input(input_shape)
    age = age_scale = tf.keras.Input(age_factor)

    repeat_num = 6

    ##########################################################################################

    s1 = tf.keras.layers.ZeroPadding2D((1,1))(s)
    s1 = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(s1)
    s1 = Conitional_IN(num_class=num_classes)([s1, age])
    s1 = tf.keras.layers.ReLU()(s1) # [256, 256, 64]

    s2 = tf.keras.layers.ZeroPadding2D((1,1))(s1)
    s2  = tf.keras.layers.Conv2D(filters=128,
                                 kernel_size=3,
                                 strides=2,
                                 padding="valid",
                                 use_bias=False,
                                 kernel_regularizer=l2(weight_decay))(s2)
    s2 = Conitional_IN(num_class=num_classes)([s2, age])
    s2 = tf.keras.layers.ReLU()(s2)  # [128, 128, 128]

    s3 = tf.keras.layers.ZeroPadding2D((1,1))(s2)
    s3 = tf.keras.layers.Conv2D(filters=256,
                                kernel_size=3,
                                strides=2,
                                padding="valid",
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay))(s3)
    s3 = Conitional_IN(num_class=num_classes)([s3, age])
    s3 = tf.keras.layers.ReLU()(s3) # [64, 64, 256]

    ##########################################################################################

    def skip_connenction_with_residual(x, style):
        ######################################################################
        h1 = tf.keras.layers.Conv2D(filters=256,
                                    kernel_size=1,
                                    strides=1,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(x)
        h1 = adaIN(h1, style)
        h1 = tf.keras.layers.LeakyReLU(0.2)(h1)
        ######################################################################

        h = tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=4,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=l2(weight_decay),
                                   groups=32)(x)
        h = adaIN(h, style)
        h = tf.keras.layers.LeakyReLU(0.2)(h)

        h = tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=4,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=l2(weight_decay),
                                   groups=32)(x)
        h = adaIN(h, style)
        h = tf.keras.layers.LeakyReLU(0.2)(h)

        return h + h1

    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=4,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = adaIN(h, s1)
    h = tf.keras.layers.LeakyReLU(0.2)(h) # [256, 256, 64]

    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=4,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = adaIN(h, s2)
    h = tf.keras.layers.LeakyReLU(0.2)(h) # [128, 128, 128]

    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=4,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = adaIN(h, s3)
    h = tf.keras.layers.LeakyReLU(0.2)(h) # [64, 64, 256]

    for _ in range(repeat_num-2):
        h = skip_connenction_with_residual(h, s3)
    
    h = tf.keras.layers.AveragePooling2D((4,4), strides=2, padding="same")(h)   # [32, 32, 256]
    s4 = tf.keras.layers.AveragePooling2D((4,4), strides=2, padding="same")(s3) # [32, 32, 256]
    
    for _ in range(repeat_num-4):
        h = skip_connenction_with_residual(h, s4)

    s5 = tf.keras.layers.AveragePooling2D((4,4), strides=2, padding="same")(s4) # [16, 16, 256]
    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=4,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = adaIN(h, s5)
    h = tf.keras.layers.LeakyReLU(0.2)(h)   # [16, 16, 256]

    s6 = tf.keras.layers.AveragePooling2D((4,4), strides=2, padding="same")(s5) # [8, 8, 256]
    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=4,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = adaIN(h, s6)
    h = tf.keras.layers.LeakyReLU(0.2)(h)   # [8, 8, 256]

    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=[inputs, style_inputs, age_scale], outputs=h)


def discriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):

    dim_ = dim
    #Norm = BatchNorm(axis=3,momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON)

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)


    return tf.keras.Model(inputs=inputs, outputs=h)