# -*- coding: utf-8 -*-
from absl import flags
from model_V6 import *
from random import random

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

F = easydict.EasyDict({"A_tr_txt_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_16_39_40_63/train/female_16_39_train.txt",
                       
                       "A_tr_img_path": "D:/[1]DB/[1]second_paper_DB/AFAD_16_69_DB/backup/fix_AFAD/",
                       
                       "B_tr_txt_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_16_39_40_63/train/male_40_63_train.txt",

                       "B_tr_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/male_40_63/",
                       
                       "in_start_age": 16,

                       "s_start_age": 40,

                       "img_size": 256,
                       
                       "load_size": 270,
                       
                       "batch_size": 1,
                       
                       "epochs": 200,

                       "num_classes": 24,
                       
                       "lr": 0.0002,
                       
                       "train": True,
                       
                       "save_checkpoint": "",
                       
                       "pre_checkpoint": False,
                       
                       "pre_checkpoint_path": "",
                       
                       "save_images": "C:/Users/Yuhwan/Pictures/sample_images",
                       
                       "graphs": ""})

g_optim = tf.keras.optimizers.Adam(F.lr)
d_optim = tf.keras.optimizers.Adam(F.lr)
s_optim = tf.keras.optimizers.Adam(F.lr)

def train_map(img_list, lab_list):

    A_imgs = tf.io.read_file(img_list)
    A_imgs = tf.image.decode_jpeg(A_imgs, 3)
    A_imgs = tf.image.resize(A_imgs, [F.load_size, F.load_size])
    A_imgs = tf.image.random_crop(A_imgs, [F.img_size, F.img_size, 3]) / 127.5 -1.

    if random() > 0.5:
        A_imgs = tf.image.flip_left_right(A_imgs)

    A_labs = tf.cast(lab_list - F.in_start_age + 1, tf.float32)

    return A_imgs, A_labs

def style_map(img_list, lab_list):

    S_imgs = tf.io.read_file(img_list)
    S_imgs = tf.image.decode_jpeg(S_imgs, 3)
    S_imgs = tf.image.resize(S_imgs, [F.load_size, F.load_size])
    S_imgs = tf.image.random_crop(S_imgs, [F.img_size, F.img_size, 3]) / 127.5 -1.

    if random() > 0.5:
        S_imgs = tf.image.flip_left_right(S_imgs)

    S_labs = tf.cast(lab_list - F.s_start_age + 1, tf.float32)

    return S_imgs, S_labs

@tf.function
def run_model(model, image, training=True):
    return model(image, training=training)

def cal_loss(A_images, A_labels,
             S_images, S_labels,
             generator_model,generator_model_,
             discriminator_model,style_model):

    with tf.GradientTape(persistent=True) as tape:
        A = tf.reduce_mean(S_labels / tf.convert_to_tensor(F.num_classes, dtype=tf.float32))
        S = tf.reduce_mean(A_labels / tf.convert_to_tensor(F.num_classes, dtype=tf.float32))

        fake_imgs = run_model(generator_model, [A_images, S_images, A], True)
        fake_imgs_ = run_model(generator_model, [fake_imgs, A_images, S], True)
        real_encode = run_model(style_model, [S_images, A_images, S], True)
        fake_encode = run_model(style_model, [fake_imgs, A_images, S], True)

        fake_d = run_model(discriminator_model, fake_imgs, True)
        real_d = run_model(discriminator_model, A_images, True)

        # g loss
        g_logits_loss = tf.reduce_mean((fake_d - tf.ones_like(fake_d))**2)

        # style_loss
        g_encode_style_loss = tf.reduce_mean(tf.abs(real_encode - fake_encode)) * 2.
        g_generator_style_loss = tf.reduce_mean(tf.abs(A_images - fake_imgs_))

        # adversarial loss
        d_adv_loss = ( tf.reduce_mean((real_d - tf.ones_like(real_d))**2) + tf.reduce_mean((fake_d - tf.zeros_like(fake_d))**2) ) * 0.5

        g_loss = g_logits_loss + g_encode_style_loss + g_generator_style_loss
        d_loss = d_adv_loss

    g_grads = tape.gradient(g_loss, generator_model.trainable_variables)
    s_grads = tape.gradient(g_loss, style_model.trainable_variables)
    d_grads = tape.gradient(d_loss, discriminator_model.trainable_variables)

    g_optim.apply_gradients(zip(g_grads, generator_model.trainable_variables))
    s_optim.apply_gradients(zip(s_grads, style_model.trainable_variables))
    d_optim.apply_gradients(zip(d_grads, discriminator_model.trainable_variables))

    return g_loss, d_loss

def main():
    
    generator_model = generator_networks(input_shape=(F.img_size, F.img_size, 3),
                               num_classes=F.num_classes)
    generator_model_ = generator_networks(input_shape=(F.img_size, F.img_size, 3),
                               num_classes=F.num_classes)
    discriminator_model = discriminator(input_shape=(F.img_size, F.img_size, 3))
    style_model = style_encoder(input_shape=(F.img_size, F.img_size, 3),
                               num_classes=F.num_classes)
    generator_model.summary()
    discriminator_model.summary()
    style_model.summary()

    if F.pre_checkpoint:
        ckpt = tf.train.Checkpoint(generator_model=generator_model,
                                   generator_model_=generator_model_,
                                   discriminator_model=discriminator_model,
                                   style_model=style_model,
                                   g_optim=g_optim,
                                   d_optim=d_optim,
                                   s_optim=s_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, F.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("=================================")
            print("* latest checkpoint resotred!!! *")
            print("=================================")

    if F.train:
        count = 0
        input_data = np.loadtxt(F.A_tr_txt_path, dtype="<U100", skiprows=0, usecols=0)
        input_data = [F.A_tr_img_path + img for img in input_data]
        input_label = np.loadtxt(F.A_tr_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        style_data = np.loadtxt(F.B_tr_txt_path, dtype="<U100", skiprows=0, usecols=0)
        style_data = [F.B_tr_img_path + img for img in style_data]
        style_label = np.loadtxt(F.B_tr_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        for epoch in range(F.epochs):

            A = list(zip(input_data, input_label))
            np.random.shuffle(A)
            input_data, input_label = zip(*A)
            input_data, input_label = np.array(input_data), np.array(input_label)
            B = list(zip(style_data, style_label))
            np.random.shuffle(B)
            style_data, style_label = np.array(style_data), np.array(style_label)

            input_gener = tf.data.Dataset.from_tensor_slices((input_data, input_label))
            input_gener = input_gener.shuffle(len(A))
            input_gener = input_gener.map(train_map)
            input_gener = input_gener.batch(F.batch_size)
            input_gener = input_gener.prefetch(tf.data.experimental.AUTOTUNE)

            style_gener = tf.data.Dataset.from_tensor_slices((style_data, style_label))
            style_gener = style_gener.shuffle(len(B))
            style_gener = style_gener.map(style_map)
            style_gener = style_gener.batch(F.batch_size)
            style_gener = style_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_idx = len(A) // F.batch_size
            tr_iter = iter(input_gener)
            s_iter = iter(style_gener)  # 원본 이미지와 style 이미지와 같은  장 수 로 가정
            for step in range(tr_idx):
                A_images, A_labels = next(tr_iter)
                S_images, S_labels = next(s_iter)
            
                g_loss, d_loss = cal_loss(A_images, A_labels,
                                          S_images, S_labels,
                                          generator_model=generator_model,
                                          generator_model_=generator_model_,
                                          discriminator_model=discriminator_model,
                                          style_model=style_model)

                print("Epochs: {} [{}/{}] g_loss = {}, d_loss = {}".format(epoch, step + 1, tr_idx, g_loss, d_loss))
                
                if count % 500 == 0:
                    A = tf.reduce_mean(tf.reduce_mean(S_labels / tf.convert_to_tensor(F.num_classes, dtype=tf.float32)))
                    fake_imgs = run_model(generator_model, [A_images, S_images, A], True)

                    plt.imsave(F.save_images + "/fake_1_{}.jpg".format(count), fake_imgs[0].numpy() * 0.5 + 0.5)
                    plt.imsave(F.save_images + "/input_1_{}.jpg".format(count), A_images[0].numpy() * 0.5 + 0.5)
                    plt.imsave(F.save_images + "/style_1_{}.jpg".format(count), S_images[0].numpy() * 0.5 + 0.5)

                count += 1


if __name__ == "__main__":
    main()