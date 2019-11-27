#encoding=utf-8
'''
So that we have 160 training examples for each class, and 80 validation examples for each class.
'''
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
import theano.sandbox.cuda

theano.sandbox.cuda.use('gpu0')

# path to the model weights file.
weights_path = '/mnt/hd2/home/xiaoxu/data/vgg16_weights.h5'

# dimensions of our images.
img_width, img_height = 256, 256
count = [800,720,640,560,480,400,320,240,160]
for i in range(14,15,1):
    path = "/mnt/hd2/home/xiaoxu/event_img2/UIUC_" + str(i) + "/"
    train =  path + "train" +str(i)
    validation =path + "validation" + str(i)
    test = path + "test" + str(i)

    train_count = count[i-6]



    # validation_data_dir = '/mnt/hd2/home/xiaoxu/LabelMe/validation10_extend'
    # test_data_dir       = '/mnt/hd2/home/xiaoxu/LabelMe/test'


    def save_bottlebeck_features():
        datagen = ImageDataGenerator(rescale=1./255)

        # build the VGG16 network
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # load the weights of the VGG16 networks
        # (trained on ImageNet, won the ILSVRC competition in 2014)
        # note: when there is a complete match between your model definition
        # and your weight savefile, you can simply call model.load_weights(filename)
        assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        print('Model loaded.')

        #生成train数据
        generator = datagen.flow_from_directory(
                train,
                target_size=(img_width, img_height),
                batch_size=32,
                class_mode= "categorical",
                shuffle=False)
        bottleneck_features_train = model.predict_generator(generator,train_count)
        np.save(open(path +'bottleneck_features_train_LM.npy', 'wb'), bottleneck_features_train)


        #生成确认数据validation
        generator = datagen.flow_from_directory(
                validation,
                target_size=(img_width, img_height),
                batch_size=32,
                class_mode= "categorical",
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(generator,80)
        np.save(open(path + 'bottleneck_features_validation_LM.npy', 'wb'), bottleneck_features_validation)


        generator = datagen.flow_from_directory(
                test,
                target_size=(img_width, img_height),
                batch_size=32,
                class_mode= "categorical",
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(generator,800)
        np.save(open(path + 'bottleneck_features_test_LM.npy', 'wb'), bottleneck_features_validation)

    save_bottlebeck_features()

print("DONE_All_ReshapeData")