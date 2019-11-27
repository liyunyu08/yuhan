import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import Model
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop,SGD
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import theano

theano.optimizers='None'

#def get_session():  
 #   sess_config = tf.ConfigProto()
 #   sess_config.gpu_options.allow_growth = True
  #  sess_config.allow_soft_placement=True
  #  return tf.Session(config=sess_config)
#KTF.set_session(get_session())



batch_size = 4
epochs = 200
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
            './train', 
            target_size = (224, 224),
            batch_size = batch_size)
image_numbers = train_generator.samples



test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
               './test', 
                target_size = (224, 224), 
                batch_size = batch_size)

#base_model = ResNet50(weights = 'imagenet')
base_model = VGG16(weights = 'imagenet')
predictions = Dense(196, activation='softmax')(base_model.output)
model = Model(inputs=base_model.input, outputs=predictions)

#rmsprop=RMSprop(lr=0.001)
#model.compile(optimizer=rmsprop, loss='categorical_crossentropy')
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'],
              callback=[]
              )

model.fit_generator(train_generator,
                steps_per_epoch = image_numbers,
                epochs = epochs)

test_acc = model.predict_generator(test_generator)

model.save_weights('weights.h5')