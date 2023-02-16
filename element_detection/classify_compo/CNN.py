import json
from keras.callbacks import History
import tensorflow as keras
from keras_applications.resnet50 import ResNet50
from keras.models import Model,load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from os.path import join as pjoin

from element_detection.classify_compo.Data import Data


class CNN:
    def __init__(self, data):
        self.data = data    # Data object
        self.model = None
        self.training_history = History()

        self.image_shape = data.image_shape
        self.class_map = data.class_map
        self.class_number = data.class_number

        self.model_path = r'model\resnet50_unfrozen_compo.h5'
        self.history_path = r'model\resnet50_unfrozen_compo_history.json'

    def build_model(self, frozen=False):
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=self.image_shape,
                              backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
        for layer in base_model.layers:
            layer.trainable = frozen
        self.model = Flatten()(base_model.output)
        self.model = Dense(128, activation='relu')(self.model)
        self.model = Dropout(0.5)(self.model)
        self.model = Dense(self.class_number, activation='softmax')(self.model)
        self.model = Model(inputs=base_model.input, outputs=self.model)

    def train(self, epoch_num=30, continue_with_loading=False, frozen=False):
        if continue_with_loading:
            self.load()
        else:
            self.build_model(frozen=frozen)
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        training_history = self.model.fit(self.data.X_train, self.data.Y_train, batch_size=64, epochs=epoch_num, verbose=1, validation_data=(self.data.X_test, self.data.Y_test))
        self.model.save(self.model_path)
        print("Trained model is saved to", self.model_path)
        self.save_training_history()
        # record training history
        if continue_with_loading and self.training_history is not None:
            self.training_history.history['loss'] += training_history.history['loss']
            self.training_history.history['accuracy'] += training_history.history['accuracy']
            self.training_history.history['val_loss'] += training_history.history['val_loss']
            self.training_history.history['val_accuracy'] += training_history.history['val_accuracy']
        else:
            self.training_history = training_history

    def load(self):
        self.model = load_model(self.model_path)
        print('*** Model Loaded From', self.model_path, '***')

    def preprocess_img(self, image):
        image = cv2.resize(image, self.image_shape[:2])
        x = (image / 255).astype('float32')
        return x

    def predict_img_files(self, img_files, show=False):
        '''
        Predict class for image files
        :param img_files: list of image file paths
        :param show: Boolean
        '''
        orgs = []
        for i in range(len(img_files)):
            img = cv2.imread(img_files[i])
            orgs.append(img)
        return self.predict_images(orgs, show)

    def predict_images(self, images, show=False):
        '''
        Predict class for cv2 images
        :param images: list of cv2 images
        :param show: Boolean
        '''
        images_proc = [self.preprocess_img(img) for img in images]
        predictions = self.model.predict(np.array(images_proc))
        labels = [self.class_map[np.argmax(pred)] for pred in predictions]
        if show:
            for i in range(len(images)):
                print(labels[i])
                cv2.imshow('img', images[i])
                key = cv2.waitKey()
                if key == ord('q'):
                    break
            cv2.destroyWindow('img')
        return labels

    def evaluate(self, data):
        x_test = data.X_test
        y_test = [np.argmax(y) for y in data.Y_test]
        y_pre = [np.argmax(y_pre) for y_pre in self.model.predict(x_test, verbose=1)]

        matrix = confusion_matrix(y_test, y_pre)
        print(matrix)
        TP, FP, FN = 0, 0, 0
        for i in range(len(matrix)):
            TP += matrix[i][i]
            FP += sum(matrix[i][:]) - matrix[i][i]
            FN += sum(matrix[:][i]) - matrix[i][i]
        precision = TP/(TP+FP)
        recall = TP / (TP+FN)
        print("Precision:%.3f, Recall:%.3f" % (precision, recall))

    def save_training_history(self):
        json.dump(self.training_history.history, open(self.history_path, 'w'), indent=4)
        print('Save training history to', self.history_path)

    def show_training_history(self):
        # summarize history for accuracy
        history = self.training_history
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    cls = CNN(Data(cls='compo'))
    cls.load()
    cls.predict_img_files(['data/a1.jpg', 'data/a2.jpg'], show=True)



