#!/usr/bin/env python

__author__ = ["Jason Gardner", "Tamar Dexheimer", "Conor Nolan", "Jacob Tanchak"]
__credits__ = ["Jason Gardner", "Tamar Dexheimer", "Conor Nolan", "Jacob Tanchak"]
__license__ = "GPL"
__version__ = "0.1.5"
__maintainer__ = ["Jason Gardner", "Tamar Dexheimer", "Conor Nolan", "Jacob Tanchak"]
__email__ = ["n01480000@unf.edu"]
__status__ = "Development"

'''
TODO: Make early cancellation optional for networks, as they might be stopping early when improvement is still possible. DONE
TODO: Output information about network training. DONE
TODO: Run input string against "best" versions of networks coming from the grid searches. DONE
TODO: Whatever Conor wanted to do with the raw CSV data.
TODO: Add model names to confusion matrices.
TODO: Color code.
'''

import numpy as np
import pandas as pd
import argparse
import logging
import os
import sys
import json
#import uuid
import pickle
import string
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Conv1D, MaxPooling1D
from tensorflow.keras.datasets import imdb as imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences as padder
from tensorflow.keras import callbacks
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.ensemble import HistGradientBoostingClassifier as hgbc
from sklearn.ensemble import RandomForestClassifier as rfcn
from sklearn.svm import SVC as svc
from sklearn.model_selection import GridSearchCV as gridsearch
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix as confusion
from matplotlib import pyplot as plt
import re
import warnings
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    
# Disable TensorFlow logging
tf.keras.utils.disable_interactive_logging()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

LOG_FORMAT_testinput = logging.Formatter("%(asctime)s — %(name)s — %(funcName)s:%(lineno)d — %(message)s")
METRICS = ["f1_score", 
           "accuracy_score", 
           "precision_score", 
           "recall_score", 
           "confusion_matrix"]
LC_METRICS = ["accuracy", "precision", "recall", "f1"]
MODEL_NAMES = ["Convolutional Neural Network", 
               "Convolutional Neural Network (LSTM)", 
               "Random Forest", 
               "Multinomial Naive Bayes", 
               "Histogram-based Gradient Boosting Classification Tree", 
               "Support Vector Machine"]
MODEL_CALLS = [rfcn(), mnb(), hgbc(), svc()]
DEFAULT_METRIC = "f1_score"
RMSPROP_CLIP = 10.0
AVERAGE = "weighted"
METRICS = "accuracy"
BATCH_SIZE = 32
PADDING_LENGTH = 256
CV = 10
SCORING = "f1_weighted"
ERROR_SCORE = 0.0
N_JOBS = -1
AVERAGE = "weighted"
RETURN_TRAIN_SCORE = True
BUFFER_LENGTH = 32

class Data:
    def __init__(self, test: bool = False) -> None:
        self.test = test
        self.x_train, self.y_train, self.x_test, self.y_test = self._load_data()
        if parser.data['test'] == True:
            logger.info("Test mode enabled. Using subset of data.")
            self.x_train = self.x_train[:100]
            self.y_train = self.y_train[:100]
            self.x_test = self.x_test[:100]
            self.y_test = self.y_test[:100]
        self.X = np.concatenate((self.x_train, self.x_test), axis = 0)
        self.Y = np.concatenate((self.y_train, self.y_test), axis = 0)
        
    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Loading data from keras.datasets.imdb.load_data()")
        (x_train, y_train), (x_test, y_test) = imdb.load_data()
        logger.info("Data loaded from keras.datasets.imdb.load_data()")
        
        logger.info(f"Padding data set sequences to: {PADDING_LENGTH}")
        x_train = padder(x_train, maxlen = PADDING_LENGTH, padding = 'post', truncating = 'post')
        x_test = padder(x_test, maxlen = PADDING_LENGTH, padding = 'post', truncating = 'post')
        logger.info(f"Data set sequences padded to: {PADDING_LENGTH}")
        
        return x_train, y_train, x_test, y_test

    def _process_data(self) -> None:
        '''
        This code needs work if we're going to manually process the data.
        Right now, using the built-in functions is the best way to go.
        
        I started to write it before I realized this one was already done, so I'm leaving it here for now,
        but it needs to be adjusted to work on the .csv file.
        
        TODO: Lowercase all words, remove punctuation, remove HTML tags, remove stopwords, etc.
        '''
        logger.info("Processing data")
        
        logger.info("Removing HTML tags from data")
        for items in [self.x_train, self.x_test]:
            for i in range(len(items)):
                for j in range(len(items[i])):
                    if re.match('<.*?>', items[i][j]):
                        items[i][j] = re.sub('<.*?>', '', items[i][j])
        logger.info("HTML tags removed from data")
        
        logger.info("Tokenizing the review data")
        word_index = imdb.get_word_index()
        inverted_word_index = dict(
            (i, word) for (word, i) in word_index.items()
        )
        for items in [self.x_train, self.x_test]:
            for i in range(len(items)):
                for j in range(len(items[i])):
                    items[i][j] = inverted_word_index.get(items[i][j] - 3, "?")
        logger.info("Review data tokenized")
        
        logger.info("Padding sequences")
        for items in [self.x_train, self.x_test]:
            items = self._pad_sequences(items)
        logger.info("Sequences padded")
        
        logger.info("Data processed")
        
    def tokenize(self, data: str) -> np.array:
        word_index = imdb.get_word_index()
        data = [data.split()]
        inverted_word_index = dict(
            (i, word) for (word, i) in word_index.items()
        )
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = inverted_word_index.get(data[i][j] - 3)
        
        return data
    
    def _pad_sequences(self, data: np.array) -> np.array:
        data = padder(data, maxlen = PADDING_LENGTH, padding = "post", truncating = "post")

        return data
    
################
###  MODELS  ###
################

class Model:
    def __init__(self) -> None:
        self.uuid = parser.data['uuid']
        self.model = None
        self.x_train = data.x_train
        self.x_test = data.x_test
        self.y_train = data.y_train
        self.y_test = data.y_test
        self.predictions = None
        self.model_score = None
        self.f1 = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.confusion = None
        self.rmse = None
        self.best_f1 = None
        self.best_accuracy = None
        self.best_precision = None
        self.best_recall = None
        self.best_confusion = None
        self.best_rmse = None
        
    def fit(self) -> None:
        self.model.fit(self.x_train, self.y_train)
    
    def predict(self, input = None) -> None:
        if input is None:
            input = self.x_test
        self.predictions =  self.model.predict(input)
    
    def score(self) -> None:
        self.model_score = self.model.evaluate(self.x_test, self.y_test)
    
class NN(Model):
    def __init__(self) -> None:
        super().__init__()
        self.optimizer = None
        self.loss_function = None
        self.vocab_size = len(imdb.get_word_index()) + 1
        self.history = None
        
    def fit(self) -> None:
        earlystopping = callbacks.EarlyStopping(monitor = "val_loss",
                                        mode = "min",
                                        patience = 5,
                                        restore_best_weights = True)
        if parser.data['cancelearly']:
            self.history = self.model.fit(self.x_train, 
                                          self.y_train, 
                                          batch_size = BATCH_SIZE, 
                                          epochs = parser.data['epochs'], 
                                          validation_data = (self.x_test, self.y_test),
                                          callbacks = [earlystopping])
        else:
            self.history = self.model.fit(self.x_train, 
                                          self.y_train, 
                                          batch_size = BATCH_SIZE, 
                                          epochs = parser.data['epochs'], 
                                          validation_data = (self.x_test, self.y_test))
        
    def optimize(self) -> None:
        self.optimizer = tf.keras.optimizers.RMSprop(clipvalue = RMSPROP_CLIP)
        # self.optimizer = tf.keras.optimizers.Adadelta(learning_rate = 0.1, ema_momentum = 0.95)
        # self.optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        self.loss_function = tf.keras.losses.Huber()
        # self.loss_function = tf.keras.losses.mean_squared_error()
        # self.loss_function = tf.keras.losses.binary_crossentropy()
        # self.loss_function = tf.keras.losses.categorical_crossentropy()
        self.model.compile(optimizer = self.optimizer, loss = self.loss_function, metrics = [METRICS])
        
    def save(self) -> None:
        logger.info(f"Saving model to './models/{self.uuid}/{self.name} - Model.weights.h5'")
        self.model.save_weights(f'./models/{self.uuid}/{self.name} - Model.weights.h5')
        logger.info(f"Saved model weights to './models/{self.uuid}/{self.name} - Model.weights.h5'")
    
    def save_metrics(self) -> None:
        history = pd.DataFrame(self.history.history)
        hist_csv_file = f"./models/{self.uuid}/{self.name} - Training History.csv"
        with open(hist_csv_file, mode='w') as f:
            history.to_csv(f)
            
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title(f'{self.name} Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        file_path = f"./models/{self.uuid}/{self.name} - Accuracy.png"
        plt.savefig(file_path)
        plt.close()
        
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title(f'{self.name} Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        file_path = f"./models/{self.uuid}/{self.name} - Loss.png"
        plt.savefig(file_path)
        plt.close()

        
    def load(self) -> None:
        self.init_model()
        if isinstance(self, NN):
            self.optimize()
        logger.info(f"Loading model from './models/{self.uuid}/{self.name} - Model.weights.h5'")
        self.model.build((None, PADDING_LENGTH))
        self.model.load_weights(f'./models/{self.uuid}/{self.name} - Model.weights.h5')
        logger.info(f"Loaded model weights from './models/{self.uuid}/{self.name} - Model.weights.h5'")
        
    def print(self) -> None:
        logger.info(f"{self.name}:  Accuracy - {self.model_score[1]}")
        logger.info(f"{self.name}:  Loss - {self.model_score[0]}")
        
class Ensemble(Model):
    def __init__(self) -> None:
        super().__init__()
        self.params = None
        self.best_params = None
        self.best_model = None
        self.model_best_score = None
        
    def search(self) -> dict:
        return gridsearch(self.model, 
                          self.params, 
                          cv = CV, 
                          scoring = SCORING, 
                          error_score = ERROR_SCORE,
                          n_jobs = N_JOBS,
                          return_train_score = RETURN_TRAIN_SCORE).fit(self.x_train, self.y_train)
        
    def grid_search(self) -> None:
        self.best_params = self.search().best_params_
        
    def best_predict(self) -> None:
        self.best_predictions = self.best_model.predict(self.x_test)
        
    def score(self) -> None:
        self.f1 = f1(self.y_test, self.predictions, average = AVERAGE, labels = np.unique(self.predictions))
        self.accuracy = accuracy(self.y_test, self.predictions)    
        self.precision = precision(self.y_test, self.predictions, average = AVERAGE, labels = np.unique(self.predictions))   
        self.recall = recall(self.y_test, self.predictions, average = AVERAGE, labels = np.unique(self.predictions))
        self.confusion = confusion(self.y_test, self.predictions)
        self.rmse = mse(self.y_test, self.predictions, squared = False)
        
    def best_score(self) -> None:
            self.best_f1 = f1(self.y_test, self.best_predictions, average = AVERAGE, labels = np.unique(self.best_predictions))
            self.best_accuracy = accuracy(self.y_test, self.best_predictions)
            self.best_precision = precision(self.y_test, self.best_predictions, average = AVERAGE, labels = np.unique(self.best_predictions))
            self.best_recall = recall(self.y_test, self.best_predictions, average = AVERAGE, labels = np.unique(self.best_predictions))
            self.best_confusion = confusion(self.y_test, self.best_predictions)
            self.best_rmse = mse(self.y_test, self.best_predictions, squared = False)
            
    def save_metrics(self, best: bool = False) -> None:
        if best:
            logger.info(f"Saving best metrics to './models/{self.uuid}/{self.name} - Metrics - Best.csv'")
            metrics = pd.DataFrame({"Best F1 Score": [self.best_f1],
                                    "BestAccuracy": [self.best_accuracy],
                                    "Best Precision": [self.best_precision],
                                    "Best Recall": [self.best_recall],
                                    "Best RMSE": [self.best_rmse]})
            metrics.to_csv(f"./models/{self.uuid}/{self.name} - Metrics - Best.csv")
            logger.info(f"Saved best metrics to './models/{self.uuid}/{self.name} - Metrics - Best.csv'")
        else:
            logger.info(f"Saving metrics to './models/{self.uuid}/{self.name} - Metrics.csv'")
            metrics = pd.DataFrame({"F1 Score": [self.f1],
                                    "Accuracy": [self.accuracy],
                                    "Precision": [self.precision],
                                    "Recall": [self.recall],
                                    "RMSE": [self.rmse]})
            metrics.to_csv(f"./models/{self.uuid}/{self.name} - Metrics.csv")
            logger.info(f"Saved metrics to './models/{self.uuid}/{self.name} - Metrics.csv'")
        self.confusion_matrix(best)
            
    def confusion_matrix(self, best: bool = False) -> None:
        if best:
            cm = self.best_confusion
        else:
            cm = self.confusion
        class_names = ['Negative', 'Positive']
        fig, ax = plt.subplots(figsize = (8, 6))
        cmap = plt.get_cmap('Blues')

        im = ax.imshow(cm, interpolation='nearest', cmap = cmap)

        cbar = ax.figure.colorbar(im, ax = ax)
        cbar.ax.set_ylabel('Counts', rotation = -90, va = "bottom")

        ax.set(xticks = np.arange(cm.shape[1]),
               yticks = np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels = class_names,
               ylabel = 'True label',
               xlabel = 'Predicted label')
        
        if best:
            ax.set(title = f'{self.name} Confusion Matrix - Best')
        else:
            ax.set(title = f'{self.name} Confusion Matrix')

        plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")

        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha = "center", va = "center",
                        color = "white" if cm[i, j] > thresh else "black")

        fig.tight_layout()

        if best:
            filename = f"./models/{self.uuid}/{self.name} - Confusion Matrix - Best.png"
        else:
            filename = f"./models/{self.uuid}/{self.name} - Confusion Matrix.png"

        plt.savefig(filename)
        plt.close(fig)
        logger.info(f"Saved best confusion matrix to '{filename}'")
        
    def save(self) -> None:
        logger.info(f"Saving model to './models/{self.uuid}/{self.name} - Model.pkl'")
        with open(f'./models/{self.uuid}/{self.name} - Model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Saved model to './models/{self.uuid}/{self.name} - Model.pkl'")
        logger.info(f"Saving best model to './models/{self.uuid}/{self.name} - Model - Best.pkl'")
        with open(f'./models/{self.uuid}/{self.name} - Model - Best.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        logger.info(f"Saved best model to './models/{self.uuid}/{self.name} - Model - Best.pkl'")
        logger.info(f"Saving best params to './models/{self.uuid}/{self.name} - Parameters - Best.json'")
        with open(f'./models/{self.uuid}/{self.name} - Parameters - Best.json', 'w') as f:
            json.dump(self.best_params, f)
        logger.info(f"Saved best params to './models/{self.uuid}/{self.name} - Parameters - Best.json'")
        
    def load(self) -> None:
        self.init_model()
        logger.info(f"Loading model from './models/{self.uuid}/{self.name} - Model.pkl'")
        with open(f'./models/{self.uuid}/{self.name} - Model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Loaded model from './models/{self.uuid}/{self.name} - Model.pkl'")
        logger.info(f"Loading best model from './models/{self.uuid}/{self.name} - Model - Best.pkl'")
        with open(f'./models/{self.uuid}/{self.name} - Model - Best.pkl', 'rb') as f:
            self.best_model = pickle.load(f)
        logger.info(f"Loaded best model from './models/{self.uuid}/{self.name} - Model - Best.pkl'")
        logger.info(f"Loading best params from './models/{self.uuid}/{self.name} - Parameters - Best.json'")
        with open(f'./models/{self.uuid}/{self.name} - Parameters - Best.json', 'r') as f:
            self.best_params = json.load(f)
        logger.info(f"Loaded best params from './models/{self.uuid}/{self.name} - Parameters - Best.json'")
    
    def print(self, best: bool = False) -> None:
        if best:
            logger.info(f"{self.name} F1 Score: {self.best_f1}")
            logger.info(f"{self.name} Accuracy: {self.best_accuracy}")
            logger.info(f"{self.name} Precision: {self.best_precision}")
            logger.info(f"{self.name} Recall: {self.best_recall}")
            logger.info(f"{self.name} RMSE: {self.best_rmse}")
        else:
            logger.info(f"{self.name} Best F1 Score: {self.f1}")
            logger.info(f"{self.name} Best Accuracy: {self.accuracy}")
            logger.info(f"{self.name} Best Precision: {self.precision}")
            logger.info(f"{self.name} Best Recall: {self.recall}")
            logger.info(f"{self.name} Best RMSE: {self.rmse}")
            
class CNN(NN):
    def __init__(self) -> None:
        super().__init__()
        self.name = MODEL_NAMES[0]
            
    def init_model(self) -> None:
        logger.info(f"Initializing {self.name} model")
        self.model = Sequential([Embedding(input_dim = self.vocab_size, output_dim = 128),
                                 Conv1D(filters = 32, kernel_size = 4, padding = 'same', activation = 'relu'),
                                 MaxPooling1D(pool_size = 2),
                                 Conv1D(filters = 64, kernel_size = 5, padding = 'same', activation = 'relu'),
                                 MaxPooling1D(pool_size = 4),
                                 Conv1D(filters = 128, kernel_size = 6, padding = 'same', activation = 'relu'),
                                 MaxPooling1D(pool_size = 8),
                                 Dense(32),
                                 Dense(1, activation = 'sigmoid')])
        logger.info(f"Model {self.name} initialized")
    
class LSTMCNN(NN):
    def __init__(self) -> None:
        super().__init__()
        self.name = MODEL_NAMES[1]
        
    def init_model(self) -> None:
        logger.info(f"Initializing {self.name} model")
        self.model = Sequential([Embedding(input_dim = self.vocab_size, output_dim = 128),
                                 Conv1D(filters = 32, kernel_size = 4, padding = 'same', activation = 'relu'),
                                 MaxPooling1D(pool_size = 2),
                                 Conv1D(filters = 64, kernel_size = 5, padding = 'same', activation = 'relu'),
                                 MaxPooling1D(pool_size = 4),
                                 Conv1D(filters = 128, kernel_size = 6, padding = 'same', activation = 'relu'),
                                 MaxPooling1D(pool_size = 8),
                                 LSTM(64, dropout = 0.2),
                                 Dense(32),
                                 Dense(1, activation = 'sigmoid')])
        logger.info(f"Model {self.name} initialized")
        
class SVC(Ensemble):
    def __init__(self) -> None:
        super().__init__()
        self.name = MODEL_NAMES[5]
        self.params = { 'C': [ 1.0, 0.5, 0.1 ],
                        'kernel': [ 'linear', 'poly', 'rbf', 'sigmoid' ],
                        'gamma': [ 'scale', 'auto' ]
                      }
        
    def init_model(self) -> None:
        self.model = svc(C = self.params['C'][0],
                         kernel = self.params['kernel'][0],
                         gamma = self.params['gamma'][0])
        
    def grid_search(self) -> None:
        self.best_params = self.search().best_params_
        
    def best_fit(self) -> None:
        self.best_model = svc(C = self.best_params['C'],
                             kernel = self.best_params['kernel'],
                             gamma = self.best_params['gamma'])
        self.best_model.fit(self.x_train, self.y_train)
    
class MNB(Ensemble):
    def __init__(self) -> None:
        super().__init__()
        self.name = MODEL_NAMES[3]
        self.params = { 'alpha': [ 1.0, 0.5, 0.1 ] }
        
    def init_model(self) -> None:
        self.model = mnb(alpha = self.params['alpha'][0])
        
    def best_fit(self) -> None:
        self.best_model = mnb(alpha = self.best_params['alpha'])
        self.best_model.fit(self.x_train, self.y_train)
        
class RFC(Ensemble):
    def __init__(self) -> None:
        super().__init__()
        self.name = MODEL_NAMES[2]
        self.params = { 'n_estimators': [ 100, 200, 300 ],
                        'max_depth': [ 10, 20, 30 ],
                        'criterion': [ 'gini', 'entropy' ]
                      }
        
    def init_model(self) -> None:
        self.model = rfcn(n_estimators = self.params['n_estimators'][0], 
                          max_depth = self.params['max_depth'][0], 
                          criterion = self.params['criterion'][0])
        
    def grid_search(self) -> None:
        self.best_params = self.search().best_params_
        
    def best_fit(self) -> None:
        self.best_model = rfcn(n_estimators = self.best_params['n_estimators'], 
                               max_depth = self.best_params['max_depth'], 
                               criterion = self.best_params['criterion'])
        self.best_model.fit(self.x_train, self.y_train)
        
class HGBC(Ensemble):
    def __init__(self) -> None:
        super().__init__()
        self.name = MODEL_NAMES[4]
        self.params = { 'max_iter': [ 100, 200, 300 ],
                        'max_depth': [ 10, 20, 30 ],
                        'learning_rate': [ 0.1, 0.01, 0.001 ]
                      }
        
    def init_model(self) -> None:
        self.model = hgbc(max_iter = self.params['max_iter'][0], 
                          max_depth = self.params['max_depth'][0], 
                          learning_rate = self.params['learning_rate'][0])
        
    def grid_search(self) -> None:
        self.best_params = self.search().best_params_
        
    def best_fit(self) -> None:
        self.best_model = hgbc(max_iter = self.best_params['max_iter'], 
                               max_depth = self.best_params['max_depth'], 
                               learning_rate = self.best_params['learning_rate'])
        self.best_model.fit(self.x_train, self.y_train)

###############
### LOGGING ###
###############
         
class Logging:
    def __init__(self, logger_name: str = '__main__') -> None:
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.get_console_handler())
        self.logger.propagate = False
        self.logger.info(f"Logging initialized -- {logger_name}")

    def get_console_handler(self) -> logging.StreamHandler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(LOG_FORMAT_testinput)

        return console_handler

    def get_logger(self) -> logging.Logger:

        return self.logger
   
#################
###  PARSER   ###
#################
   
class Arguments(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__(prog = "project", 
                        description = "Train various models for sentiment analysis on the IMDB dataset",
                        allow_abbrev = True)
        
        self.add_argument("-t",
                          "--test",
                          help = "run the program in test mode",
                          action = "store_true",
                          default = False)
        
        self.add_argument("-u", 
                            "--uuid", 
                            help = "the model UUID to load", 
                            type = str, 
                            default = None)
        
        self.add_argument("-s",
                            "--seed",
                            help = "the seed to use",
                            type = int,
                            default = 42)
        
        self.add_argument("-n",
                          "--neuralnetworks",
                          help = "remove neural networks",
                          action = "store_false",
                          default = True)
        
        self.add_argument("-e",
                          "--ensemble",
                          help = "remove ensemble models",
                          action = "store_false",
                          default = True)
        
        self.add_argument("-i",
                          "--testinput",
                          help = "Test string to categorize",
                          type = str,
                          default = None)
        
        self.add_argument("-c",
                          "--cancelearly",
                          help = "Enable early cancellation",
                          action = "store_true",
                          default = False)
        
        self.add_argument("-p",
                          "--testpercentage",
                          help = "Percentage of data to use for testing",
                          type = float,
                          default = 0.2)
        
        self.add_argument("-o",
                          "--epochs",
                          help = "Number of epochs to run",
                          type = int,
                          default = 25)
      
###############
###  AGENT  ###
###############
      
class Agent:
    def __init__(self) -> None:
        self.algorithms = self._process_flags()
        self.uuid = parser.data['uuid']
        np.random.seed(parser.data['seed'])
        self.testinput = parser.data['testinput']
        self.models: list[Model] = []
        
    def init_agent(self) -> None:
        if parser.data['uuid'] is None:
            logger.info(f"No UUID provided. Generating new UUID:")
            #self.uuid = uuid.uuid4().hex
            self.uuid = time.strftime("%Y-%m-%d-%H%M%S")
            parser.data['uuid'] = self.uuid
            logger.info(f"New UUID: {self.uuid}")
            self.save()
            logger.info(f"Logging to './models/{self.uuid}/log.txt'")
            self._add_log_to_file(f"./models/{self.uuid}/log.txt")
        else:
            logger.info(f"Loading data for UUID: {self.uuid}")
            self.load()
            for algorithm in self.algorithms:
                model = algorithm()
                logger.info(f"Loading {model.name} model")
                model.load()
                self.models.append(model)
        logger.info(f"Agent initialized with ID: {self.uuid}")

    def run(self) -> None:
        if self.models != []:
            self._run_models()
        else:
            self._create_models()
                
    def _create_models(self) -> None:
        for algorithm in self.algorithms:
            model = algorithm()
            model.init_model()
            if isinstance(model, NN):
                model.optimize()
            model.fit()
            model.predict()
            model.score()
            model.print()
            model.save_metrics()
            if isinstance(model, Ensemble):
                model.grid_search()
                model.best_fit()
                model.best_predict()
                model.best_score()
                model.print(best = True)
                model.save_metrics(best = True)
            self.models.append(model)
            model.save()
        self.learning_curves()
        if self.testinput is not None:
            self._run_models()
            
    def _run_models(self) -> None:
        if self.testinput is None:
            logger.warning("No test input provided. Exiting...")
            exit(1)
        logger.info(f"Running models on input text")
        
        logger.info(f"Processing input test input: '{self.testinput}'")
        self.testinput = self._remove_punctuation(self.testinput)
        self.testinput = self._tokenize(self.testinput)
        self.testinput = self._pad_sequences(self.testinput)
        self.testinput = np.squeeze(self.testinput)
        logger.info(f"Input processed")
        
        predictions = []
        for model in self.models:
            model_prediction = model.model.predict(np.array([self.testinput]))
            if isinstance(model, NN):
                predictions.append(model_prediction[0][0])
            else:
                predictions.append(model_prediction[0])
                
            logger.info(f'Prediction from {model.name}: {Fore.GREEN + "Positive" + Style.RESET_ALL if predictions[-1] > 0.5 else Fore.RED + "Negative" + Style.RESET_ALL}')

            
        for model in self.models:
            if isinstance(model, Ensemble) and model.best_model is not None:
                model_prediction = model.model.predict(np.array([self.testinput]))
                predictions.append(model_prediction[0])
                    
                logger.info(f'Prediction from Best {model.name}: {Fore.GREEN + "Positive" + Style.RESET_ALL if predictions[-1] > 0.5 else Fore.RED + "Negative" + Style.RESET_ALL}')

        binary_outcomes = [1 if pred > 0.5 else 0 for pred in predictions]

        from statistics import mode
        try:
            final_prediction = mode(binary_outcomes)
        except:
            final_prediction = 1 if sum(binary_outcomes) >= len(binary_outcomes) / 2 else 0

        logger.info(f'Final prediction after majority voting: {Fore.GREEN + "Positive" + Style.RESET_ALL if final_prediction == 1 else Fore.RED + "Negative" + Style.RESET_ALL}')

        return final_prediction
    
    ################################
    ### INPUT DATA MANIPULATION ###
    ###############################

    def _remove_punctuation(self, data: str) -> str:
        
        return data.translate(str.maketrans('', '', string.punctuation))

    def _tokenize(self, data: str) -> np.array:
        word_index = imdb.get_word_index()

        words = data.lower().split()

        indices = [word_index.get(word, 2) + 3 for word in words]

        return np.array(indices)
        
    def _pad_sequences(self, data) -> np.array:
        data = [data]
        data = padder(data, maxlen = PADDING_LENGTH, padding = "post", truncating = "post")

        return data
    
    def _process_flags(self) -> None:
        algorithms = [CNN, LSTMCNN, RFC, MNB, HGBC, SVC]
        if not parser.data['neuralnetworks']:
            algorithms.remove(CNN)
            algorithms.remove(LSTMCNN)
        if not parser.data['ensemble']:
            algorithms.remove(RFC)
            algorithms.remove(MNB)
            algorithms.remove(HGBC)
            algorithms.remove(SVC)
            
        return algorithms
    
    def learning_curves(self) -> None:
        common_params = {
            "X": data.X,
            "y": data.Y,
            "train_sizes": np.linspace(0.1, 1.0, 5),
            "cv": ShuffleSplit(n_splits = CV, test_size = parser.data['testpercentage'], random_state = parser.data['seed']),
            "n_jobs": -1,
            "line_kw": {"marker": "o"},
            "std_display_style": "fill_between",
        }

        for scorer in LC_METRICS:
            for estimator in MODEL_CALLS:
                fig, ax = plt.subplots(figsize = (8, 5))
                LearningCurveDisplay.from_estimator(
                    estimator,
                    **common_params,
                    score_name=scorer.title(),
                    scoring=scorer,
                    ax=ax
                )

                handles, label = ax.get_legend_handles_labels()
                ax.legend(handles[:2], ["Training Score", "Test Score"], loc = "lower right")
                ax.set_title(f"{scorer.title()} Learning Curve for {estimator.__class__.__name__}")
                
                file_path = f"./models/{self.uuid}/{estimator.__class__.__name__} - {scorer.title()} - Learning Curve.png"
                plt.savefig(file_path)
                plt.close(fig)

                logger.info(f"Saved learning curve to '{file_path}'")

            
    def save(self) -> None:
        logger.info(f"Saving data to './models/{self.uuid}/'")
        logger.info(f"Creating directory './models/{self.uuid}'")
        results_dir_path = f"./models/{self.uuid}"
        if not os.path.exists(results_dir_path):
            if not os.path.exists('./models'):
                try:
                    os.mkdir('./models')
                except OSError:
                    logger.warning(f"Creation of the directory {'./models'} failed")
                    exit(1)
            try:
                os.mkdir(results_dir_path)
            except OSError:
                logger.warning(f"Creation of the directory {results_dir_path} failed")
                exit(1)
            else:
                logger.info(f"Successfully created the directory {results_dir_path}")

        logger.info(f"Saving Numpy random state to ./models/{self.uuid}/numpy_random_state.pkl")
        with open(f"./models/{self.uuid}/numpy_random_state.pkl", 'wb') as f:
            pickle.dump(np.random.get_state(), f)
        logger.info(f"Saved Numpy random state to ./models/{self.uuid}/random_state.npy")

        logger.info(f"Saving state to './models/{self.uuid}/state.json'")
        with open(f'./models/{self.uuid}/state.json', 'w') as f:
            json.dump(parser.data, f)
        logger.info(f"Saved state to './models/{self.uuid}/state.json'")

    def load(self) -> None:
        logger.info(f"Loading Numpy random state from './models/{self.uuid}/numpy_random_state.pkl'")
        if not os.path.exists(f"./models/{self.uuid}/numpy_random_state.pkl"):
            logger.warning(f"File './models/{self.uuid}/numpy_random_state.pkl' does not exist. Exiting...")
            exit(1)
        else:
            with open(f"./models/{self.uuid}/numpy_random_state.pkl", 'rb') as f:
                random_state = pickle.load(f)
                np.random.set_state(random_state)
            logger.info(f"Loaded Numpy random state from './models/{self.uuid}/numpy_random_state.pkl'")

        if not os.path.exists(f'./models/{self.uuid}/state.json'):
            logger.warning(f"File './models/{self.uuid}/state.json' does not exist. Exiting...")
            exit(1)
        else:
            with open(f'./models/{self.uuid}/state.json', 'r') as f:
                logger.info(f"Loading state from './models/{self.uuid}/state.json'")
                parser.data = json.load(f)
                logger.info(f"Loaded state from './models/{self.uuid}/state.json'")  
                
    def _add_log_to_file(self, file_path: str) -> None:
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(LOG_FORMAT_testinput)
        logger.addHandler(file_handler)
                
##########################
###  ARGUMENT PARSING  ###
##########################

class Parsing:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args : dict = args
        self.data : dict = {}
        self.testinput = None
        for key, value in vars(args).items():
                self.data[key] = value

        if args.uuid is not None:
            if not os.path.exists(f'./models/{self.data["uuid"]}'):
                logger.warning(f"The Model at ./models/{self.data['uuid']} does not exist. Exiting...")
                exit(1)

###############
###  MAIN   ###
###############

if __name__ == '__main__':
    logger = Logging().get_logger()
    logger.info(f"TensorFlow version: {tf.__version__}")
    args = Arguments().parse_args()
    parser = Parsing(args)
    data = Data()
    agent = Agent()
    agent.init_agent()
    agent.run()
    exit(0)