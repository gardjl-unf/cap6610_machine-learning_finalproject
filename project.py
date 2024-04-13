#!/usr/bin/env python

__author__ = ["Jason Gardner", "Tamar Dexheimer", "Conor Nolan"]
__credits__ = ["Jason Gardner", "Tamar Dexheimer", "Conor Nolan"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = ["Jason Gardner", "Tamar Dexheimer", "Conor Nolan"]
__email__ = ["n01480000@unf.edu"]
__status__ = "Development"

import numpy as np
import pandas as pd
import argparse
import logging
import os
import sys
import json
import uuid
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Embedding, Dense, Conv1D, MaxPooling1D
from tensorflow.keras.datasets import imdb as imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences as padder
from tensorflow.keras import callbacks
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.ensemble import HistGradientBoostingClassifier as hgbc
from sklearn.ensemble import RandomForestClassifier as rfcn
from sklearn.model_selection import GridSearchCV as gridsearch
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.metrics import f1_score #, accuracy_score, precision_score, recall_score, confusion_matrix
import re
import sys, os, warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

LOG_FORMAT_STRING = logging.Formatter("%(asctime)s — %(name)s — %(funcName)s:%(lineno)d — %(message)s")
RMSPROP_CLIP = 10.0
AVERAGE = "weighted"
METRICS = "accuracy"
BATCH_SIZE = 32
EPOCHS = 25
PADDING_LENGTH = 256
CV = 2
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
            self.x_train = self.x_train[10:20]
            self.y_train = self.y_train[10:20]
            self.x_test = self.x_test[10:20]
            self.y_test = self.y_test[10:20]
        
    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Loading data from keras.datasets.imdb.load_data()")
        (x_train, y_train), (x_test, y_test) = imdb.load_data()
        logger.info("Data loaded from keras.datasets.imdb.load_data()")
        
        logger.info("Padding sequences")
        x_train = padder(x_train, maxlen = PADDING_LENGTH, padding = 'post', truncating = 'post')
        x_test = padder(x_test, maxlen = PADDING_LENGTH, padding = 'post', truncating = 'post')
        logger.info("Sequences padded")
        
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
    
    def _pad_sequences(self, data: pd.DataFrame) -> pd.DataFrame:
        data = tf.keras.preprocessing.sequence.pad_sequences(data, maxlen = PADDING_LENGTH, padding = "post", truncating = "post")

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
        
    def fit(self) -> None:
        self.model.fit(self.x_train, self.y_train)
    
    def predict(self) -> None:
        self.predictions =  self.model.predict(self.x_test)
    
    def score(self) -> None:
        self.model_score = self.model.evaluate(self.x_test, self.y_test)
    
class NN(Model):
    def __init__(self) -> None:
        super().__init__()
        self.optimizer = None
        self.loss_function = None
        self.vocab_size = len(imdb.get_word_index()) + 1
        
    def fit(self) -> None:
        earlystopping = callbacks.EarlyStopping(monitor = "val_loss",
                                        mode = "min",
                                        patience = 5,
                                        restore_best_weights = True)
        self.model.fit(self.x_train, 
                       self.y_train, 
                       batch_size = BATCH_SIZE, 
                       epochs = EPOCHS, 
                       validation_data = (self.x_test, self.y_test),
                       callbacks = [earlystopping])
        
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
        logger.info(f"Saving model to './models/{self.uuid}/{self.name}-model.weights.h5'")
        self.model.save_weights(f'./models/{self.uuid}/{self.name}-model.weights.h5')
        logger.info(f"Saved model weights to './models/{self.uuid}/{self.name}-model.weights.h5'")
        
    def load(self) -> None:
        self.model = tf.keras.Model()
        logger.info(f"Loading model from './models/{self.uuid}/{self.name}-model.weights.h5'")
        self.model.load_weights(f'./models/{self.uuid}/{self.name}-model.weights.h5')
        logger.info(f"Loaded model weights from './models/{self.uuid}/{self.name}-model.weights.h5'")
        
    def print(self) -> None:
        logger.info(f"{self.name}:  Loss - {self.model_score[0]}, Test Data Accuracy - {self.model_score[1]}")
        
class Ensemble(Model):
    def __init__(self) -> None:
        super().__init__()
        self.params = None
        self.best_params = None
        self.best_model = None
        self.model_best_score = None
        
    def score(self) -> None:
        self.model_score = f1_score(self.y_test, self.predictions, average = AVERAGE, labels = np.unique(self.predictions))
        
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
        
    def best_score(self) -> None:
        self.model_best_score = f1_score(self.y_test, self.best_predictions, average= AVERAGE, labels = np.unique(self.best_predictions))
        
    def print(self, best : bool = False) -> None:
        if best:
            print(f"Best {self.name} F1 Score: {self.model_best_score}")
            print(f"Best {self.name} Parameters: {self.best_params}")
        else:
            print(f"{self.name} F1 Score: {self.model_score}")
            print(f"{self.name} Parameters: {self.params}")
        
    def save(self) -> None:
        logger.info(f"Saving model to './models/{self.uuid}/{self.name}-model.pkl'")
        with open(f'./models/{self.uuid}/{self.name}-model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Saved model to './models/{self.uuid}/{self.name}-model.pkl'")
        logger.info(f"Saving best model to './models/{self.uuid}/{self.name}-best-model.pkl'")
        with open(f'./models/{self.uuid}/{self.name}-best-model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        logger.info(f"Saved best model to './models/{self.uuid}/{self.name}-best-model.pkl'")
        logger.info(f"Saving best params to './models/{self.uuid}/{self.name}-best-params.json'")
        with open(f'./models/{self.uuid}/{self.name}-best-params.json', 'w') as f:
            json.dump(self.best_params, f)
        logger.info(f"Saved best params to './models/{self.uuid}/{self.name}-best-params.json'")
        
    def load(self) -> None:
        logger.info(f"Loading model from './models/{self.uuid}/{self.name}-model.pkl'")
        with open(f'./models/{self.uuid}/{self.name}-model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Loaded model from './models/{self.uuid}/{self.name}-model.pkl'")
        logger.info(f"Loading best model from './models/{self.uuid}/{self.name}-best-model.pkl'")
        with open(f'./models/{self.uuid}/{self.name}-best-model.pkl', 'rb') as f:
            self.best_model = pickle.load(f)
        logger.info(f"Loaded best model from './models/{self.uuid}/{self.name}-best-model.pkl'")
        logger.info(f"Loading best params from './models/{self.uuid}/{self.name}-best-params.json'")
        with open(f'./models/{self.uuid}/{self.name}-best-params.json', 'r') as f:
            self.best_params = json.load(f)
        logger.info(f"Loaded best params from './models/{self.uuid}/{self.name}-best-params.json'")
    
class RNN(NN):
    def __init__(self) -> None:
        super().__init__()
        self.name = "RNN"
            
    def init_model(self) -> tf.keras.Model:
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
        self.name = "LSTM"
        
    def init_model(self) -> tf.keras.Model:
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
    
class MNB(Ensemble):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Multinomial Naive Bayes"
        self.params = { 'alpha': [ 1.0, 0.5, 0.1 ] }
        
    def init_model(self) -> None:
        self.model = mnb(alpha = self.params['alpha'][0])
        
    def best_fit(self) -> None:
        self.best_model = mnb(alpha = self.best_params['alpha'])
        self.best_model.fit(self.x_train, self.y_train)
        
class RFC(Ensemble):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Random Forest Classifier"
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
        self.name = "Histogram-based Gradient Boosting Classification Tree"
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
        console_handler.setFormatter(LOG_FORMAT_STRING)

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
      
###############
###  AGENT  ###
###############
      
class Agent:
    def __init__(self) -> None:
        self.algorithms = self._process_flags()
        self.uuid = parser.data['uuid']
        np.random.seed(parser.data['seed'])

    def run(self) -> None:
        self.save()
        for algorithm in self.algorithms:
            model = algorithm()
            model.init_model()
            if isinstance(model, NN):
                model.optimize()
            model.fit()
            model.predict()
            model.score()
            model.print()
            if isinstance(model, Ensemble):
                model.grid_search()
                model.best_fit()
                model.best_predict()
                model.best_score()
                model.print(best = True)
            model.save()
            
    def _process_flags(self) -> None:
        algorithms = [RNN, LSTMCNN, RFC, MNB, HGBC]
        if not parser.data['neuralnetworks']:
            algorithms.remove(RNN)
            algorithms.remove(LSTMCNN)
        if not parser.data['ensemble']:
            algorithms.remove(RFC)
            algorithms.remove(MNB)
            algorithms.remove(HGBC)
            
        return algorithms
            
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
                
##########################
###  ARGUMENT PARSING  ###
##########################

class Parsing:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args : dict = args
        self.data : dict = {}
        for key, value in vars(args).items():
                self.data[key] = value

        if args.uuid is not None:
            if not os.path.exists(f'./models/{self.data["uuid"]}'):
                logger.info(f"The Model at ./models/{self.data['uuid']} does not exist. Exiting...")
                exit(1)
                
                
        # Temporary UUID generation until I add in the code to load if it is provided, or generate if it is not.
        else:
            self.data['uuid'] = uuid.uuid4().hex
            logger.info(f"Model UUID not provided. Generating new UUID: {self.data['uuid']}")

###############
###  MAIN   ###
###############

if __name__ == '__main__':
    logger = Logging().get_logger()
    args = Arguments().parse_args()
    parser = Parsing(args)
    data = Data()
    agent = Agent()
    agent.run()
    exit(0)