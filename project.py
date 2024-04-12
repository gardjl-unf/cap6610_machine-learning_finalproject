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
import tensorflow as tf
import argparse
import logging
import os
import sys
import json
import uuid
import pickle
from tensorflow.keras.layers import LSTM, SimpleRNN, Embedding, Dense, Conv1D
from tensorflow.keras.datasets import imdb as imdb
from tensorflow.keras.models import Sequential
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import RandomForestClassifier as rfcn
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.preprocessing import StandardScaler as scaler
#https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
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
EPOCHS = 10
INTERNAL_DIMENSION = 128

class Data:
    def __init__(self, test: bool = False) -> None:
        self.test = test
        self.x_train, self.y_train, self.x_test, self.y_test = self._load_data()
        self.x_train = self.x_train[10:20]
        self.y_train = self.y_train[10:20]
        self.x_test = self.x_test[10:20]
        self.y_test = self.y_test[10:20]
        
    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        https://keras.io/api/datasets/imdb/
        
        keras.datasets.imdb.load_data(
            path="imdb.npz",
            num_words=None,
            skip_top=0,
            maxlen=None,
            seed=113,
            start_char=1,
            oov_char=2,
            index_from=3,
            **kwargs
        )
        
        Arguments

            path: where to cache the data (relative to ~/.keras/dataset).
            num_words: integer or None. Words are ranked by how often they occur (in the training set) and only the num_words most frequent words are kept. Any less frequent word will appear as oov_char value in the sequence data. If None, all words are kept. Defaults to None.
            skip_top: skip the top N most frequently occurring words (which may not be informative). These words will appear as oov_char value in the dataset. When 0, no words are skipped. Defaults to 0.
            maxlen: int or None. Maximum sequence length. Any longer sequence will be truncated. None, means no truncation. Defaults to None.
            seed: int. Seed for reproducible data shuffling.
            start_char: int. The start of a sequence will be marked with this character. 0 is usually the padding character. Defaults to 1.
            oov_char: int. The out-of-vocabulary character. Words that were cut out because of the num_words or skip_top limits will be replaced with this character.
            index_from: int. Index actual words with this index and higher.

        Returns

            Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).


        '''
        
        logger.info("Loading data from keras.datasets.imdb.load_data()")
        (x_train, y_train), (x_test, y_test) = imdb.load_data()
        logger.info("Data loaded from keras.datasets.imdb.load_data()")
        
        return pd.DataFrame(x_train), pd.DataFrame(y_train), pd.DataFrame(x_test), pd.DataFrame(y_test)

    def _process_data(self) -> None:
        '''
        This code needs work if we're going to manually process the data.
        Right now, using the built-in functions is the best way to go.
        
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
        data = tf.keras.preprocessing.sequence.pad_sequences(data, padding = "post", maxlen = 256)

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
    
class NN(Model):
    def __init__(self) -> None:
        super().__init__()
        self.input_width = max(self.x_train[0])
        self.optimizer = None
        self.loss_function = None
        
    def optimize(self) -> None:
        self.optimizer = tf.keras.optimizers.RMSprop(clipvalue = RMSPROP_CLIP)
        # self.optimizer = tf.keras.optimizers.Adadelta(learning_rate = 0.1, ema_momentum = 0.95)
        # self.optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        self.loss_function = tf.keras.losses.Huber()
        # self.loss_function = tf.keras.losses.mean_squared_error
        # self.loss_function = tf.keras.losses.binary_crossentropy
        self.model.compile(optimizer = self.optimizer, loss = self.loss_function, metrics = [METRICS])
        
    def fit(self) -> None:
        self.model.fit(self.x_train, self.y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 1, validation_data = (self.x_test, self.y_test))
    
    def predict(self) -> None:
        self.predictions =  self.model.predict(self.x_test)
    
    def score(self) -> None:
        self.model_score = self.model.evaluate(self.x_test, self.y_test)
        
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
        
    def scale(self) -> None:
        self.train_x = scaler().fit_transform(self.train_x)
        self.test_x = scaler().fit_transform(self.test_x)
        
    def save_model(self) -> None:
        logger.info(f"Saving model to './models/{self.uuid}/{self.model_name}-model.pkl'")
        pickle.dump(self.model, f'./models/{self.uuid}/{self.model_name}-model.pkl')
        logger.info(f"Saved model to './models/{self.uuid}/{self.model_name}-model.pkl'")
        logger.info(f"Saving best model to './models/{self.uuid}/{self.model_name}-best-model.pkl'")
        pickle.dump(self.best_model, f'./models/{self.uuid}/{self.model_name}-best-model.pkl')
        logger.info(f"Saved best model to './models/{self.uuid}/{self.model_name}-best-model.pkl'")
        logger.info(f"Saving best params to './models/{self.uuid}/{self.model_name}-best-params.json'")
        with open(f'./models/{self.uuid}/{self.model_name}-best-params.json', 'w') as f:
            json.dump(self.best_params, f)
        logger.info(f"Saved best params to './models/{self.uuid}/{self.model_name}-best-params.json'")
        
    def load_model(self) -> None:
        logger.info(f"Loading model from './models/{self.uuid}/{self.model_name}-model.pkl'")
        self.model = pickle.load(f'./models/{self.uuid}/{self.model_name}-model.pkl')
        logger.info(f"Loaded model from './models/{self.uuid}/{self.model_name}-model.pkl'")
        logger.info(f"Loading best model from './models/{self.uuid}/{self.model_name}-best-model.pkl'")
        self.best_model = pickle.load(f'./models/{self.uuid}/{self.model_name}-best-model.pkl')
        logger.info(f"Loaded best model from './models/{self.uuid}/{self.model_name}-best-model.pkl'")
        logger.info(f"Loading best params from './models/{self.uuid}/{self.model_name}-best-params.json'")
        with open(f'./models/{self.uuid}/{self.model_name}-best-params.json', 'r') as f:
            self.best_params = json.load(f)
        logger.info(f"Loaded best params from './models/{self.uuid}/{self.model_name}-best-params.json'")
    
class RNN(NN):
    def __init__(self) -> None:
        super().__init__()
        self.name = "RNN"
            
    def init_model(self) -> tf.keras.Model:
        logger.info("Initializing {self.name} model")
        self.model = Sequential([Embedding(input_dim = self.input_width, output_dim = INTERNAL_DIMENSION),
                            SimpleRNN(units = INTERNAL_DIMENSION, return_sequences = True),
                            SimpleRNN(units = INTERNAL_DIMENSION),
                            Dense(units = 1, activation = 'sigmoid')])
        logger.info("Model {self.name} initialized")
    
class LSTM(NN):
    def __init__(self) -> None:
        super().__init__()
        
    def init_model(self) -> tf.keras.Model:
        logger.info("Initializing {self.name} model")
        self.model = Sequential([Embedding(input_dim = self.input_width, output_dim = INTERNAL_DIMENSION),
                            LSTM(units = INTERNAL_DIMENSION, return_sequences = True),
                            LSTM(units = INTERNAL_DIMENSION),
                            Dense(units = 1, activation = 'sigmoid')])
        logger.info("Model {self.name} initialized")
    
class MNB(Ensemble):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "Multinomial Naive Bayes"
        self.params = { 'alpha': [ 1.0, 0.5, 0.1 ] }
        self.model = mnb(alpha = self.params['alpha'][0])
    
class KNN(Ensemble):
    def __init__(self, data) -> None:
        super().__init__(data)
        self.model_name = "K-Nearest Neighbors"
        self.params = { 'n_neighbors': [ 5 ],
                        'p': [ 2 ],
                        'weights': [ 'uniform' ],
                        'algorithm': [ 'kd_tree' ]
                      }
        self.model = knn(n_neighbors = self.params['n_neighbors'][0], 
                         p = self.params['p'][0], 
                         weights = self.params['weights'][0], 
                         algorithm = self.params['algorithm'][0])
        
    def grid_search(self) -> None:
        self.params = { 'n_neighbors': [ 10, 12, 15 ],
                        'p': [ 1, 2, 3 ],
                        'weights': [ 'uniform', 'distance' ],
                        'algorithm': [ 'ball_tree', 'kd_tree', 'brute' ]
                      }
        self.best_params = self.search().best_params_
        
    def best_fit(self) -> None:
        self.best_model = knn(n_neighbors=self.best_params['n_neighbors'], 
                              p=self.best_params['p'], 
                              weights=self.best_params['weights'], 
                              algorithm=self.best_params['algorithm'])
        self.best_model.fit(self.train_x, self.train_y)
        
class RFC(Ensemble):
    def __init__(self, data) -> None:
        super().__init__(data)
        self.model_name = "Random Forest Classifier"
        self.params = { 'n_estimators': [ 100 ],
                        'max_depth': [ 10 ],
                        'criterion': [ 'gini' ]
                      }
        self.model = rfcn(n_estimators = self.params['n_estimators'][0], 
                          max_depth = self.params['max_depth'][0], 
                          criterion = self.params['criterion'][0])
        
    def grid_search(self) -> None:
        self.params = { 'n_estimators': [ 100, 200, 300 ],
                        'max_depth': [ 10, 20, 30 ],
                        'criterion': [ 'gini', 'entropy' ]
                      }
        self.best_params = self.search().best_params_
        
    def best_fit(self) -> None:
        self.best_model = rfcn(n_estimators=self.best_params['n_estimators'], 
                               max_depth=self.best_params['max_depth'], 
                               criterion=self.best_params['criterion'])
        self.best_model.fit(self.train_x, self.train_y)

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
        
        self.add_argument("-u", 
                            "--uuid", 
                            help = "the model UUID to load", 
                            type = str, 
                            default = None)
        
        self.add_argument("-d",
                            "--seed",
                            help = "the seed to use",
                            type = int,
                            default = 42)
      
###############
###  AGENT  ###
###############
      
class Agent:
    def __init__(self) -> None:
        self.algorithms = [RNN, LSTM, MNB, KNN, RFC]
        self.uuid = parser.data['uuid']
        np.random.seed(parser.data['seed'])

    def run(self) -> None:
        # testing
        self.algorithms = [RNN]
        for algorithm in self.algorithms:
            model = algorithm()
            model.init_model()
            model.optimize()
            model.fit()
            model.predict()
            model.score()
            self.save()
            model.save()
            self.load()
            model.load()
            model.print()
            
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