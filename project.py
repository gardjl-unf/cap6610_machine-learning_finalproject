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
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import RandomForestClassifier as rfcn
from sklearn.tree import DecisionTreeClassifier as dtc
import re
import sys, os, warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

SEED = 42
LOG_FORMAT_STRING = logging.Formatter("%(asctime)s — %(name)s — %(funcName)s:%(lineno)d — %(message)s")
RMSPROP_CLIP = 10.0

np.random.seed(SEED)

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
        logger.info("Loading data")
        (x_train, y_train), (x_test, y_test) = imdb.load_data()
        logger.info("Data loaded")
        
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

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        return self.x_train, self.x_test, self.y_train, self.y_test
    
################
###  MODELS  ###
################

class Model:
    def __init__(self, state : dict = None, input_width: int = None) -> None:
        self.state = state
        self.input_width = input_width
        self.uuid = self.state['uuid']

        if self.state['uuid'] is not None:
            logger.info(f"Loading model from ./model/{self.uuid}")
            self.model = self.load_model()
        else:
            logger.info("No model UUID provided. Generating new UUID")
            self.state['uuid'] = str(uuid.uuid4())
            self.uuid = self.state['uuid']
            logger.info(f"New UUID: {self.uuid}")
            logger.info(f"Initializing new model")
            self.model = self.init_model()

    def init_model(self) -> tf.keras.Model:
        logger.info("Initializing model")
        model = Sequential([Embedding(input_dim = self.input_width, output_dim = 128),
                            SimpleRNN(units = 128, return_sequences = True),
                            SimpleRNN(units = 128),
                            Dense(units = 1, activation = 'sigmoid')])
      
        optimizer = tf.keras.optimizers.RMSprop(clipvalue = RMSPROP_CLIP)
        # self.optimizer = tf.keras.optimizers.Adadelta(learning_rate = 0.1, ema_momentum = 0.95)
        # self.optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        loss_function = tf.keras.losses.Huber()
        # self.loss_function = tf.keras.losses.mean_squared_error
        # self.loss_function = tf.keras.losses.binary_crossentropy
        model.compile(optimizer = optimizer, loss = loss_function)
        logger.info("Model initialized")
        
        return model
           
    def save_model(self) -> None:
        logger.info(f"Saving model to './models/{self.uuid}/model.h5'")
        logger.info(f"Creating directory './models/{self.uuid}'")
        results_dir_path = f"./models/{self.uuid}"
        if not os.path.exists(results_dir_path):
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

        self.model.save_weights(f'./models/{self.uuid}/model.weights.h5')
        logger.info(f"Saved model weights to ./models/{self.uuid}/model.weights.h5")

        logger.info(f"Saving Numpy random state to ./models/{self.uuid}/numpy_random_state.pkl")
        with open(f"./models/{self.uuid}/numpy_random_state.pkl", 'wb') as f:
            pickle.dump(np.random.get_state(), f)
        logger.info(f"Saved Numpy random state to ./models/{self.uuid}/random_state.npy")

        with open(f'./models/{self.uuid}/state.json', 'w') as f:
            logger.info(f"Saving Network with UUID {self.uuid}")
            json.dump(self.state, f)
            logger.info(f"Saved state to './models/{self.uuid}/state.json'")

    def load_model(self) -> tf.keras.Model:
        model = tf.keras.Model()
        logger.info(f"Loading model from './models/{self.uuid}/model.weights.h5'")
        model.load_weights(f'./models/{self.uuid}/model.weights.h5')
        logger.info(f"Loaded model weights from './models/{self.uuid}/model.weights.h5'")
        
        with open(f"./models/{self.uuid}/numpy_random_state.pkl", 'rb') as f:
            random_state = pickle.load(f)
            np.random.set_state(random_state)
        logger.info(f"Loaded Numpy random state from './models/{self.uuid}/numpy_random_state.pkl'")

        if os.path.exists(f'./models/{self.uuid}/state.json'):
            with open(f'./models/{self.uuid}/state.json', 'r') as f:
                logger.info(f"Loading state from './models/{self.uuid}/state.json'")
                self.state = json.load(f)
                logger.info(f"Loaded state from './models/{self.uuid}/state.json'")     

        return model

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
    def __init__(self, state: dict, data : tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], input_width: int) -> None:
        self.algorithms = [#("RNN", SimpleRNN()),
                           #("LSTM", LSTM()),
                           #("Neural Network", nn()),
                           ("Naive Bayes", MultinomialNB()),
                           ("Decision Tree", dtc()),
                           ("Logistic Regression", lr()),
                           ("Random Forest", rfcn()),
                           ("KNN", knn())]
        self.uuid = state['uuid']
        self.input_width = input_width
        self.optimizer = tf.keras.optimizers.RMSprop(clipvalue = RMSPROP_CLIP)
        # self.optimizer = tf.keras.optimizers.Adadelta(learning_rate = 0.1, ema_momentum = 0.95)
        # self.optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        self.loss_function = tf.keras.losses.Huber()
        # self.loss_function = tf.keras.losses.mean_squared_error
        self.x_train, self.x_test, self.y_train, self.y_test = data
        self.M = Model(state = parser.data, input_width = self.input_width)
        self.model = self.M.model

    def run(self) -> None:
        self.model.fit(self.x_train, self.y_train, epochs = 10, batch_size = 32, verbose = 1, validation_data = (self.x_test, self.y_test))
        self.M.save_model()
        self.model.evaluate(self.x_test, self.y_test)
        self.model.predict(self.x_test)
        self.model.summary()

##########################
###  ARGUMENT PARSING  ###
##########################

class Parsing:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args : dict = args
        self.model : tf.keras.Model = None
        self.data : dict = {}
        for key, value in vars(args).items():
                self.data[key] = value

        if args.uuid is not None:
            if not os.path.exists(f'./models/{self.data["uuid"]}'):
                logger.info(f"The Model at ./models/{self.data['uuid']} does not exist. Exiting...")
                exit(1)

###############
###  MAIN   ###
###############

if __name__ == '__main__':
    logger = Logging().get_logger()
    args = Arguments().parse_args()
    parser = Parsing(args)
    data = Data().get_data()
    logger.info(f"Model data width: {len(data[0][0])}")
    input_width = len(data[0][0])
    agent = Agent(parser.data, data, input_width)
    agent.run()
    logger.info("Training complete")
    exit(0)