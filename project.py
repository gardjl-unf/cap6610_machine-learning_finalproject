#!/usr/bin/env python

__author__ = "Jason Gardner"
__credits__ = ["Jason Gardner"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Jason Gardner"
__email__ = "n01480000@unf.edu"
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
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb as imdb
import re

TEST = False
DEBUG = True

SEED = 42
TEST_RATIO= 0.2
LOG_FORMAT_STRING = logging.Formatter("%(asctime)s — %(name)s — %(funcName)s:%(lineno)d — %(message)s")
RMSPROP_CLIP = 10.0
INPUT_SHAPE = (43)

np.random.seed(SEED)

class Data:
    def __init__(self, test: bool = False) -> None:
        self.test = test
        self.x_train, self.y_train, self.x_test, self.y_test = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
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
        
        return x_train, y_train, x_test, y_test

    def _process_data(self) -> None:
        print(self.x_train)
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
        logger.info("Padding sequences")
        data = tf.keras.preprocessing.sequence.pad_sequences(data, padding = "post", maxlen = 256)
        logger.info("Sequences padded")
        
        return data

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        return self.x_train, self.x_test, self.y_train, self.y_test
    
###############
###  MODEL  ###
###############

class Model:
    def __init__(self, state : dict = None, size: int = None) -> None:
        self.state = state
        self.size = size
        self.uuid = self.state['uuid']

        if self.state['uuid'] is not None:
            print(f"Loading model from {self.uuid}")
            self.load_model()
        else:
            logger.info("No model UUID provided. Generating new UUID")
            self.state['uuid'] = str(uuid.uuid4())
            self.uuid = self.state['uuid']
            logger.info(f"New UUID: {self.uuid}")
            logger.info(f"Initializing new model")
            self.init_model()

    def init_model(self) -> tf.keras.Model:
        logger.info("Initializing model")
        self.model = tf.keras.models.Sequential([
            layers.Input(shape = (self.size, 43)),
            layers.Conv1D(filters = 32, kernel_size = 3, strides = 1, activation = "relu", padding = "same"),
            layers.MaxPooling1D(pool_size = 2, padding = "same"),
            layers.Conv1D(filters = 64, kernel_size = 3, strides = 1, activation = "relu", padding = "same"),
            layers.MaxPooling1D(pool_size = 2, padding = "same"),
            layers.LSTM(units = 256, return_sequences = True),
            layers.LSTM(units = 128),
            layers.Dense(units = 128, activation="relu"),
            layers.Dense(units = 2, activation="sigmoid")
        ])
      
        self.optimizer = tf.keras.optimizers.RMSprop(clipvalue = RMSPROP_CLIP)
        # self.optimizer = tf.keras.optimizers.Adadelta(learning_rate = 0.1, ema_momentum = 0.95)
        # self.optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        self.loss_function = tf.keras.losses.Huber()
        # self.loss_function = tf.keras.losses.mean_squared_error
        self.model.compile(optimizer = self.optimizer, loss = self.loss_function)
        logger.info("Model initialized")
           
    def save_model(self) -> None:
        logger.info(f"Saving model to './models/{self.uuid}/model.h5'")
        logger.info(f"Creating directory './models/{self.uuid}'")
        results_dir_path = f"./models/{self.uuid}"
        if not os.path.exists(results_dir_path):
            try:
                os.mkdir(results_dir_path)
            except OSError:
                logger.warning(f"Creation of the directory {results_dir_path} failed")
                exit(1)
            else:
                logger.info(f"Successfully created the directory {results_dir_path}")

        self.model.save_weights(f'./{self.state["uuid"]}/model.weights.h5')
        logger.info(f"Saved model weights to ./models/{self.uuid}/model.weights.h5")

        logger.info(f"Saving Numpy random state to ./models/{self.uuid}/numpy_random_state.pkl")
        with open(f"./models/{self.uuid}/numpy_random_state.pkl", 'wb') as f:
            pickle.dump(np.random.get_state(), f)
        logger.info(f"Saved Numpy random state to ./models/{self.uuid}/random_state.npy")

        with open(f'./{self.state["uuid"]}/state.json', 'w') as f:
            logger.info(f"Saving Network with UUID {self.uuid}")
            json.dump(self.state, f)
            logger.info(f"Saved state to './{self.uuid}/state.json'")

    def load_model(self) -> tf.keras.Model:
        self.model = tf.keras.Model()
        logger.info(f"Loading model from './{self.uuid}/model.weights.h5'")
        self.model.load_weights(f'./{self.uuid}/model.weights.h5')
        logger.info(f"Loaded model weights from './{self.uuid}/model.weights.h5'")
        
        with open(f"./models/{self.uuid}/numpy_random_state.pkl", 'rb') as f:
            random_state = pickle.load(f)
            np.random.set_state(random_state)
        logger.info(f"Loaded Numpy random state from './{self.uuid}/numpy_random_state.pkl'")

        if os.path.exists(f'./{self.uuid}/state.json'):
            with open(f'./{self.uuid}/state.json', 'r') as f:
                logger.info(f"Loading state from './{self.uuid}/state.json'")
                self.state = json.load(f)
                logger.info(f"Loading {self.state['network']} ({self.uuid}) network environment.\n Game: {self.state['environment']}")
                logger.info(f"Loaded state from './{self.uuid}/state.json'")     

        return self.model

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
        super().__init__(prog = "DQPOMDP", 
                        description = "Train an LSTM model on the NF-UQ-NIDS-v2 dataset to detect network intrusions",
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
    def __init__(self, state: dict, data : tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
        self.uuid = state['uuid']
        self.optimizer = tf.keras.optimizers.RMSprop(clipvalue = RMSPROP_CLIP)
        self.loss_function = tf.keras.losses.Huber()
        self.x_train, self.x_test, self.y_train, self.y_test = data
        self.M = Model(state = parser.data, size = len(self.y_train))
        self.model = self.M.model
        
        self.debug = None

    def run(self) -> None:
        print(f"Data: {self.x_train.shape}")
        self.model.fit(self.x_train, self.y_train, epochs = 10, batch_size = 32, verbose = 1, validation_data = (self.x_test, self.y_test))
        self.model.save_model()
        self.model.evaluate(self.x_test, self.y_test)
        self.model.predict(self.x_test)

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
            if not os.path.exists(f'./{self.data["uuid"]}'):
                logger.info(f"The Model at ./{self.data['uuid']} does not exist. Exiting...")
                exit(1)

###############
###  MAIN   ###
###############

if __name__ == '__main__':
    logger = Logging().get_logger()
    args = Arguments().parse_args()
    parser = Parsing(args)
    data = Data().get_data()
    agent = Agent(parser.data, data)
    agent.run()
    logger.info("Training complete")
    exit(0)