import os.path, logging
import numpy as np
import pandas as pd
from keras.layers import Input, Embedding, concatenate
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

