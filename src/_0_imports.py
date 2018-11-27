from collections import defaultdict
from itertools import combinations

import numpy as np
import os

import pandas as pd
from scipy.stats import gmean
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
from torch import nn

import os
import time

import torch
from sklearn.metrics import f1_score
from torch import nn
import os

import torch
from sklearn.model_selection import StratifiedKFold
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors

from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from fuzzywuzzy import fuzz

import networkx as nx

from collections import Counter

from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, NMF

import warnings

warnings.filterwarnings('ignore')

label = pd.read_csv('../input/train.csv')['label'].values
truth = pd.read_csv('../input/train.csv')['label'].values
