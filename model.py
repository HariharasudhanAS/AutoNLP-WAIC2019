# python3 run_local_test.py -dataset_dir="./AutoDL_sample_data/O5" -code_dir="./AutoDL_sample_code_submission/svm"
import os
import re
import sys
from time import time
import jieba
import logging
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
try:
    import lightgbm as lgb
except:
    os.system('pip install lightgbm')
    import lightgbm as lgb
warnings.filterwarnings('ignore')

MAX_VOCAB_SIZE = 20000

def clean_en_text(dat):
    REPLACE_BY_SPACE_RE = re.compile(r'[\/\!\'\"\<\>\?\%\$\:\=\_\&\(\)\{\}\[\]\|\@\,\-\.\;]')
    REPLACE_BY_SPACE_RE2 = re.compile(r'[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')
    ret = []
    for line in dat:
        line = line.lower()
        line = re.sub(r"\0rs ", " rs ", line)
        line = re.sub(r'i\'m', 'i am', line)
        line = re.sub(r'he\'s', 'he is', line)
        line = re.sub(r'she\'s', 'she is', line)
        line = re.sub(r'that\'s', 'that is', line)
        line = re.sub(r'what\'s', 'what is ', line)
        line = re.sub(r'where\'s', 'where is ', line)
        line = re.sub(r"\'s", " ", line)
        line = re.sub(r'\'d', ' would', line)
        line = re.sub(r'can\'t', 'cannot', line)
        line = re.sub(r'won\'t', 'will not', line)
        line = re.sub(r'doesn\'t', 'does not', line)
        line = re.sub(r"n\'t", " not ", line)
        line = re.sub(r"\'ve", " have", line)
        line = re.sub(r"\'re", " are ", line)
        line = re.sub(r"\'ll", " will ", line)
        line = re.sub(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', ' email', line)
        line = re.sub(r'^@?(\w){1,15}$', ' handle', line)
        line = re.sub(r'\B(\#[a-zA-Z]+\b)(?!;)', ' hashtag', line)
        line = re.sub(r"\.\.", " ", line)
        line = re.sub(r"\0s", "0", line)
        line = re.sub(r"\-", " ", line)
        line = re.sub(r"\s{2,}", " ", line)
        line = line.strip()
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = REPLACE_BY_SPACE_RE2.sub(' ', line)
        line = re.sub(r"[^A-Za-z]", " ", line)
        line = line.strip()
        ret.append(' '.join([y for y in line.split(' ') if len(y)>1]))
    return ret

def clean_zh_text(dat):
    REPLACE_BY_SPACE_RE = re.compile(r'[\/\!\'\"\<\>\?\%\$\:\=\_\&\(\)\{\}\[\]\|\@\,\-\.\;]')
    REPLACE_BY_SPACE_RE2 = re.compile(r'[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')
    jieba_stop_words = [
        '的', '了', '和', '是', '就', '都', '而', '及', '與', 
        '著', '或', '一個', '沒有', '我們', '你們', '妳們', 
        '他們', '她們', '是否'
    ]
    ret = []
    for line in dat:
        line = line.strip()
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = REPLACE_BY_SPACE_RE2.sub(' ', line)
        line = re.sub(r"[0-9]", " ", line)
        line = ' '.join(jieba.cut(line, cut_all=True))
        line = ' '.join([x for x in line.split(' ') if x not in jieba_stop_words])
        line = line.strip()
        ret.append(line)
    return ret

def vectorize_data(x_train, x_val=None):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        max_features = MAX_VOCAB_SIZE)
    if x_val:
        full_text = x_train + x_val
    else:
        full_text = x_train
    vectorizer.fit(full_text)
    train_vectorized = vectorizer.transform(x_train)
    if x_val:
        val_vectorized = vectorizer.transform(x_val)
        return train_vectorized, val_vectorized, vectorizer
    return train_vectorized, vectorizer

# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)

class Model(object):
    def __init__(self, metadata):
        """
        Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 10,
             "language": ZH,
             "num_train_instances": 10000,
             "num_test_instances": 1000,
             "time_budget": 300}
        """
        self.done_training = False
        self.metadata = metadata
        self.train_output_path = './'
        self.test_input_path = './'
        self.current_samples = 1000
        self.last_run = False
        self.yet_to_ensure = True
        self.first_time_test = True

    def train(self, train_dataset, remaining_time_budget=None):
        """
        Model training on train_dataset.
        :param train_dataset: tuple, (x_train, y_train)
            x_train: list of str, input training sentences.
            y_train: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
        """
        if self.done_training:
            return
        _start_ = time()
        x_train, y_train = train_dataset
        if self.yet_to_ensure:
            # To ensure solver gets samples of at least 2 classes in the data
            while len(Counter(list(ohe2cat(y_train[:self.current_samples]))).keys())<=1:
                self.current_samples = self.current_samples*10
            self.yet_to_ensure = False
        # Clean Chinese words
        _start = time()
        if self.metadata['language'] == 'ZH':
            x_train = clean_zh_text(x_train[:self.current_samples])
        else:
            x_train = clean_en_text(x_train[:self.current_samples])
        logger.info('Train data cleaning took {} sec'.format(time()-_start))
        logger.info('Training on {} samples'.format(len(x_train)))
        _start = time()
        x_train, self.tokenizer = vectorize_data(x_train)
        logger.info('Tokenizing train data took {} sec'.format(time()-_start))
        _start = time()
        d_train = lgb.Dataset(x_train, label=ohe2cat(y_train[:self.current_samples]))
        params = {}
        params['learning_rate'] = 0.003
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'multiclass'
        params['num_class'] = self.metadata['class_num']
        params['sub_feature'] = 0.5
        params['num_leaves'] = 10
        params['min_data'] = 50
        params['max_depth'] = 10
        # For faster speed
        params['bagging_fraction'] = 0.7
        params['bagging_freq'] = 2
        params['max_bin'] = 64
        # 
        self.model = lgb.train(params, train_set=d_train, num_boost_round=100)
        logger.info('Training took {} sec'.format(time()-_start))
        _time_ = time()-_start_
        logger.info('Total train function time {} sec'.format(_time_))
        logger.info('Current position on graph time: {}'.format(self.trans_time(2400-remaining_time_budget+_time_)))
        if self.last_run:
            logger.info('==========Classes distribution:==========')
            logger.info(Counter(ohe2cat(y_train)))
            self.done_training = True
            return
        if self.current_samples*10 > len(y_train):
            self.current_samples = len(y_train)
            self.last_run = True
        else:
            self.current_samples = self.current_samples*10

    def test(self, x_test, remaining_time_budget=None):
        """
        :param x_test: list of str, input test sentences.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                 here `sample_count` is the number of examples in this dataset as test
                 set and `class_num` is the same as the class_num in metadata. The
                 values should be binary or in the interval [0,1].
        """
        # Clean Chinese words
        _start_ = time()
        if self.first_time_test:
            _start = time()
            if self.metadata['language'] == 'ZH':
                self.x_test = clean_zh_text(x_test)
            else:
                self.x_test = clean_en_text(x_test)
            self.first_time_test = False
            logger.info('Test Data cleaning took {} sec'.format(time()-_start))
        logger.info('Testing on {} samples'.format(len(self.x_test)))
        _start = time()
        x_test = self.tokenizer.transform(self.x_test)
        logger.info('Tokenizing test data took {} sec'.format(time()-_start))
        _start = time()
        result = self.model.predict(x_test)
        logger.info('Testing took {} sec'.format(time()-_start))
        _time_ = time()-_start_
        logger.info('Total test function time {} sec'.format(_time_))
        logger.info('Current position on graph time: {}'.format(self.trans_time(2400-remaining_time_budget+_time_)))
        # return y_test
        return result

    def trans_time(self,x):
        assert x<=2400 and x>=0
        return (np.log(1+(x/60)))/np.log(41)

    def inv_tans_time(self,x):
        assert x>1.0 and x<0
        return (np.exp(x*np.log(41))-1)*60
    

def get_logger(verbosity_level):
    """Set logging format to something like:
        2019-04-25 12:52:51,924 INFO model.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger

logger = get_logger('INFO')