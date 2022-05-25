# load some packages

# setup script wasn't working in class - uncomment if you get errors loading modules
#!pip install nltk beautifulsoup4 contractions tensorflow scikit-learn
import os
import sys
#sys.path.append(os.path.join("..", "CDS-LANG"))
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split

# simple text processing tools
import re
import tqdm
import unicodedata
import contractions
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')

# data wranling
import pandas as pd
import numpy as np

# tensorflow
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, 
                                    Flatten,
                                    Conv1D, 
                                    MaxPooling1D, 
                                    Embedding)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.regularizers import L2

# scikit-learn
from sklearn.metrics import (confusion_matrix, 
                            classification_report)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

# import argparse
import argparse
# visualisations 
#import matplotlib.pyplot as plt
#%matplotlib inline


# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

# defines some helper functions for text processing we saw in class

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def pre_process_corpus(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        doc = strip_html_tags(doc)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = doc.lower()
        doc = remove_accented_chars(doc)
        doc = contractions.fix(doc)
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()  
        norm_docs.append(doc)
  
    return norm_docs

def load_preprocess():
    filepath = os.path.join("..", "..", "..", "CDS-LANG", "toxic", "VideoCommentsThreatCorpus.csv")
    data = pd.read_csv(filepath)
    data_balanced = clf.balance(data, 1000)
    # now we split the data into training and testing 
    X = data_balanced["text"]
    y = data_balanced["label"]
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, 
                                                        test_size = 0.25, #this creates a 75/25 split
                                                        random_state = 42) # random state for reproducibility
    
    # using a pre-defined cleaning and normalizing function
    X_train_norm = pre_process_corpus(X_train)
    X_test_norm = pre_process_corpus(X_test)
    
    # define out-of-vocabulary token
    t = Tokenizer(oov_token = '<UNK>')
    # fit the tokenizer on then documents
    t.fit_on_texts(X_train_norm)
    # set padding value
    t.word_index["<PAD>"] = 0 
    # tokenize sequences 
    X_train_seqs = t.texts_to_sequences(X_train_norm)
    X_test_seqs = t.texts_to_sequences(X_test_norm)
    
    # sequence normalization
    MAX_SEQUENCE_LENGTH = 1000
    
    # add padding to sequences
    X_train_pad = sequence.pad_sequences(X_train_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    X_test_pad = sequence.pad_sequences(X_test_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    
    # encoding labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    
    # define paramaters for model
    # overall vocublarly size
    VOCAB_SIZE = len(t.word_index)
    
    return X_train_pad, X_test_pad, y_train, y_test, t, MAX_SEQUENCE_LENGTH, VOCAB_SIZE

def create_compile_mdl(VOCAB_SIZE, embed_size, MAX_SEQUENCE_LENGTH, learn_rate):
    # create the model
    model = Sequential()
    # embedding layer
    model.add(Embedding(VOCAB_SIZE, 
                        int(embed_size), 
                        input_length=MAX_SEQUENCE_LENGTH))

    # first convolution layer and pooling
    model.add(Conv1D(filters=128, 
                            kernel_size=4, 
                            padding='same',
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # second convolution layer and pooling
    model.add(Conv1D(filters=64, 
                            kernel_size=4, 
                            padding='same', 
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=32, 
                            kernel_size=4, 
                            padding='same', 
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # fully-connected classification layer
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                            optimizer= keras.optimizers.Adam(learning_rate = float(learn_rate)), 
                            metrics=['accuracy'])
    return model
    
# train the model
def train_eval_mdl(model, X_train_pad, y_train, eps, b_size, X_test_pad, y_test, rep_name):
    model.fit(X_train_pad, 
              y_train, 
              epochs = int(eps),
              batch_size = int(b_size),
              validation_split = 0.1, 
              verbose = True)
    
    # 0.5 decision boundary
    predictions = (model.predict(X_test_pad) > 0.5).astype("int32")
    # assign labels
    #predictions = ['threat' if item == 1 else 'non-threat' for item in predictions]
    
    # confusion matrix and classification report
    #labels = ['non-threat', 'threat']
    clf_report = classification_report(y_test, predictions, output_dict = True)
    outpath = ("out")
    report_df = pd.DataFrame(clf_report).transpose()
    report_df = report_df.rename(index = {"0" : "non-toxic", "1" : "toxic"})
    report_df.to_csv(os.path.join(outpath, rep_name))
    return report_df

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-rp", "--rep_name", required = False, help = "Filename of the classification report")
    ap.add_argument("-em", "--embed_size", required = True, help = "The size of the embedding layer")
    ap.add_argument("-e", "--eps", required = True, help = "The amount of epochs you want to train for")
    ap.add_argument("-b", "--b_size", required = True, help = "The batch size of the model")
    ap.add_argument("-lr", "--learn_rate", required = True, help = "The learning rate of the model")
    args = vars(ap.parse_args())
    return args

def main():
    args = parse_args()
    X_train_pad, X_test_pad, y_train, y_test, t, MAX_SEQUENCE_LENGTH, VOCAB_SIZE = load_preprocess()
    model = create_compile_mdl(VOCAB_SIZE, args["embed_size"], MAX_SEQUENCE_LENGTH, args["learn_rate"])
    report_df = train_eval_mdl(model, X_train_pad, y_train, args["eps"], args["b_size"], X_test_pad, y_test, args["rep_name"])

if __name__ == "__main__":
    main()
    


    
    
