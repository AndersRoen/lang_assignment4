# system tools
import os
import sys
import argparse

# data munging tools
import pandas as pd
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

# Visualisation
import matplotlib.pyplot as plt
#import seaborn as sns

#surpress warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
# warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_data(): #load the data and balance it
    filepath = os.path.join("in", "VideoCommentsThreatCorpus.csv")
    data = pd.read_csv(filepath)
    # balancing the data
    data_balanced = clf.balance(data, 1000)
    # now we split the data into training and testing 
    X = data_balanced["text"]
    y = data_balanced["label"]
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = 0.25, #this creates a 75/25 split
                                                        random_state = 42) # random state for reproducibility
    return X_train, X_test, y_train, y_test

def vectorize_fit(X_train, X_test):
    vectorizer = TfidfVectorizer(ngram_range = (1,3),
                             lowercase = True,
                             max_df = 0.93,
                             min_df = 0.07,
                             max_features = 150)
   
    # fit the vectorizer to the training data
    X_train_feats = vectorizer.fit_transform(X_train)
    
    # the same for test data
    X_test_feats = vectorizer.transform(X_test)
    return X_train_feats, X_test_feats

def classify_predict(X_train_feats, y_train, X_test_feats, y_test, rep_name):
    classifier = LogisticRegression(random_state = 42).fit(X_train_feats, y_train)
    y_pred = classifier.predict(X_test_feats)
    cl_report = metrics.classification_report(y_test, y_pred, output_dict = True)
    report_df = df = pd.DataFrame(cl_report).transpose()
    outpath = ("out")
    report_df.to_csv(os.path.join(outpath, rep_name))
    return report_df

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-rp", "--rep_name", required = False, help = "Filename of the classification report")
    args = vars(ap.parse_args())
    return args

def main():
    args = parse_args()
    X_train, X_test, y_train, y_test = load_data()
    X_train_feats, X_test_feats = vectorize_fit(X_train, X_test)
    report_df = classify_predict(X_train_feats, y_train, X_test_feats, y_test, args["rep_name"])
    
if __name__ == "__main__":
    main()
    
