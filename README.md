# lang_assignment4

## Language analytics assignment 4 description
Assignment 3 - Text classification

We've seen in the past few sessions how text classification works, both from the perspective of classical machine learning and from more contemporary deep learning perspectives. In most cases, we've seen how a simple model like a LogisticRegression Classifier provides a solid benchmark which we can build on using more sophisticated models.

We've also seen that word embeddings can be employed to make dense vector representations of words, encoding linguistic information about the word and it's relationship to other words. In class this week, we saw that these word embeddings can be fed into classifiers, potentially improving the performance and generalizability of our classifier models.

The assignment for this week builds on these concepts and techniques. We're going to be working with the data in the folder CDS-LANG/toxic and trying to see if we can predict whether or not a comment is a certain kind of toxic speech. You should write two scripts which do the following:

    The first script should perform benchmark classification using standard machine learning approaches
        This means CountVectorizer() or TfidfVectorizer(), LogisticRegression classifier
        Save the results from the classification report to a text file
    The second script should perform classification using the kind of deep learning methods we saw in class
        Keras Embedding layer, Convolutional Neural Network
        Save the classification report to a text file
Bonus tasks

    Add a range of different Argparse parameters that would allow the user to interact with the code, such as the embedding dimension size, the CountVector parameters.
        Think about which parameters are most likely to be modified by a user.

## Methods
This assignment compares the performance of two classifier models, one a logistic regression classifier, the other a convolutional neural network. The logistic regression classifier is quite simple compared to the CNN.
For this, two scripts were made: ```logistic_regression.py``` and ```keras_cnn.py```. 
The logistic regression classifier is quite simple, it loads in the data, splits it into training and testing data. Then it builds a logistic regression classifier, which then makes predictions on the testing data. The output is then saved to a csv with a user-defined filename.
The cnn is quite a bit more complex. It first loads in the data and makes a train/test split, then uses a pre-defined cleaning and normalising function, then it tokenizes the sequences. 
Then it builds the model, with a user-defined embedding size and learning rate. It then trains the model with a user-defined batch size and user-defined amount of epochs. Lastly it creates a classification report which is saved to a csv with a user-defined filename.

## Usage
First, you must put the corpus into the ```in``` folder, which you can find here: https://www.simula.no/sites/default/files/publications/files/cbmi2019_youtube_threat_corpus.pdf 
Then, run the ```setup_lang.sh``` script.
The ```logistic_regression.py``` script is quite simple. To run it, point the command line to the ```lang_assignment4``` folder and run the script from the ```src``` folder. The only command line argument you need to include is ```-rp``` which is the name you will give the classification report, for instance ```classification_rep.csv```.
The ```keras_cnn.py``` has more command line arguments:
```-rp``` which is the same as the ```logistic_regression.py``` argument.
```-em``` which is the size of the embedding layer. I've used 256 
```-e``` which is the amount of epochs. I've used 10
```b``` which is the batch size of the model. I've used 128
```lr``` which is the learning rate of the model. I've used 0.001, which seems to be the recommended value for the ```adam``` optimizer.

## Results
Fairly unsurprisingly, the cnn performs quite a bit better than the logistic regression classifier. This just shows how powerful a tool neural networks can be when working with language. That being said, the cnn also takes quite a bit longer to run. In this case, the amount of text isn't huge, but if one were working with a larger dataset, it could take a long time to run such a model, especially if one didn't have access to a super-computer like we do.
