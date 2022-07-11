import pickle
import numpy as np

from nltk import word_tokenize
from nltk.corpus import wordnet
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

def generate_dictionary_unigrams(path):
    """
    Generates a unigram dictionary file containig each word from the corpus given as path

    The structure of the generated file is: word number_of_appearences number_of_sencences_in_which_it_appears
    The name of the generated file is dictionary_unigrams

    Parameters:
    path (string): The path to the corpus file
    """
    f = open(path, "r")

    count = 1
    dictionary = {}
    for line in f:
        print(count, end='\r')
        tokens = word_tokenize(line)
        tokens = list(filter(lambda x: any(c.isalpha() for c in x), tokens))
        tokens = list(map(lambda x: x.lower(), tokens))
        seen = []
        for token in tokens:
            if token in dictionary:
                dictionary[token][0] += 1
                if token not in seen:
                    dictionary[token][1] += 1
                    seen.append(token)
            else:
                dictionary[token] = [1, 1]
        count += 1

    res = open("datasets/dictionary_unigrams", "w")
    for token in dictionary:
        res.write(token + " " + str(dictionary[token][0]) + " " + str(dictionary[token][1]) + "\n")


def generate_dictionary_bigrams(path):
    """
    Generates a bigram dictionary file containig each pair of consecutive words from the corpus given as path

    The structure of the generated file is: word1 word2 number_of_appearences
    The name of the generated file is dictionary_bigrams

    Parameters:
    path (string): The path to the corpus file
    """
    f = open(path, "r")

    count = 1
    dictionary = {}
    for line in f:
        print(count, end='\r')
        tokens = word_tokenize(line)
        tokens = list(filter(lambda x: any(c.isalpha() for c in x), tokens))
        tokens = list(map(lambda x: x.lower(), tokens))
        for i in range(len(tokens) - 1):
            if (tokens[i], tokens[i+1]) in dictionary:
                dictionary[((tokens[i], tokens[i+1]))] += 1
            else:
                dictionary[((tokens[i], tokens[i+1]))] = 1
        count += 1

    res = open("datasets/dictionary_bigrams", "w")
    for touple in dictionary:
        res.write(touple[0] + " " + touple[1] + " " + str(dictionary[touple]) + "\n")


def load_dictionary_unigrams(path):
    """
    Loads into memory the dictionary of unigrams given as path

    The returned dictionary structure is (word)->[number_of_appearences, number_of_sencences_in_which_it_appears]

    Parameters:
    path (string): The path to the unigram dictionary file

    Returs:
    class<dict>: The dictionary described above
    int: The total number of words in the corpus
    """
    f = open(path, "r")

    dictionary = {}
    for line in f:
        tokens = line.split(" ")
        dictionary[tokens[0]] = [int(tokens[1]), int(tokens[2])]

    return dictionary, sum(x[0] for x in dictionary.values())


def load_dictionary_bigrams(path):
    """
    Loads into memory the dictionary of bigrams given as path

    The returned dictionary structure is (word1, word2)->number_of_appearences

    Parameters:
    path (string): The path to the bigram dictionary file

    Returs:
    class<dict>: The dictionary described above
    """
    f = open(path, "r")

    dictionary = {}
    for line in f:
        tokens = line.split(" ")
        dictionary[(tokens[0], tokens[1])] = int(tokens[2])

    return dictionary


def generate_features(word, dictionary, total):
    """
    Generates a line vector with the features of a given word

    The features are the number of appearences of the word, the number of sentences in which the word appears,
    the unigram probability of the word, the length of the word and the size of its synset with respect to wordnet

    Parameters:
    word (string): The word for which to generate the features
    dictionary (dict): The unigram dictionary for the current corpus (see load_dictionary_unigrams)
    total (int): the total number of words in the corpus 

    Returns:
    np.array(5): A line vector containing the features
    """
    x = np.zeros((5))

    if word in dictionary:
        x[0] = dictionary[word][0]
        x[1] = dictionary[word][1]
        x[2] = dictionary[word][0] / total
    x[3] = len(word)

    synset = wordnet.synsets(word)
    x[4] = len(synset)

    return x


def train_model():
    """
    Train a model for word complexity calssfication; the model will be a shallow neural network

    Returns:
    model: The scikitlearn MLPClassifier object
    score (int): The accuracy score for the found model
    """
    dictionary = load_dictionary_unigrams("datasets/dictionary_unigrams")[0]
    total = sum(x[0] for x in dictionary.values())

    f = open("datasets/train_lexicon", "r")

    X = []
    y = []
    for line in f:
        tokens = line.split("\t")
        tokens[0] = tokens[0].lower()
        if tokens[0] not in dictionary:
            continue

        x = generate_features(tokens[0], dictionary, total)
        X.append(x)
        if float(tokens[1]) >= 3:
            y.append(1)
        else:
            y.append(0)

    X = np.asarray(X)
    y = np.asarray(y)

    f = open("datasets/test_lexicon", "r")

    X_test = []
    y_test = []
    for line in f:
        tokens = line.split("\t")
        tokens[0] = tokens[0].lower()
        if tokens[0] not in dictionary:
            continue

        x = generate_features(tokens[0], dictionary, total)
        X_test.append(x)
        if float(tokens[1]) >= 3:
            y_test.append(1)
        else:
            y_test.append(0)

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    model = MLPClassifier(hidden_layer_sizes=(3), max_iter=1000)
    model.fit(X, y)

    return model, model.score(X_test, y_test)


def search_for_best():
    """
    Runs 20 iteration for the model described in the train_model() function and saves with pickle the best model found
    """
    best_score = -1

    for i in range(20):
        print("Current iteration: " + str(i), end="\r")
        model, score = train_model()
        if score > best_score:
            pickle.dump(model, open("best_model.sav", "wb"))
            best_score = score

    print("Best score: " + str(best_score))


def predict_interactive(model, dictionary):
    """
    Interactive debug function in which a word can be entered via the keyboard and a prediction will be made

    Parameters:
    model (scikitlearn model): a pre-trained model object
    dictionary (dict): a dictionary of unigrams for a language model (see generate_dictionary_unigrams)
    """
    while(True):
        word = input()
        if word not in dictionary:
            print("Word is not in dictionary!")
        else:
            word = word.lower()
            total = sum(x[0] for x in dictionary.values())
            x = generate_features(word, dictionary, total)
            X_test = [x]
            print("Prediction for " + word + " is " + str(model.predict(X_test)[0]))



def predict(model, dictionary, total, word):
    """
    Predict if a given word is complex or not, based on a given ML model and language model

    Parameters:
    model (scikitlearn model): a pre-trained model object
    dictionary (dict): a dictionary of unigrams for a language model (see generate_dictionary_unigrams)
    total (int): total number of words in the language model
    word (str): the word for which the prediction should be made

    Returns:
    int: 0(the word is not complex) or 1(the word is complex)
    """
    if word not in dictionary:
        return 1
    else:
        word = word.lower()
        x = generate_features(word, dictionary, total)
        X_test = [x]
        return model.predict(X_test)[0]


def get_confusion_matrix(model, dictionary):
    """
    Get the full confusion matrix for a given model

    Parameters:
    model (scikitlearn model): a pre-trained model object
    dictionary (dict): a dictionary of unigrams for a language model (see generate_dictionary_unigrams)

    Returns:
    scikitlearn confusion matrix: a scikitlearn object containing the full statistics for the performances of the current model
    """
    total = sum(x[0] for x in dictionary.values())
    f = open("datasets/test_lexicon", "r")
    
    X_test = []
    y_test = []
    for line in f:
        tokens = line.split("\t")
        tokens[0] = tokens[0].lower()
        if tokens[0] not in dictionary:
            continue

        x = generate_features(tokens[0], dictionary, total)
        X_test.append(x)
        if float(tokens[1]) >= 3:
            y_test.append(1)
        else:
            y_test.append(0)

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    y_pred = model.predict(X_test)
    return metrics.classification_report(y_test, y_pred, labels=[0, 1])
