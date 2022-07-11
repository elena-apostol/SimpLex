# coding: utf-8

import pickle
import numpy as np

from nltk import word_tokenize
from nltk.corpus import wordnet
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split

ITER = 10

def generate_dictionary_unigrams(path):
    """
    Generates a unigram dictionary file containig each word from the corpus given as path

    The structure of the generated file is: word number_of_appearences number_of_sencences_in_which_it_appears
    The name of the generated file is dictionary_unigrams

    Parameters:
    path (string): The path to the corpus file
    """
    f = open(path, "r", encoding="utf-8")

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

    res = open("datasets/dictionary_unigrams", "w", encoding="utf-8")
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
    f = open(path, "r", encoding="utf-8")

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

    res = open("datasets/dictionary_bigrams", "w", encoding="utf-8")
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
    f = open(path, "r", encoding="utf-8")

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
    f = open(path, "r", encoding="utf-8")

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

def create_dataset():
    """
    create the dataset

    Returns:
    model: The scikitlearn MLPClassifier object
    score (int): The accuracy score for the found model
    """
    dictionary = load_dictionary_unigrams("datasets/dictionary_unigrams")[0]
    total = sum(x[0] for x in dictionary.values())

    f = open("datasets/full_lexicon", "r", encoding="utf-8")

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
    return X, y

def train_model(X_train, X_test, y_train, y_test, model_name="MLP"):
    """
    Train a model for word complexity calssfication; the model will be a shallow neural network

    Returns:
    model: The scikitlearn MLPClassifier object
    score (int): The accuracy score for the found model
    """
    
    

    if model_name == "MLP":
        model = MLPClassifier(hidden_layer_sizes=(3), max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_name == "SVM":
        model = SVC(gamma='auto')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_name == "RFC":
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_name == "ETC":
        model = ExtraTreesClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=[0, 1])

    return model, accuracy, precision, recall, report

def create_dataset_iteration():
    X, y  = create_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y)
    return X_train, X_test, y_train, y_test

def search_for_best(model_name="MLP"):
    """
    Runs 20 iteration for the model described in the train_model() function and saves with pickle the best model found
    """
    best_score = -1
    best_model = None
    best_report = None
    sum_accuracy = 0
    sum_precision = 0
    sum_recall = 0


    idx = 0
    for i in range(ITER):
        print("Current iteration: " + str(i), end="\r")
        model, acc, prec, rec, report = train_model(X_train_iter[i], X_test_iter[i], y_train_iter[i], y_test_iter[i], model_name=model_name)
        if prec != 0:
            sum_accuracy += acc
            sum_precision += prec
            sum_recall += rec
            if acc > best_score:
                pickle.dump(model, open("best_model_" + model_name +".sav", "wb"))
                best_score = acc
                best_model = model
                best_report = report
            idx += 1
    sum_accuracy /= idx
    sum_precision /= idx
    sum_recall /= idx
    print("\n\n", model_name)
    print("Accuracy:", sum_accuracy)
    print("Precision:", sum_precision)
    print("Recall:", sum_recall)
    print(best_report)
    print("IDX:", idx)
    return best_model



if __name__ == "__main__":

    X_train_iter = []
    X_test_iter = []
    y_train_iter = []
    y_test_iter = []

    for i in range(ITER):
        X_train, X_test, y_train, y_test = create_dataset_iteration()
        X_train_iter.append(X_train)
        X_test_iter.append(X_test)
        y_train_iter.append(y_train)
        y_test_iter.append(y_test)

    best_model = search_for_best(model_name="MLP")
    
    best_model = search_for_best(model_name="RFC")
    
    best_model = search_for_best(model_name="ETC")

    best_model = search_for_best(model_name="SVM")