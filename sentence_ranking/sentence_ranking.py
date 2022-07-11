import math
import numpy as np

from nltk import word_tokenize

ALPHA = math.pow(10, -10)

def perplexity(sentence, dictionary, total):
    """
    Compute the perplexity of a sentence given a language model

    Parameters:
    sentence (str): The sentence for which the perplexity is to be computed
    dictionary (dict): The unigram dictionary for the current corpus (see load_dictionary_unigrams)
    total (int): total number of words in the corpus

    Returns:
    float: the perplexity of the sentence
    """
    tokens = word_tokenize(sentence)
    tokens = list(filter(lambda x: any(c.isalpha() for c in x), tokens))
    tokens = list(map(lambda x: x.lower(), tokens))

    exponent = 0
    for token in tokens:
        if token in dictionary:
            exponent += math.log(dictionary[token][0] / total)
        else:
            exponent += math.log(ALPHA)
    exponent /= len(tokens)

    return math.pow(2, -exponent)


def perplexity_bigrams(sentence, dictionary_unigrams, dictionary_bigrams, total):
    """
    Compute the perplexity of a sentence given a language model

    Parameters:
    sentence (str): The sentence for which the perplexity is to be computed
    dictionary_unigrams (dict): The unigram dictionary for the current corpus (see load_dictionary_unigrams)
    dictionary_bigrams (dict): The bigram dictionary for the current corpus (see load_dictionary_bigrams)
    total (int): total number of words in the corpus

    Returns:
    float: the perplexity of the sentence
    """
    tokens = word_tokenize(sentence)
    tokens = list(filter(lambda x: any(c.isalpha() for c in x), tokens))
    tokens = list(map(lambda x: x.lower(), tokens))

    exponent = 0
    for i in range(len(tokens)):
        if i == 0:
            if tokens[i] in dictionary_unigrams:
                exponent += math.log(dictionary_unigrams[tokens[i]][0] / total)
            else:
                exponent += math.log(ALPHA)
        else:
            if (tokens[i-1], tokens[i]) in dictionary_bigrams and tokens[i-1] in dictionary_unigrams:
                exponent += math.log(dictionary_bigrams[(tokens[i-1], tokens[i])] / dictionary_unigrams[tokens[i-1]][0])
            else:
                exponent += math.log(ALPHA)
    exponent /= len(tokens)

    return math.pow(2, -exponent)


def debug_print(msg, debug=False):
    """
    Prints a debug message

    Parameters:
    msg (str): a message to be displayed
    debug (boolean): if the debug mode is enabled or not
    """
    if debug:
        print("[SENTENCE RANKING] " + msg)


def choose_synonym(tokens, counter, sentence, synonyms, dictionary_unigrams, dictonary_bigrams, total, debug=False, factor=0):
    """
    Function that chooses a synonym from a list based on the unigram and bigram perplexity of the sentences.
    Basically, the function replaces the target word in the sentence with every synonym, computes the
    perplexity, and then chooses the sentence with the smallest perplexity

    Parameters:
    tokens (list<str>): the list of tokens that make up the sentence
    counter (int): the index of the token that needs to be potentially replaced
    sentence (str): the sentence to be analyzed
    synonyms (list<(str, float)>): a list of touples containing the synonyms for a target word and their cosine similarities to the target word
    dictionary_unigrams (dict): a dictionary of unigrams from a language model
    total (int): the total number of words in the language model
    debug (bool): weather to print debug info to stdout or not
    factor (float): number between 0 and 1 that indicates the weight that the bigram model has in ranking the sentences

    Returns:
    str: the modified sentence according to the described behaviour
    """
    if factor != 0:
        best_perplexity = (1 - factor) * perplexity(sentence, dictionary_unigrams, total) + \
                        factor * perplexity_bigrams(sentence, dictionary_unigrams, dictonary_bigrams, total)
    else:
        best_perplexity = perplexity(sentence, dictionary_unigrams, total)

    debug_print("Original sentence: " + sentence + " has perplexity " + str(best_perplexity), debug=debug)
    for syn, sim in synonyms:
        new_sentence = ""
        for i in range(len(tokens)):
            if i != 0:
                new_sentence += " "
            if i == counter:
                new_sentence += syn
            else:
                new_sentence += tokens[i][0]

        current_perplexity = 0
        if factor != 0:
            current_perplexity = (1 - factor) * perplexity(new_sentence, dictionary_unigrams, total) + \
                                factor * perplexity_bigrams(new_sentence, dictionary_unigrams, dictonary_bigrams, total)
        else:
            current_perplexity = perplexity(new_sentence, dictionary_unigrams, total)
        debug_print("Candidate sentence: " + new_sentence + " has perplexity " + str(current_perplexity), debug=debug)

        if current_perplexity < best_perplexity:
            best_perplexity = current_perplexity
            sentence = new_sentence
            tokens[counter] = (syn, tokens[counter][1])

    return sentence


def choose_synonym_transformers(tokens, counter, sentence, synonyms, model, debug=False):
    """
    Function that chooses a synonym from a list based on the similarity between the embeddings of the resulted sentences.
    The embeddings are generated by a transformer, using the simple transformers library
    Basically, for each synonym, a sentence with the target word replaced by that synonym is generated, then the sentence
    with the highest cosine similarity with the target sentence is chosen as being the best

    Parameters:
    tokens (list<str>): the list of tokens that make up the sentence
    counter (int): the index of the token that needs to be potentially replaced
    sentence (str): the sentence to be analyzed
    synonyms (list<(str, float)>): a list of touples containing the synonyms for a target word and their cosine similarities to the target word
    model (transformer object): an object which represents a transformer model
    debug (bool): weather to print debug info to stdout or not

    Returns:
    str: the modified sentence according to the described behaviour
    """
    debug_print("Original sentence: " + sentence, debug=debug)
    sentences = [sentence]
    for syn, sim in synonyms:
        new_sentence = ""
        for i in range(len(tokens)):
            if i != 0:
                new_sentence += " "
            if i == counter:
                new_sentence += syn
            else:
                new_sentence += tokens[i][0]
        
        sentences.append(new_sentence)

    vectors = model.encode_sentences(sentences, combine_strategy="mean")

    best_similarity = -float("inf")
    best_index = 0
    for i in range(1, len(sentences)):
        similarity = np.dot(vectors[0], vectors[i])/(np.linalg.norm(vectors[0])*np.linalg.norm(vectors[i]))
        debug_print("Candidate sentence: " + sentences[i] + " has similarity " + str(similarity), debug=debug)
        if similarity > best_similarity:
            best_similarity = similarity
            best_index = i
            tokens[counter] = (synonyms[i-1][0], tokens[counter][1])

    return sentences[best_index]
    