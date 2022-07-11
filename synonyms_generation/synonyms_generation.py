import pickle

from PyDictionary import PyDictionary
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from pyinflect import getInflection
from pattern.text.en import superlative, pluralize, comparative
from complexity_prediction.complexity_prediction import predict


def create_model(path, limit):
    """
    Store a gensim object of the current word2vec embeddings using pickle; the file will be named word2vec_model.sav

    Parameters:
    path (str): path to the word2vec pre-trained model
    limit (int): the number of embeddings to be saved in the gensim model
    """
    model = KeyedVectors.load_word2vec_format(path, limit=limit)
    pickle.dump(model, open("word2vec_model.sav", "wb"))


def tagConversion(tag):
    """
    Convert a complex nltk POS tag to a simpler wornet POS tag

    Parameters:
    tag (str): a nltk POS tag

    Returns:
    str: a corresponding wordnet POS tag
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV


def debug_print(msg, debug=False):
    """
    Prints a debug message

    Parameters:
    msg (str): a message to be displayed
    debug (boolean): if the debug mode is enabled or not
    """
    if debug:
        print("[SYNONYMS GENERATION] " + msg)


def get_synonyms(word, nltk_pos, path, debug=False):
    """
    Get a list of synonyms for a given word and its corresponding POS

    The function selects only those synonyms that have a cosine similarity above the average similarity of all
    found synonyms; also, the function performs morphological changes to the words to preserve gramatical corectness.

    Parameters:
    word (str): the word for which the synonyms must be found
    nltk_pos (str): the nltk POS tag for the word given
    path (str): path to the word2vec saved model
    debug (boolean) (default:False): if the debug mode is enabled or not

    Returns:
    list: a list of touples formed by the synonym and its cosine similarity to the original word
    """
    dictionary = PyDictionary()
    model = pickle.load(open(path, "rb"))

    lemmatizer = WordNetLemmatizer()
    word = lemmatizer.lemmatize(word, pos=tagConversion(nltk_pos))

    synonyms = dictionary.synonym(word)

    if synonyms == None:
        return []

    # loop through the synonyms and compute the cosine similarities
    selected_synonyms = []
    average_similarity = 0
    for syn in synonyms:
        if syn in model.key_to_index and word in model.key_to_index:
            similarity = model.similarity(word, syn)
            debug_print("Similarity between " + word + " and " + syn + " is " + str(similarity), debug=debug)
            selected_synonyms.append((syn, similarity))
            average_similarity += similarity
        else:
            debug_print("Similarity between " + word + " and " + syn + " is unknown", debug=debug)

    if word not in model.key_to_index:
        for syn in synonyms:
            selected_synonyms.append((syn, 0))

    if len(selected_synonyms) == 0:
        return []
    
    average_similarity /= len(selected_synonyms)
    selected_synonyms = list(filter(lambda x: x[1] >= average_similarity, selected_synonyms))

    # do morphological changes to the synonyms
    for i in range(len(selected_synonyms)):
        if nltk_pos in ["VBD", "VBP", "VBZ", "VBG", "VBN"]:
            inflection = getInflection(selected_synonyms[i][0], nltk_pos)
            if inflection != None:
                selected_synonyms[i] = (inflection[0], selected_synonyms[i][1])
        elif nltk_pos in ["JJS", "RBS"]:
            selected_synonyms[i] = (superlative(selected_synonyms[i][0]), selected_synonyms[i][1])
        elif nltk_pos in ["NNS"]:
            selected_synonyms[i] = (pluralize(selected_synonyms[i][0]), selected_synonyms[i][1])
        elif nltk_pos in ["JJR", "RBR"]:
            selected_synonyms[i] = (comparative(selected_synonyms[i][0]), selected_synonyms[i][1])

    return selected_synonyms


def get_synonyms_transformers(word, nltk_pos, model, dictionary_unigrams, total, debug=False):
    """
    Get a list of synonyms for a given word and its corresponding POS

    This function shall be used in the transformer pipeline to determine which synonyms remain as candidates
    for the sentence ranking stage. This function will select only those synonyms that are predicted by the
    complexity prediction model as being easy words

    Parameters:
    word (str): the word for which the synonyms must be found
    nltk_pos (str): the nltk POS tag for the word given
    model (scikit-learn): the complexity prediction model to be used
    dictionary_unigrams (dict): a dictionary of unigrams from a language model
    total (int): the total number of words in the language model
    debug (bool): weather or not to print informations to stdout

    Returns:
    list: a list of touples formed by the synonym and the value 0 (the second value is for compatibility purposes and shall be ignored)
    """
    dictionary = PyDictionary()

    lemmatizer = WordNetLemmatizer()
    word = lemmatizer.lemmatize(word, pos=tagConversion(nltk_pos))
    
    synonyms = dictionary.synonym(word)

    if synonyms == None:
        return []

    selected_synonyms = []
    for syn in synonyms:
        tokens = syn.split(" ")
        simple = True
        for token in tokens:
            if predict(model, dictionary_unigrams, total, token) == 1:
                simple = False
                break
        if simple:
            debug_print("Synonym " + syn + " is a simple word", debug=debug)
            selected_synonyms.append((syn, 0))
        else:
            debug_print("Synonym " + syn + " is not a simple word", debug=debug)
    
    for i in range(len(selected_synonyms)):
        if nltk_pos in ["VBD", "VBP", "VBZ", "VBG", "VBN"]:
            inflection = getInflection(selected_synonyms[i][0], nltk_pos)
            if inflection != None:
                selected_synonyms[i] = (inflection[0], selected_synonyms[i][1])
        elif nltk_pos in ["JJS", "RBS"]:
            selected_synonyms[i] = (superlative(selected_synonyms[i][0]), selected_synonyms[i][1])
        elif nltk_pos in ["NNS"]:
            selected_synonyms[i] = (pluralize(selected_synonyms[i][0]), selected_synonyms[i][1])
        elif nltk_pos in ["JJR", "RBR"]:
            selected_synonyms[i] = (comparative(selected_synonyms[i][0]), selected_synonyms[i][1])

    return selected_synonyms
