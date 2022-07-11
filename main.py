import pickle
import argparse

from copy import deepcopy
from nltk import word_tokenize, pos_tag
from simpletransformers.language_representation import RepresentationModel

from complexity_prediction.complexity_prediction import load_dictionary_unigrams, load_dictionary_bigrams, predict
from synonyms_generation.synonyms_generation import get_synonyms, get_synonyms_transformers
from sentence_ranking.sentence_ranking import choose_synonym, choose_synonym_transformers
from tester.tester import test

DEBUG = False

def debug_print(msg):
    """
    Prints a debug string

    Parameters:
    msg (str): the string to be printed
    """
    if DEBUG:
        print("[MAIN MODULE] " + msg)


def simplify(sentence, dictionary_unigrams, total, dictionary_bigrams, complexity_pred_model, 
             path="perplexity", bigram_factor=0, transformer_model=None):
    """
    Simplifies a given sentence

    Parameters:
    sentence (str): the sentence to be simplified
    dictionary_unigrams (dict): The unigram dictionary for the current corpus (see load_dictionary_unigrams)
    total (int): the total number of words in the language model
    dictionary_bigrams (dict): The bigram dictionary for the current corpus (see load_dictionary_bigrams)
    complexity_pred_model (scikit-learn model): the complexity prediction model
    path (str): the path for the simplification strategy
    bigram_factor (float): number between 0 and 1 that indicates the weight that the bigram model has in ranking the sentences
    transformer_model (object): the model for the transformer

    Returns:
    touple<str, list, list>: a touple consisting of the simplified sentence, the original tokens and the new tokens
    """
    replaceble_pos = ["JJ", "JJR", "JJS", "NN", "NNS", "RB",
                      "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

    sentence = sentence.rstrip()

    tokens = pos_tag(word_tokenize(sentence))
    old_tokens = deepcopy(tokens)

    counter = -1
    for (token, pos) in tokens:
        counter += 1
        if pos not in replaceble_pos:
            continue
        
        # Stage 1: identify the complex words that are candidates for replacement
        if predict(complexity_pred_model, dictionary_unigrams, total, token.lower()) == 1:
            debug_print("Complex word detected " + token)
            # Stage 2: get the synonyms for the given difficult word
            synonyms = []
            if path == 'transformers':
                synonyms = get_synonyms_transformers(token, pos, complexity_pred_model, dictionary_unigrams, total, debug=DEBUG)
            else:
                synonyms = get_synonyms(token, pos, "synonyms_generation/word2vec_model.sav", debug=DEBUG)
            debug_print("Selected synonyms are: " + str(synonyms))

            if len(synonyms) == 0:
                continue
            # Stage 3: modify the sentence according to the perplexity or the transformer similaritiy metric
            if path == 'transformers':
                sentence = choose_synonym_transformers(tokens, counter, sentence, synonyms, transformer_model, debug=DEBUG)
            else:
                sentence = choose_synonym(tokens, counter, sentence, synonyms, 
                                            dictionary_unigrams, dictionary_bigrams, 
                                            total, debug=DEBUG, factor=bigram_factor)

    # check for indefinite article discrepancies
    for i in range(len(tokens) - 1):
        if tokens[i][0] == "a" and tokens[i+1][0][0] in "aeiou":
            tokens[i] = ("an", 0)
        elif tokens[i][0] == "an" and tokens[i+1][0][0] not in "aeiou":
            tokens[i] = ("a", 0)
    
    sentence = ""
    for token in tokens:
        sentence += token[0] + " "
        
    return sentence, old_tokens, tokens


def request_simplification(sentence, path, bigram_factor, transformer):
    """
    Request a simplification from the server with custom parameters

    Parameters:
    sentence (str): the sentence to be simplified
    path (str): the simplification path to be chosen (transformers or perplexity)
    bigram_factor (float): number between 0 and 1 that indicates the weight that the bigram model has in ranking the sentences
    transformer (str): a transformer type (bert, roberta or gpt2)

    Returns:
    touple<list, list>: a touple consisting of the original tokens and the new tokens
    """
    transformers = {"bert": "bert-base-uncased",
                    "roberta": "roberta-base",
                    "gpt2": "gpt2"}

    dictionary_unigrams, total = load_dictionary_unigrams("complexity_prediction/datasets/dictionary_unigrams")

    dictionary_bigrams = None
    if path == "perplexity" and bigram_factor > 0:
        dictionary_bigrams = load_dictionary_bigrams("complexity_prediction/datasets/dictionary_bigrams")

    complexity_pred_model = pickle.load(open("complexity_prediction/best_model.sav", "rb"))

    transformer_model = None
    if path == "transformers":
        transformer_model = RepresentationModel(model_type=transformer, model_name=transformers[transformer], use_cuda=False)

    _, old_tokens, new_tokens = simplify(sentence, dictionary_unigrams, total, dictionary_bigrams, complexity_pred_model, 
                                           path=path, bigram_factor=bigram_factor, transformer_model=transformer_model)

    return old_tokens, new_tokens


def main():
    """
    Main function of the lexical simplification module
    Reads the sentences from a file and simplifies them, dumping the results
    in a file called results.out
    """

    transformers = {"bert": "bert-base-uncased",
                    "roberta": "roberta-base",
                    "gpt2": "gpt2"}

    # setting up the parser
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=str, 
                        help="The file in which the sentences to be simplified are stored",)
    parser.add_argument("--sentence_ranking_strategy", type=str, choices=["perplexity", "transformers"],
                        help="The strategy for ranking sentences. Defaults to perplexity")
    parser.add_argument("--bigram_factor", type=float, 
                        help="The factor in which the bigram model counts to computing the perplexity.\
                        Default value is 0. Can be any float number between 0 and 1")
    parser.add_argument("--verbose", action='store_true',
                        help="If the program will show the decision process during simplification")
    parser.add_argument("--out_file", type=str,
                        help="The name of the file in which the results will be written. Defaults to results.out")
    parser.add_argument("--test", type=str,
                        help="The name of the reference file with which the testing will be done")
    parser.add_argument("--transformer", type=str, choices=["bert", "roberta", "gpt2"],
                        help="The transformer model to be used. Taken into consideration iff the path is transformers. \
                        Defaults to bert")

    args = parser.parse_args()

    global DEBUG
    if args.verbose:
        DEBUG = True
    else:
        DEBUG = False

    transformer = None
    if args.transformer:
        transformer = args.transformer
    else:
        transformer = "bert"

    debug_print("Loading unigram dictionary")
    dictionary_unigrams, total = load_dictionary_unigrams("complexity_prediction/datasets/dictionary_unigrams")
    debug_print("Successfully loaded unigram dictionary")

    dictionary_bigrams = None
    if args.sentence_ranking_strategy != "transformers" and args.bigram_factor:
        debug_print("Loading bigram dictionary")
        dictionary_bigrams = load_dictionary_bigrams("complexity_prediction/datasets/dictionary_bigrams")
        debug_print("Successfully loaded bigram dictionary")

    debug_print("Loading complexity prediction model")
    complexity_pred_model = pickle.load(open("complexity_prediction/best_model.sav", "rb"))
    debug_print("Successfully loaded complexity prediction model")

    transformer_model = None
    if args.sentence_ranking_strategy == "transformers":
        debug_print("Loading transformer model")
        transformer_model = RepresentationModel(model_type=transformer, model_name=transformers[transformer], use_cuda=False)
        debug_print("Successfully loaded the transformer model")

    res_file = None
    res_path = None
    if args.out_file:
        res_file = open(args.out_file, "w")
        res_path = args.out_file
    else:
        res_file = open("results.out", "w")
        res_path = "results.out"

    bigram_factor = 0
    if args.bigram_factor:
        bigram_factor = min(args.bigram_factor, 1)
        bigram_factor = max(bigram_factor, 0)

    sent_counter = 1
    input_file = open(args.input_file)
    for line in input_file:
        print("[MAIN MODULE] Sentence number " + str(sent_counter))
        res = simplify(line, dictionary_unigrams, total, dictionary_bigrams, complexity_pred_model,
                       path=args.sentence_ranking_strategy, bigram_factor=bigram_factor, 
                       transformer_model=transformer_model)[0]

        res_file.write(res + "\n")
        sent_counter += 1

    res_file.close()
    input_file.close()
    if args.test:
        test(args.test, res_path, args.input_file, dictionary_unigrams, total)

if __name__ == "__main__":
    main()    
