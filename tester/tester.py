from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
from tester.SARI import SARIsent
from sentence_ranking.sentence_ranking import perplexity


def test(ref_path, out_path, input_path, dictionary, total):
    """
    Print a series of result metrics for a given simplification and a given reference file

    The metrics that will be shown are SARI, BLEU and the perplexity decrease

    Parameters:
    ref_path (str): path to the reference simplification
    out_path (str): path to the simplification output
    input_path (str): path to the input file
    dictionary (dict): a dictionary of unigrams from a language model
    total (int): the total number of words in the language model
    """
    counter = 1
    out_file = open(out_path, "r")
    input_file = open(input_path, "r")

    average_bleu = 0
    average_sari = 0
    average_decr_perpl = 0

    for line in open(ref_path, "r"):
        tokens_ref = word_tokenize(line)
        tokens_ref = list(filter(lambda x: any(c.isalpha() for c in x), tokens_ref))
        tokens_ref = list(map(lambda x: x.lower(), tokens_ref))

        out_line = out_file.readline()
        input_line = input_file.readline()

        tokens_out = word_tokenize(out_line)
        tokens_out = list(filter(lambda x: any(c.isalpha() for c in x), tokens_out))
        tokens_out = list(map(lambda x: x.lower(), tokens_out))

        bleu = sentence_bleu([tokens_ref], tokens_out)
        sari = SARIsent(input_line, out_line, [line])
        old_perpl = perplexity(input_line, dictionary, total)
        new_perpl = perplexity(out_line, dictionary, total)

        print("[TESTER] BLEU score for sentence " + str(counter) + " is " + str(bleu))
        print("[TESTER] SARI score for sentence " + str(counter) + " is " + str(sari))
        print("[TESTER] Initial perplexity for sentece " + str(counter) + " is " + str(old_perpl))
        print("[TESTER] New perplexity for sentence " + str(counter) + " is " + str(new_perpl))

        average_bleu += bleu
        average_sari += sari
        average_decr_perpl += (1 - new_perpl / old_perpl)

        counter += 1

    average_bleu /= (counter - 1)
    average_sari /= (counter - 1)
    average_decr_perpl /= (counter - 1)

    print("[TESTER] Average BLEU score is " + str(average_bleu))
    print("[TESTER] Average SARI score is " + str(average_sari))
    print("[TESTER] Average perplexity decrese is " + str(average_decr_perpl))
    