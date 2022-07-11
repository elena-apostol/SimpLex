# SimpLex: a lexical text simplification tool
Simplex is a text simplification architecture focusing on lexical text simplification using Machine Learning and Natural Language Processing techniques.

# Usage

## Prerequizites
In order to use the simplifier, the following Python3 libraries must be installed:
- `scikit learn`
- `numpy`
- `nltk`
- `simpletransformers`
- `PyDictionary`
- `gensim`
- `pyinflect`
- `pattern`

## Models
Please unzip all the archives in the individual directories.
After this is done, run the script called `setup.sh` located in the root of the project

## Running the simplifier manually
The simplifier can be run in the following manner:
```
usage: main.py [-h] [--sentence_ranking_strategy {perplexity,transformers}] [--bigram_factor BIGRAM_FACTOR] [--verbose] [--out_file OUT_FILE] [--test TEST] [--transformer {bert,roberta,gpt2}]
               input_file

positional arguments:
  input_file            The file in which the sentences to be simplified are stored

optional arguments:
  -h, --help            show this help message and exit
  --sentence_ranking_strategy {perplexity,transformers}
                        The strategy for ranking sentences. Defaults to perplexity
  --bigram_factor BIGRAM_FACTOR
                        The factor in which the bigram model counts to computing the perplexity. Can be any float number between 0 and 1. Defaults to 0
  --verbose             If the program will show the decision process during simplification
  --out_file OUT_FILE   The name of the file in which the results will be written. Defaults to results.out
  --test TEST           The name of the reference file with which the testing will be done
  --transformer {bert,roberta,gpt2}
                        The transformer model to be used. Taken into consideration iff the path is transformers. Defaults to bert
```
The input file must have exactly one sentence per line.

## Running the simplifier in server mode using docker
To run the simplifier in a containerized server mode, run `docker-compose up` in the root of the project. Once this is done, you can run the SimpLex GUI by running the `/gui/gui.py` python script. If tkinter errors occur, run `apt-get install python-tk`.
