FROM python:3.6

WORKDIR /simplex

RUN pip3 install scikit-learn==0.24.0
RUN pip3 install numpy
RUN pip3 install nltk
RUN pip3 install simpletransformers
RUN pip3 install PyDictionary
RUN pip3 install gensim
RUN pip3 install pyinflect
RUN pip3 install pattern
RUN pip3 install torch
RUN pip3 install python-Levenshtein
RUN pip3 install flask

RUN python -c 'import nltk; nltk.download("punkt")'
RUN python -c 'import nltk; nltk.download("averaged_perceptron_tagger")'

COPY . .

ENTRYPOINT ["python3", "server.py"]