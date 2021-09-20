import csv, string
import numpy as np
from scipy.stats import spearmanr
from embeddings import Embeddings
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import math



def read_sts(infile = 'data/sts-dev.csv'):
    sts = {}
    for row in csv.reader(open(infile), delimiter='\t'):
        if len(row) < 7: continue
        val = float(row[4])
        s1, s2 = row[5], row[6]
        sts[s1, s2] = val / 5.0
    return sts

def calculate_sentence_embedding(embeddings, sent, weighted = False):
    """
    Calculate a sentence embedding vector.

    If weighted is False, this is the elementwise sum of the constituent word vectors.
    If weighted is True, multiply each vector by a scalar calculated
    by taking the log of its word_rank. The word_rank value is available
    via a dictionary on the Embeddings class, e.g.:
       embeddings.word_rank['the'] # returns 1

    In either case, tokenize the sentence with the `word_tokenize` function,
    lowercase the tokens, and ignore any words for which we don't have word vectors. 

    Parameters
    ----------
    sent : str
        A sentence for which to calculate an embedding.

    weighted : bool
        Whether or not to use word_rank weighting.

    Returns
    -------
    np.array of floats
        Embedding vector for the sentence.
    
    """
    # >>> YOUR ANSWER HERE

    # tokenizing the sentence
    tokens = word_tokenize(sent.lower())

    if weighted:
        #initializing sentence_vector
        sentence_vector = 0

        #for each token, add the corresponding word vector times its word rank to the current sentence_vector (elementwise sum)
        for t in tokens:
            sentence_vector += embeddings.__getitem__(t) * math.log(embeddings.word_rank[t])
        return sentence_vector

    else:
        #initializing sentence_vector
        sentence_vector = 0

        #for each token, add the corresponding word vector to the current sentence_vector (elementwise sum)
        for t in tokens:
            sentence_vector += embeddings.__getitem__(t)
        return sentence_vector
    # >>> END YOUR ANSWER



def score_sentence_dataset(embeddings, dataset, weighted = False):
    """
    Calculate the correlation between human judgments of sentence similarity
    and the scores given by using sentence embeddings.

    Parameters
    ----------
    dataset : dictionary of the form { (sentence, sentence) : similarity_value }
        Dataset of sentence pairs and human similarity judgments.
    
    weighted : bool
        Whether or not to use word_rank weighting.

    Returns
    -------
    float
        The Spearman's Rho ranked correlation coefficient between
        the sentence emedding similarities and the human judgments.     
    """
    # >>> YOUR ANSWER HERE
    model_values = []
    golden_values = []

    #looping through the keys of the dictionary "dataset" to get a list of golden values and a list of model generated values
    for key in dataset:
        w1 = key[0]
        w2 = key[1]
        
        #getting sentence vectors
        w1_vector = calculate_sentence_embedding(embeddings,w1,weighted)
        w2_vector = calculate_sentence_embedding(embeddings,w2,weighted)

        #calculating sentence similarity using the word vectors
        model_values.append(embeddings.cosine_similarity(w1_vector, w2_vector))
        #getting the human scores of sentence similarity
        golden_values.append(dataset[key])

    return spearmanr(golden_values, model_values)[0]
    # >>> END YOUR ANSWER

if __name__ == '__main__':
    embeddings = Embeddings()
    sts = read_sts()
    
    print('STS-B score without weighting:', score_sentence_dataset(embeddings, sts))
    print('STS-B score with weighting:', score_sentence_dataset(embeddings, sts, True))
