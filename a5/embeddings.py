import math
import numpy as np

class Embeddings:

    def __init__(self, glove_file = '/projects/e31408/data/a5/glove_top50k_50d.txt'):
        self.embeddings = {}
        self.word_rank = {}
        for idx, line in enumerate(open(glove_file)):
            row = line.split()
            word = row[0]
            vals = np.array([float(x) for x in row[1:]])
            self.embeddings[word] = vals
            self.word_rank[word] = idx + 1

    def __getitem__(self, word):
        return self.embeddings[word]

    def __contains__(self, word):
        return word in self.embeddings

    def vector_norm(self, vec):
        """
        Calculate the vector norm (aka length) of a vector.

        This is given in SLP Ch. 6, equation 6.8. For more information:
        https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm

        Parameters
        ----------
        vec : np.array
            An embedding vector.

        Returns
        -------
        float
            The length (L2 norm, Euclidean norm) of the input vector.
        """
        # >>> YOUR ANSWER HERE
        norm = float(0)
        for v in vec: norm += pow(v,2)
        return math.sqrt(norm)
        # >>> END YOUR ANSWER

    def cosine_similarity(self, v1, v2):
        """
        Calculate cosine similarity between v1 and v2; these could be
        either words or numpy vectors.

        If either or both are words (e.g., type(v#) == str), replace them 
        with their corresponding numpy vectors before calculating similarity.

        Parameters
        ----------
        v1, v2 : str or np.array
            The words or vectors for which to calculate similarity.

        Returns
        -------
        float
            The cosine similarity between v1 and v2.
        """
        # >>> YOUR ANSWER HERE

        #changing v1 and v2 to their respective word vectors if they are words
        if type(v1) == str: self.__getitem__(v1)
        if type(v2) == str: self.__getitem__(v2)

        nominator = float(0)
        mult = v1 * v2
        for v in mult: nominator += v

        denominator = self.vector_norm(v1) * self.vector_norm(v2)

        return nominator / denominator
        # >>> END YOUR ANSWER

    def most_similar(self, vec, n = 5, exclude = []):
        """
        Return the most similar words to `vec` and their similarities. 
        As in the cosine similarity function, allow words or embeddings as input.


        Parameters
        ----------
        vec : str or np.array
            Input to calculate similarity against.

        n : int
            Number of results to return. Defaults to 5.

        exclude : list of str
            Do not include any words in this list in what you return.

        Returns
        -------
        list of ('word', similarity_score) tuples
            The top n results.        
        """
        # >>> YOUR ANSWER HERE
        
        #changing vec to its word vector if it is a word
        if type(vec) == str: self.__getitem__(vec)

        #looping through the dictionary of word vectors
        top_words = []
        for key in self.embeddings:
            top_words.append((key,self.cosine_similarity(self.embeddings[key], vec)))
        
        #sorting top_words
        def look_at_the_cosine_similarity(x): return x[1] 
        top_words.sort(reverse = True, key = look_at_the_cosine_similarity)

        #returning only the top n words
        top_n_words = []
        for i in range(n+1): top_n_words.append(top_words[i+1])
        return top_n_words
        # >>> END YOUR ANSWER

if __name__ == '__main__':
    embeddings = Embeddings()
    word = 'lemon'
    print(f'Most similar to {word}:')
    for item in embeddings.most_similar(word, exclude=[word]):
        print('\t',item[0], '\t', item[1])
