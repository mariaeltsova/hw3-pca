import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from math import log
import random
# do not import any non-python native libraries because github tests might fail
# consult with TAs if you find them crucial for solving the exercise


def read_text_file(filename: str) -> str:
    """
    Read text file as a long string

    Arguments
    ---------
    filename: str
        Name of the input text file with '.txt' suffix.

    Return
    ------
    str
        A string of characters from the file
    """
    file = open(filename, "r", encoding='utf-8')
    #read whole file to a string
    string = file.read()
    #close file
    file.close()
    return string


def preprocess_text(text: str) -> list["str"]:
    """
    Preprocess text string by:
        1. removing any character that is not a letter (e.g. +*.,#!"$%&/'()=+? and so on ...)
        2. convert any uppercase letters into lowercase
        3. split a string into a list of words

        Caution: There should be no empty strings in the list of words.

    Arguments
    ---------
    text: str
        A string of text

    Return
    ------
    list['str']
        A list of words as strings in the same order as in the original document.
    """
    #print(text, "\n", "\n")
    text = "".join([ c if c.isalnum() else " " for c in text ])
    text = text.lower()
    return text.split()


def words_into_kmers(words: list["str"], k: int) -> dict:
    """
    Convert a list of words into a dictionary of {k-mer : number of k-mers in words} with k as a parameter.
    If a word is shorter than k characters, discard it.

    Arguments
    ---------
    words: list['str']
        A list of words as strings.
    k: int
        The length of k-mers.

    Return
    ------
    dict
        Dictionary with keys k-mers (string) and values number of occurances (int)
    """
    dict = {}
    
    for word in words:
        while k <= len(word):
            s = word[0:k]
            #check if dictionary contains
            if s in dict.keys():
                dict[s] += 1
            else:
                dict[s] = 1 
            word = word[1:]
    return dict


def words_into_bag_of_words(words) -> dict:
    """
    Convert a list of words into a dictonary of {word : number of words in text}.

    Arguments
    ---------
    words: list['str']
        A list of words as strings.

    Return
    ------
    dict
        Dictionary with keys words (string) and values number of occurances (int)
    """
    dict = {}
    for word in words:
        if word in dict.keys():
            dict[word] += 1
        else:
            dict[word] = 1
    return dict


def words_into_phrases(words: list["str"], phrase_encoding: list["int"]) -> dict["str"]:
    """
    Convert a list of words in to a dictonary of {phrase : number of phrases in text}.

    Phrase encoding is a list of integers where 1 means the word is in and 0 that it is
    not in the phrase. We encode a phrase by joining words in the phrase with "-" sign
    and use "/" to represent ommited word in the phrase.

    Example text 1: "it is raning man".
    Using encoding [1,1] we get three phrases ("it-is", "is-raning", "raining-man").
    Using encoding [1, 0, 1] we get only two phrases ("it-/-raining", "is-/-man").

    Example text 2: "we did not start the fire"
    Using encoding [1, 0, 1] we get four phrases ("we-/-not", "did-/-start", "not-/-the", "start-/-fire").
    Using encoding [0, 1, 0, 1] we get three phrases ("/-did-/-start", "/-not-/-the", "/-start-/-fire").

    As you can see, phrase encoding does not have to start or end with 1.

    Arguments
    ---------
    words: list['str']
        A list of words as strings.
    phrase_encoding: list['int']
        Phrases are consecutive words where 1 means a word is in the phrase and 0 that
        the word is not in the phrase. Example is above.

    Return
    ------
    dict
        Dictionary with keys phrase (string) and values number of occurances (int)
    """
    dict = {}
   # print(words)
        #sprehodimo se od 0 word-a do len(words) - len(phrase_encoding)
    for startIndex in range(0, len(words) - len(phrase_encoding) + 1):
        phrase = ""
        for curIndex in range(startIndex, startIndex + len(phrase_encoding) ):
            if(curIndex - startIndex != 0):
                phrase += "-"
            if(phrase_encoding[curIndex - startIndex] == 0):
                phrase += "/"
            else:
                phrase += words[curIndex]
        #print("startindex=", startIndex,  "phrase=", phrase)
        if phrase in dict.keys():
            dict[phrase] += 1
        else:
            dict[phrase] = 1

    #print(dict.keys())
    return dict


def term_frequency(encoding: dict) -> dict:
    """
    Calculate the frequency of each term in the encoding.

    Arguments
    ---------
    encoding: dict
        Dictonary with keys strings (kmers, words, phrases) and values (number of occurances).

    Return
    ------
    dict
        Dictonary with keys strings (kmers, words, phrases) and values (FREQUENCY of occurances in this document).
    """
    numOcc = sum(encoding.values())
   # print("sum = ", numOcc)
    for key in encoding.keys():
        encoding[key] = encoding[key] / numOcc
    return encoding


def inverse_document_frequency(documents: list["dict"]):
    """
    Calculate inverse document frequency (idf) of all terms in the encoding of a document.
    Use the corrected formula for the idf (lecture notes page 36):
        idf(t) = log(|D| / (1 + |{d : t in d}|)),
    where |D| is the number of documents and |{d : t in d}| is the number of documents with the term t.
    Use natrual logarithm.

    Arguments
    ---------
    documents: list['dict']
        List of encodings for all documents in the study.

    Return
    ------
    dict
        Dictonary with keys strings (kmers, words, phrases) and values (FREQUENCY of occurances in this document.
    """
   # print(documents)
   # print(len(documents))
   # print(documents[0].keys())
    D = len(documents)
    allTerms = list(documents[0].keys())
    for doc in documents:
        allTerms.extend(list(doc))
        allTerms = list(set(allTerms))
    
    #create a dictionary from list
    inv_freq_dict = dict.fromkeys(allTerms, 0)
    #count actual frequency
    for term in allTerms:
        count = 0
        for doc in documents:
            if term in doc.keys():
                count += 1
        idf= log(D / (1 + count))
        inv_freq_dict[term] = idf
    return inv_freq_dict


def tf_idf(encoding: dict, term_importance_idf: dict) -> dict:
    """
    Calculate term frequency - inverse document frequency (tf-idf) using precomputed idf (with your function)
    and term_frequency function you implemented above (use it in this function).

    The output should contain only the terms that are listed inside the term_importance_idf dictionary.
    If the term does not exist in the document, asign it a value 0.
    Filter terms AFTER you calculated term frequency.

    Arguments
    ---------
    encoding: dict
        Dictonary with keys strings (kmers, words, phrases) and values (frequency of occurances).
    term_importance_idf: dict
        Term importance as an output of inverse_document_frequency function.

    Return
    ------
    dict
        Dictonary with keys strings (kmers, words, phrases) and values (tf-idf value).
        Includes only keys from the IDF dictionary.
    """
    
    res = dict.fromkeys(term_importance_idf.keys(), 0)
    for term in res.keys():
        if term not in encoding.keys():
            res[term] = 0
        else:
            res[term] = term_importance_idf[term] * encoding[term]
    return res


def cosine_similarity(vektor_a: np.array, vektor_b: np.array) -> float:
    """
    Cosine similariy between vectors a and b.

    Arguments
    ---------
    vector_a, vector_b: np.array
        Vector of a document in the feature space

    Return
    ------
    float
        Cosine similarity
    """
    dist = ((vektor_a @ vektor_b) / (np.linalg.norm(vektor_a)*np.linalg.norm(vektor_b)))
    return dist


def jaccard_similarity(vektor_a, vektor_b) -> float:
    """
    Jaccard similarity

    Arguments
    ---------
    vector_a, vector_b: np.array
        Vector of a document in the feature space

    Return
    ------
    float
        Jaccard similarity
    """
    raise NotImplementedError()

class PCA:
    def __init__(
        self,
        n_components: int,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        rnd_seed: int = 0,
    ):
        assert (
            type(n_components) == int
        ), f"n_components is not of type int, but {type(n_components)}"
        assert (
            n_components > 0
        ), f"n_components has to be greater than 0, but found {n_components}"

        self.n_components = n_components
        self.eigenvectors = []
        self.eigenvalues = []

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.rnd_seed = rnd_seed
        self.mean = 0

    def fit(self, X: np.ndarray) -> None:
        lenOfS = X.shape[1]
        #get mean
        #print("mean =", np.average(X, axis=0))
        self.mean =np.average(X, axis=0)
        for i in range(lenOfS):
            S = np.cov(X.T)
            x = S[random.randint(0, lenOfS - 1)]
            u, lambda1 = self.potencna_metoda(S, x)
            p = np.dot(X, u)
            X = X - np.outer(np.dot(X, u), u)
            self.eigenvalues.append(lambda1); self.eigenvectors.append(u)


    def fit_dual_pca(self, X: np.ndarray) -> None:
        lenOfS = X.shape[0]
        self.mean =np.average(X, axis=0)
        N = X.shape[0]
        X_orig = np.copy(X)
        for i in range(lenOfS):
            S = np.cov(X)
            x = S[random.randint(0, lenOfS - 1)]
            u, lambda1 = self.potencna_metoda(S, x)
            p = np.dot(X.T, u)
            X = X - np.outer(u, np.dot(X.T, u))
            self.eigenvalues.append(lambda1); self.eigenvectors.append(u)
        Xd = (X_orig - self.mean).T
        #print("Xd.shape", Xd.shape)
        #print("self.eigenvectros.shape", np.array(self.eigenvectors).shape)
        S = np.array(self.eigenvalues) + 10e-15
        #print("Xd", Xd)
        Xd = np.nan_to_num(Xd)
        #print("np.diag(np.sqrt(1 / (S * (N - 1))) )", np.diag(np.sqrt(1 / (S * (N - 1))) ))
        d = np.nan_to_num(np.diag(np.sqrt(1 / (S * (N - 1))) ))
        #print("d", d)
       # print("self.eigenvectors", self.eigenvectors)
        self.eigenvectors = np.nan_to_num(np.array(self.eigenvectors))
        U =  Xd @ self.eigenvectors @ d
       # print("U = ", U)
        #U =  np.nan_to_num(U)
        
        self.eigenvectors = U


    def potencna_metoda(
        self, M: np.ndarray, vector: np.array, iteration: int = 0
    ) -> tuple:
        
        if iteration == self.max_iterations - 1:
            eigenvalue = vector.T @ M @ vector
            #print(vector, eigenvalue)
            return (vector, eigenvalue)
        vector = vector @ M
        
        #print(M)
        #vector = np.dot(vector, M)
        vector = vector / np.linalg.norm(vector)
        iteration += 1
        return self.potencna_metoda( M, vector, iteration)

    def transform(self, X: np.ndarray) -> np.ndarray:
        print("len(self.mean)", len(self.mean))
        print("X.shape", X.shape)
        return np.dot(X - self.mean, np.array(self.eigenvectors[0:self.n_components]).T)
    
    def transform_dual_pca(self, X: np.ndarray) -> np.ndarray:
   
        return np.dot((X - self.mean).T, np.array(self.eigenvectors)[:,:self.n_components])
       

    def get_explained_variance(self):
        return self.eigenvalues
        

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:     
        #data_original = np.dot(X, self.eigenvectors) + pca.mean_
        return np.dot(X, self.eigenvectors) + self.mean
    
    def get_n_eigenvector(self, n):
        return np.array(self.eigenvectors[n])
