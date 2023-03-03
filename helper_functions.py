import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

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
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()


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

    def fit(self, X: np.ndarray) -> None:
        """
        Fit principle component vectors.
        Center the data around zero.

        Arguments
        ---------
        X: np.ndarray
            Data matrix with shape (n_samples, n_features)
        """
        raise NotImplementedError()

    def potencna_metoda(
        self, M: np.ndarray, vector: np.array, iteration: int = 0
    ) -> tuple:
        """
        Perform the power method for calculating the eigenvector with the highest corresponding
        eigenvalue of the covariance matrix.
        This should be a recursive function. Use 'max_iterations' and 'tolerance' to terminate
        recursive call when necessary.

        Arguments
        ---------
        M: np.ndarray
            Covariance matrix of the zero centered data.
        vector: np.array
            Candidate eigenvector in the iteration.
        iteration: int
            Index of the consecutive iteration for termination purpose of the

        Return
        ------
        np.array
            The unit eigenvector of the covariance matrix.
        float
            The corresponding eigenvalue fo the covariance matrix.
        """
        raise NotImplementedError()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data (X) using fitted eigenvectors

        Arguments
        ---------
        X: np.ndarray
            New data with the same number of features as the fitting data.

        Return
        ------
        np.ndarray
            Transformed data with the shape (n_samples, n_components).
        """
        raise NotImplementedError()

    def get_explained_variance(self):
        """
        Return the explained variance ratio of the principle components.
        Prior to calling fit() function return None.
        Return only the ratio for the top 'n_components'.

        Return
        ------
        np.array
            Explained variance for the top 'n_components'.
        """
        raise NotImplementedError()

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data from the principle component space into
        the real space.

        Arguments
        ---------
        X: np.ndarray
            Data  in PC space with the same number of features as
            the fitting data.

        Return
        ------
        np.ndarray
            Transformed data in original space with
            the shape (n_samples, n_components).
        """
        raise NotImplementedError()
