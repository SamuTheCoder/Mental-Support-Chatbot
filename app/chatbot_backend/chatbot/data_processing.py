import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

#nltk.download('punkt_tab')

def tokenize(sentence):
    """
    Tokenize a sentence.
    Ex:
    sentence = "Hello, how are you?"
    words = ["Hello", ",", "how", "are", "you", "?"]
    """

    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stem a word.
    Ex:
    word = "running"
    stemmed_word = "run"
    """

    return stemmer.stem(word.lower())

def bag_of_words(sentence, words):
    """
    Create a bag of words from a list of words.
    Ex:
    sentence = "Hello, how are you?"
    words = ["Hello", "how", "are", "you", "bye", "Good", "morning"]
    bag = [1, 1, 1, 1, 0, 0, 0]
    """

    sentence_words = tokenize(sentence)
    ##sentence_words = [word for word in sentence_words if word not in nltk.corpus.stopwords.words('english')] #might cause issues
    sentence_words = [stem(word) for word in sentence_words]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, word in enumerate(words):
        if word in sentence_words:
            bag[idx] = 1

    return bag




