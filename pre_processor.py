from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import numpy as np
from utils.wordnet_pos import GetWordnetPos
import string

class Preprocess(object):
    def __init__(self, sentence):
        self.sentence = sentence

    def get_tokens(self):
        tokens = word_tokenize(str(self.sentence))
        return tokens
    def get_pos_tag(self, tokens):
        pos1 = pos_tag(tokens)
        pos1 = [(element[0].lower(),element[1]) for element in pos1]
        return pos1

    def remove_stop_words(self,**kwargs):
        stop_words = stopwords.words('english') + list(string.punctuation)
        final_sentence = []
        pos = kwargs.get("pos")
        tokens = kwargs.get("tokens")
        if pos is None:
            for item in tokens:
                if item not in stop_words:
                    final_sentence.append(item)
        else:
            for item in pos:
                if item[0] not in stop_words:
                    final_sentence.append(item)

        return final_sentence

    def get_lemmaization(self, final_sentence):
        lemmatizer = WordNetLemmatizer()
        lemma = []
        for item in final_sentence:
            (word, pos_tag_name) = item
            lemmaizer = GetWordnetPos(pos_tag_name)
            wordnet_tag = lemmaizer.get_wordnet_pos()
            if wordnet_tag is None:
                lemma.append(lemmatizer.lemmatize(word))
            else:
                lemma.append(lemmatizer.lemmatize(word, pos=wordnet_tag))
        return lemma

    def preprocess_without_lemma(self):
        tokens = self.get_tokens()
        tokens = dict(tokens=tokens)
        final_sentence = self.remove_stop_words(**tokens)
        return final_sentence

    def preprocess_with_lemma(self):
        tokens = self.get_tokens()
        pos1 = self.get_pos_tag(tokens)
        pos = dict(pos=pos1)
        final_sentence = self.remove_stop_words(**pos)
        
        res_lemma = self.get_lemmaization(final_sentence)

        return res_lemma
