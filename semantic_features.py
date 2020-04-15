import numpy as np 
import pandas as pd 
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from utils.lesk_algorithm import Lesk
from utils.semantic_similarity_measures import SemanticMeasures
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

class SemanticFeatures(object):

    @staticmethod
    def get_lesk(ques):
        """ get each word meaning out of the given question"""
        lesk_obj = Lesk(ques)
        sentence_means = []
        for word in ques:
            sentence_means.append(lesk_obj.lesk(word, ques))
        return sentence_means

    

    @staticmethod
    def compute_wup_similarity(sentence_means1, sentence_means2):
        """ get the wup similarity score for two sentences"""
        score = SemanticMeasures.computeWup(sentence_means1, sentence_means2)

        return score

    
    @staticmethod
    def integ_wuptfidf(tokenList1, tokenList2):
        sentence_means1 = SemanticFeatures.get_lesk(tokenList1)
        sentence_means2 = SemanticFeatures.get_lesk(tokenList2)

        R1 = SemanticFeatures.compute_wup_similarity(sentence_means1, sentence_means2)
        
        bowList1 = {}
        bowList2 = {}
        for i in range(0, len(tokenList1)):
            if tokenList1[i] in bowList1:
                bowList1[tokenList1[i]] += 1
            else:
                bowList1[tokenList1[i]] = 1
        for i in range(0, len(tokenList2)):
            if tokenList2[i] in bowList2:
                bowList2[tokenList2[i]] += 1
            else:
                bowList2[tokenList2[i]] = 1

        x = list(bowList1.values())
        y = list(bowList2.values())
        if len(x) == 0:
            return 0.0
        if len(x) > len(y):
            y.extend([0 for _ in range(0, len(x) - len(y))])
        elif len(y) > len(x):
            x.extend([0 for _ in range(0, len(y) - len(x))])
        x = np.array(x).reshape(1,-1)
        y = np.array(y).reshape(1,-1)
        
        score1 = SemanticMeasures.overallSim(sentence_means1, sentence_means2, R1)*x
        score2 = SemanticMeasures.overallSim(sentence_means1, sentence_means2, R1)*y
        #return cosine_similarity(score1, score2)[0][0]
        return distance.cosine(score1, score2)
        
