import sys
import pandas as pd
import numpy as np
from pre_processor import Preprocess
from semantic_features import SemanticFeatures
from machine_learning_processor import MachineLearningClassifier
from sklearn.metrics import precision_recall_fscore_support

"""

CODE SHEET :
ONE = SSVSM + WU-Palmer


EIGHT = Naive Bayes
NINE = Neural Network
"""

class MeasureSimilarity(object):
    def __init__(self, feature_code, classifier_code):
        print("hello")
        self.training_dataset_size = 140000
        self.testing_dataset_size = 2827
        self.feature_code = feature_code
        self.classifier_code = classifier_code
        print(" feature_code : ",self.feature_code)
        print(" classifier_code : ",self.classifier_code)
        self.data_setup()

    def data_setup(self):
        """ sets up data and call functions for feature generations and classifer"""
        # Make sure the dataset is download and put into the data folder
        training_data = pd.read_csv('./data/dbpedia_company.csv', sep=',', nrows=self.training_dataset_size)
        testing_data = pd.read_csv('./data/dbpediatest_company.csv', sep=',' , nrows=self.training_dataset_size)
        question_list1 = training_data['topic']
        question_list2 = training_data['contents']
        is_duplicate = training_data['label']
        question_listtest1 = testing_data['topic']
        question_listtest2 = testing_data['contents']
        is_duplicatetest = testing_data['label']
        # for will
        X = []
        Y = []
        for i in range(4990, 5010):
            print("*"*20, i ,"*"*20 )
            feature = self.call_feature_generator(question_list1[i],question_list2[i], self.feature_code )
            X.append(feature)
            Y.append(is_duplicate[i])
            print(feature)
            print(is_duplicate[i])
            print(question_list1[i])
            print(question_list2[i])


        classifer = self.call_classifier(X, Y, self.classifier_code)
        testX = []
        testY = []

        for i in range(99, 106):
            print("-"*20, i ,"-"*20 )
            feature = self.call_feature_generator(question_listtest1[i],question_listtest2[i], self.feature_code )
            testX.append(feature)
            testY.append(is_duplicatetest[i])

        X= np.array(testX).reshape(-1,1)
        calculate_y = classifer.predict(X)
        print(calculate_y)
        result = precision_recall_fscore_support(testY, calculate_y, labels=np.unique(calculate_y))
        print ("Precision: Class 1 - ", result[0][0], "% and Class 0 - ", result[0][1], "%")
        print ("Recall: Class 1 - ", result[1][0], "% and Class 0 - ", result[1][1], "%")
        print ("F-Score: Class 1 - ", result[2][0], "% and Class 0 - ", result[2][1], "%")
        
        
    def call_classifier(self, x, y, code):
        print(code)
        if code == "EIGHT":
            classifier = MachineLearningClassifier.nb_classifier(x, y)
            return classifier
        elif code == "NINE":
            classifier = MachineLearningClassifier.mlp_classifier(x, y)
            return classifier
        else:
            raise ValueError('Enter correct Code for classifier')


    def call_feature_generator(self, ques1, ques2, code):
        print(str(code))
        processer1 = Preprocess(ques1)
        lemma_ques1 = processer1.preprocess_with_lemma()
        processer2 = Preprocess(ques2)
        lemma_ques2 = processer2.preprocess_with_lemma()
        semantic_obj = SemanticFeatures()

        if code == "ONE":
            combined_similarity_score = semantic_obj.integ_wuptfidf(lemma_ques1, lemma_ques2)
            return combined_similarity_score
        else:
            raise ValueError('Enter correct Code for feature')


if __name__ == '__main__':
    feature_code = sys.argv[1]
    classifier_code = sys.argv[2]
    if feature_code is None:
        feature_code = "ONE"
    elif classifier_code is None:
        classifier_code = "EIGHT"
    my_obj = MeasureSimilarity(feature_code, classifier_code)
