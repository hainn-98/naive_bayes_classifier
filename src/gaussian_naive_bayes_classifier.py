import os
import re
import math

import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from tabulate import tabulate
from matplotlib import pyplot as plt



class NaiveBayesClassifier:
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_data, self.test_data = train_test_split(self.dataset, test_size=0.3)
        self.spam_coef = 0
        self.ham_coef = 0
        self.count_vectorizer = CountVectorizer()
        self.spam_matrix = None
        self.ham_matrix = None
        self.mean_spam = None
        self.std_spam = None
        self.mean_ham = None
        self.std_ham = None
        
        

    def process_message(self, message):
        # Remove special characters, urls and numbers
        dummies_text = [r'&amp', r'&lt;\S+&gt;', r'&gt;', r'&lt;', r'http://\S+', r'\d+', r'[^A-Za-z0-9\s]+']
        for regex in dummies_text:
            message = re.sub(regex, ' ', message)
        # Remove stopwords
        words = set(stopwords.words('english'))
        message = ' '.join([word for word in message.split(' ') if word not in words])
        # Remove single letters
        message = ' '.join([word for word in message.split(' ') if len(word) > 1])
        # Convert message to lower case and strip whitespace
        message = ' '.join([word.lower().strip() for word in message.split(' ') if word])
        return message
        
        
    def calculate_probability(self,x, mean, std):
        exponent = -((x-mean)**2/(2*std**2))
        return math.log(1/(math.sqrt(2*math.pi)*std)) + exponent
        
    def __preprocess_data(self):
        pd.options.mode.chained_assignment = None
        self.train_data.loc[:, 'v2'] = self.train_data['v2'].map(lambda message: self.process_message(message))

    def train(self):
        self.__preprocess_data()
        message_counter = pd.value_counts(self.train_data['v1'])
        self.spam_coef = message_counter['spam'] / len(self.train_data)
        self.ham_coef = message_counter['ham'] / len(self.train_data)
        vectors = self.count_vectorizer.fit_transform(self.train_data['v2']).toarray()
        bag_of_words = self.count_vectorizer.get_feature_names()
        self.spam_matrix = np.zeros((len(vectors), len(bag_of_words)))
        self.ham_matrix = np.zeros((len(vectors), len(bag_of_words)))
        for i, _ in enumerate(bag_of_words):
            word_in_spam = 0
            word_in_ham = 0
            for idx, vector in enumerate(vectors):
                if vector[i] != 0:
                    if self.train_data['v1'].iloc[idx] == 'spam':
                        self.spam_matrix[idx][i] +=vector[i]
                    else:
                        self.ham_matrix[idx][i] +=vector[i]
        self.mean_spam=np.mean(self.spam_matrix, axis=0)
        self.std_spam = np.std(self.spam_matrix, axis=0)
        self.mean_ham = np.mean(self.ham_matrix, axis=0)
        self.std_ham = np.std(self.ham_matrix, axis=0)
        
    def classify(self, new_message):
        predict_spam = math.log(self.spam_coef)
        predict_ham = math.log(self.ham_coef)
        new_message = self.process_message(new_message)
        vecterized_message = self.count_vectorizer.transform([new_message]).toarray()[0]
        for idx, val in enumerate(vecterized_message):
            if val != 0:
                predict_ham += val*self.calculate_probability(val, self.mean_ham[idx], self.std_ham[idx])
                predict_spam +=  val*self.calculate_probability(val, self.mean_spam[idx], self.std_spam[idx])
        return True if predict_spam > predict_ham else False

    def predict(self):
        true_positive = true_negative = false_positive = false_negative = 0
        for v1, v2 in self.test_data.iloc:
            result = self.classify(v2)
            if result:
                if v1 == 'spam':
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if v1 == 'spam':
                    false_negative += 1
                else:
                    true_negative += 1
        print(tabulate({
            '': ['Actual: Spam', 'Actual: Ham'],
            'Predicted: Spam': [true_positive, false_positive],
            'Predicted: Ham': [false_negative, true_negative]
        }, headers='keys'))
        accuracy = round((true_negative + true_positive) / len(self.test_data), 4)
        print(f"Accuracy: {accuracy}")
        print(f"Total: {len(self.test_data)}")


if __name__ == '__main__':
    naive_bayes_classifier = NaiveBayesClassifier(dataset=pd.read_csv('../resources/spam.csv'))
    naive_bayes_classifier.train()
    naive_bayes_classifier.predict()

