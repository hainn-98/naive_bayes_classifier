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


def smoothing(a, b, c):
    return (a + 1) / (b + c)


class NaiveBayesClassifier:
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_data, self.test_data = train_test_split(self.dataset, test_size=0.3)
        self.spam_coef = 0
        self.ham_coef = 0
        self.count_vectorizer = CountVectorizer()
        self.bayes_matrix = None

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

    def __preprocess_data(self):
        pd.options.mode.chained_assignment = None
        self.train_data.loc[:, 'v2'] = self.train_data['v2'].map(lambda message: self.process_message(message))

    def train(self):
        self.__preprocess_data()
        message_counter = pd.value_counts(self.train_data['v1'])
        self.spam_coef = message_counter['spam'] / len(self.train_data)
        self.ham_coef = message_counter['ham'] / len(self.train_data)
        num_spam_words = sum(len(line.split(' ')) for line in self.train_data[self.train_data['v1'] == 'spam']['v2'])
        num_ham_words = sum(len(line.split(' ')) for line in self.train_data[self.train_data['v1'] == 'ham']['v2'])
        vectors = self.count_vectorizer.fit_transform(self.train_data['v2']).toarray()
        bag_of_words = self.count_vectorizer.get_feature_names()
        self.bayes_matrix = np.zeros((2, len(bag_of_words)))
        for i, _ in enumerate(bag_of_words):
            word_in_spam = 0
            word_in_ham = 0
            for idx, vector in enumerate(vectors):
                if vector[i] != 0:
                    if self.train_data['v1'].iloc[idx] == 'spam':
                        word_in_spam += vector[i]
                    else:
                        word_in_ham += vector[i]
            self.bayes_matrix[0][i] = smoothing(word_in_ham, num_ham_words, len(bag_of_words))
            self.bayes_matrix[1][i] = smoothing(word_in_spam, num_spam_words, len(bag_of_words))

    def classify(self, new_message):
        predict_spam = math.log(self.spam_coef)
        predict_ham = math.log(self.ham_coef)
        new_message = self.process_message(new_message)
        vecterized_message = self.count_vectorizer.transform([new_message]).toarray()[0]
        for idx, val in enumerate(vecterized_message):
            if val != 0:
                predict_ham += val * math.log(self.bayes_matrix[0][idx])
                predict_spam += val * math.log(self.bayes_matrix[1][idx])
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

