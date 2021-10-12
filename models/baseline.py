from sklearn.linear_model import LogisticRegression

from .abstract import AbstractModel


class Baseline(AbstractModel):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = LogisticRegression()

    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length  # normalized word length
        len_tokens = len(word.split(' '))  # dealing with multiple words per phrase

        return [len_chars, len_tokens]

    def train(self, train_set):
        X = []
        y = []
        for sent in train_set:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])

        self.model.fit(X, y)

    def test(self, test_set):
        X = []
        for sent in test_set:
            X.append(self.extract_features(sent['target_word']))

        return self.model.predict(X)