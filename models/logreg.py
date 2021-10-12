from sklearn.linear_model import LogisticRegression

from models.abstract import AbstractModel
from models.freq import FrequencyCalculator
from models.ngram import NgramCalculator


class LogReg(AbstractModel):

    def __init__(self, language):
        self.language = language

        self.model = LogisticRegression()
        self.freq_calc = FrequencyCalculator(language)
        self.ngram_calc = NgramCalculator(language)

    def extract_features(self, word):
        length = len(word)
        starts_uppercase = int(word[0].isupper())
        freq = self.freq_calc.get_freq(word)
        ngram_prob = self.ngram_calc.full_ngram_prob(word)

        return [length, starts_uppercase, freq, ngram_prob]

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
