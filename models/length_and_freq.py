from models.abstract import AbstractModel
from calculators.freq import FrequencyCalculator


class LengthFreq(AbstractModel):
    def __init__(self, language, length_cutoff, list_limit=None):
        self.language = language

        self.freq_calc = FrequencyCalculator(language, list_limit)
        self.length_cutoff = length_cutoff

    def train(self, train_set):
        pass

    def test(self, test_set):
        result = []
        for sent in test_set:
            result.append(self.calc(sent['target_word']))

        return result

    def calc(self, word):
        freq = self.freq_calc.get_freq(word)
        # is_uppercase = word[0].isupper() and self.language not in ['german']
        result = len(word) >= self.length_cutoff and freq == 1  # and not is_uppercase
        return str(int(result))
