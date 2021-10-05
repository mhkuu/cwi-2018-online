from nltk.stem import PorterStemmer

from .abstract import AbstractModel

LIST_LIMIT = 1000  # Limit that we have on size of word list - we could think of making this a parameter even?


class LengthFreq(AbstractModel):

    def __init__(self, length_cutoff):
        self.length_cutoff = length_cutoff

        self.frequency = dict()
        with open('datasets/english/freq-bnc.txt', 'r') as f:
            for n, line in enumerate(f):
                if n == 0:
                    continue
                columns = line.strip('\n').split('\t')
                word = columns[1]
                freq = columns[3]
                # if len(word) >= LENGTH_CUTOFF:  # only include long words?!
                self.frequency[word] = int(freq)
                if len(self.frequency) == LIST_LIMIT:
                    break

        # We could also consider WordNetLemmatizer instead -- but we have a stemmer in production!
        self.stemmer = PorterStemmer()

    def train(self, train_set):
        pass

    def test(self, test_set):
        result = []
        for sent in test_set:
            result.append(self.calc(sent['target_word']))

        return result

    def calc(self, word):
        stemmed = self.stemmer.stem(word, to_lowercase=True)
        result = len(word) >= self.length_cutoff and stemmed not in self.frequency.keys()
        #if len(word) > LENGTH_CUTOFF:
        #    if word in self.frequency.keys():
        #        print(word)
        return str(int(result))
