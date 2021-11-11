from abc import ABC

from nltk.stem import PorterStemmer

LIST_LIMIT = 1000  # Limit that we have on size of word list


class FrequencyCalculator(object):
    def __init__(self, language, list_limit=LIST_LIMIT):
        self.frequency = dict()

        if language == 'english':
            with open('datasets/{}/freq-bnc.txt'.format(language), 'r') as f:
                for n, line in enumerate(f):
                    if n == 0:
                        continue
                    columns = line.strip('\n').split('\t')
                    word = columns[1]
                    freq = columns[3]
                    # if len(word) >= LENGTH_CUTOFF:  # only include long words?!
                    self.frequency[word] = int(freq)
                    if list_limit and len(self.frequency) == list_limit:
                        break

        if language == 'spanish':
            with open('datasets/{}/freq-crea.txt'.format(language), 'r') as f:
                for n, line in enumerate(f):
                    if n == 0:
                        continue
                    columns = line.strip('\n').split('\t')
                    word = columns[1]
                    freq = columns[2].strip().replace(',', '')
                    # if len(word) >= LENGTH_CUTOFF:  # only include long words?!
                    self.frequency[word] = int(freq)
                    if list_limit and len(self.frequency) == list_limit:
                        break

        # The English frequency list is based on lemmata, so make sure to use a stemmer
        # We could also consider WordNetLemmatizer instead -- but we have a stemmer in production!
        self.stemmer = PorterStemmer() if language == 'english' else None

    def get_freq(self, word):
        stemmed = self.stemmer.stem(word, to_lowercase=True) if self.stemmer else word.lower()
        return self.frequency.get(stemmed, 1)
