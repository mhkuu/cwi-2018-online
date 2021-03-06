from abc import ABC

from nltk.stem import PorterStemmer

LIST_LIMIT = None  # If set, only include the n most frequent words


class FrequencyCalculator(object):
    def __init__(self, language, list_limit=LIST_LIMIT, use_opensubtitles=True):
        self.frequency = dict()
        self.stemmer = None

        if language == 'english' and not use_opensubtitles:
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
            # This alternative English frequency list is based on lemmata, so make sure to use a stemmer
            # We could also consider WordNetLemmatizer instead -- but we have a stemmer in production!
            self.stemmer = PorterStemmer()
        elif language == 'spanish' and not use_opensubtitles:
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
        else:
            with open('datasets/{}/freq-opensubtitles.txt'.format(language), 'r') as f:
                for n, line in enumerate(f):
                    columns = line.strip('\n').split()
                    word = columns[0]
                    freq = columns[1]
                    # if len(word) >= LENGTH_CUTOFF:  # only include long words?!
                    self.frequency[word] = int(freq)
                    if list_limit and len(self.frequency) == list_limit:
                        break

    def get_freq(self, word):
        normalized = self.stemmer.stem(word, to_lowercase=True) if self.stemmer else word.lower()
        return self.frequency.get(normalized, 1)
