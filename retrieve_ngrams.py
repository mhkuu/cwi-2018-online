from collections import Counter

from nltk import word_tokenize, ngrams

from utils.dataset import Dataset

if __name__ == '__main__':
    data = Dataset('spanish')
    results = []
    sentences = set()
    for line in data.trainset:
        s = line['sentence']
        if s in sentences:
            continue
        sentences.add(s)
        for w in word_tokenize(s):
            w = w.lower()
            result = ngrams(w, 2)
            if result:
                results.extend(result)

    for k, v in Counter(results).most_common(100):
        print(k[0] + k[1], v, sep='\t')
