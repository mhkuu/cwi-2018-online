from sklearn.metrics import f1_score, precision_recall_fscore_support


def report_score(gold_labels, predicted_labels, detailed=False):
    macro_f1 = f1_score(gold_labels, predicted_labels, average='macro')
    # print("macro-F1: {:.3f}".format(macro_f1))
    if detailed:
        scores = precision_recall_fscore_support(gold_labels, predicted_labels)
        print("{:^10}{:^10}{:^10}{:^10}{:^10}".format("Label", "Precision", "Recall", "F1", "Support"))
        print('-' * 50)
        print("{:^10}{:^10.2f}{:^10.2f}{:^10.2f}{:^10}".format(0, scores[0][0], scores[1][0], scores[2][0], scores[3][0]))
        print("{:^10}{:^10.2f}{:^10.2f}{:^10.2f}{:^10}".format(1, scores[0][1], scores[1][1], scores[2][1], scores[3][1]))
        print()
    return macro_f1


def train_and_report(model, data, detailed=False):
    model.train(data.trainset)
    predictions = model.test(data.testset)
    if any(predictions):
        gold_labels = [sent['gold_label'] for sent in data.testset]
        macro_f1 = report_score(gold_labels, predictions, detailed=detailed)
        return macro_f1, predictions
    return 0, []


def print_results(data, predictions):
    for n, sent in enumerate(data.testset):
        if sent['gold_label'] != predictions[n]:
            print(sent['target_word'], sent['gold_label'], predictions[n])
        if n == 1000:
            break
