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