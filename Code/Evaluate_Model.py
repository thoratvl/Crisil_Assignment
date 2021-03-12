import spacy
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
import sklearn.metrics
import seqeval.metrics

import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def evaluate_model_token_level(model_path, filename):
    nlp = spacy.load(model_path)

    f = open(filename, 'r')
    Examples = []
    TEXT = ''
    for line in f:
        words = line.split('\t')
        # print(words)
        if words[0] != '\n':
            Word = words[1]
            Word = Word.replace('\n', '')
            TEXT += Word + " "
        else:
            TEXT = TEXT.strip()
            Examples.append(TEXT)
            TEXT = ''
    TEXT = TEXT.strip()
    Examples.append(TEXT)

    labels = []
    for test_text in Examples:
        doc = nlp(test_text)
        ents_text = [ent.text for ent in doc.ents]
        ents_label = [ent.label_ for ent in doc.ents]
        for word in test_text.split(' '):
            if word in ents_text:
                labels.append([word, ents_label[ents_text.index(word)]])
            else:
                labels.append([word, 'O'])

    results = pd.DataFrame(labels, columns=['Word', 'Predicted'])
    ner_test = pd.read_csv(filename, delimiter='\t', header=None)
    results['Actual'] = ner_test[0]

    results.to_csv("../Results/ner_test_predicted.csv", index=False)

    print(sklearn.metrics.classification_report(results['Actual'], results['Predicted']))
    print("Accuracy:", sklearn.metrics.accuracy_score(results['Actual'], results['Predicted']))

def evaluate_model_entity_level(model_path, filename):
    nlp = spacy.load(model_path)

    f = open(filename, 'r')
    Examples = []
    TEXT = ''
    true_labels = []
    labels = []
    for line in f:
        words = line.split('\t')
        # print(words)
        if words[0] != '\n':
            Tag = words[0]
            Word = words[1]
            Word = Word.replace('\n', '')
            TEXT += Word + " "
            labels.append(Tag)
        else:
            TEXT = TEXT.strip()
            Examples.append(TEXT)
            true_labels.append(labels)
            TEXT = ''
            labels = []
    TEXT = TEXT.strip()
    Examples.append(TEXT)
    true_labels.append(labels)

    pred_labels = []
    labels = []
    for test_text in Examples:
        doc = nlp(test_text)
        ents_text = [ent.text for ent in doc.ents]
        ents_label = [ent.label_ for ent in doc.ents]
        for word in test_text.split(' '):
            if word in ents_text:
                labels.append(ents_label[ents_text.index(word)])
            else:
                labels.append('O')
        pred_labels.append(labels)
        labels = []

    print(seqeval.metrics.classification_report(true_labels, pred_labels))
    print("Accuracy:", seqeval.metrics.accuracy_score(true_labels, pred_labels))


if __name__ == '__main__':
    test_filename = "../Data/ner_test.txt"
    model_path = "../Model/Model_12March/"

    print("Token Level Results : ")
    evaluate_model_token_level(model_path, test_filename)

    print("Entity Level Results : ")
    evaluate_model_entity_level(model_path, test_filename)
