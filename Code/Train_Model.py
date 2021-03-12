# Import requirements

import spacy
import random
from spacy.util import minibatch, compounding


def create_data(filename):
    f = open(filename, 'r')
    Entity_list = []
    TRAIN_DATA = []
    start_index = 0
    TEXT = ''
    count = 0
    for line in f:
        count += 1
        words = line.split('\t')
        if words[0] != '\n':
            Tag = words[0]
            Word = words[1]
            Word = Word.replace('\n', '')
            if Tag != 'O':
                Entity_list.append((start_index, start_index + len(Word), Tag))
            start_index += len(Word) + 1
            TEXT += Word + " "
        else:
            TEXT = TEXT.strip()
            Entity_dict = {"entities": Entity_list}
            TRAIN_DATA.append((TEXT, Entity_dict))
            TEXT = ''
            start_index = 0
            Entity_list = []

    return (TRAIN_DATA)


def train_model(TRAIN_DATA, iterations=30, drop_value=0.5):
    nlp = spacy.load('en_core_web_sm')

    ner = nlp.get_pipe("ner")

    # Adding labels to the `ner`

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Disable pipeline components you dont need to change
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # TRAINING THE MODEL
    with nlp.disable_pipes(*unaffected_pipes):
        # Training for 30 iterations
        for iteration in range(iterations):

            # shuufling examples  before every iteration
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=drop_value,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)
    return nlp


def single_sentence_test(nlp, text):
    doc = nlp(text)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])


if __name__ == '__main__':
    train_filename = "../Data/ner_train.txt"
    train_data = create_data(train_filename)

    nlp = train_model(train_data, iterations=30)

    model_path = "../Model/Model_12March/"
    nlp.meta['name'] = 'New_Model'
    nlp.to_disk(model_path)

    test_text = "show me 1980s action movies"
    single_sentence_test(nlp, test_text)
