#!/home/knielbo/virtenvs/nuke/bin/python

import os
from util import Corpus
import gensim
import logging
import pickle


def main():
    # data
    fpath = os.path.join("dat","tabular")
    DATA = Corpus("foobar", fpath)# instantiate a corpus object
    DATA.sentsplit("lemma")
    sent_tokens = [sentence.split() for sentence in DATA.sentences]
    logging.basicConfig(
        format='%(asctime)s : %(levelname)  s : %(message)s',
        level=logging.INFO
        )
    # model
    mdl = gensim.models.Word2Vec(size=128, window=5, min_count=5, workers=4)
    mdl.build_vocab(sent_tokens)
    mdl.train(sent_tokens, total_examples=mdl.corpus_count, epochs=mdl.iter)
    # result
    embeddings = dict()
    for word in list(mdl.wv.vocab.keys()):
        embeddings[word] = mdl[word]
    
    with open(os.path.join("mdl","embeddings.pcl"), 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()