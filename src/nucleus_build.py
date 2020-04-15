#!/home/knielbo/virtenvs/nuke/bin/python
import os
import sys
import pickle
import numpy as np

from util import nmax_idx, flatten, load_mdl

from scipy import spatial

import stanfordnlp

##### UTIL
def norm_seeds(lst, lang="da"):
    nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma',lang=lang)
    seeds = " ".join(lst)
    doc = nlp(seeds)
    seeds = [word.lemma.lower() for sent in doc.sentences for word in sent.words]
    return sorted(list(set(seeds)))

##### MAIN
def build_delta(DB, seeds, k=10, m=3):
    lexicon = sorted(DB.keys())

    nucle_types = dict()
    for source in seeds:
        if source in lexicon:
            deltas = list()
            for i, target in enumerate(lexicon):
                deltas.append(1 - spatial.distance.cosine(DB[source], DB[target]))
            #  deltas.append(spatial.distance.cosine(DB[source], DB[target]))# TODO: distance instead of similarity
            # print(i)
        else:
            continue

        idxs = nmax_idx(deltas, n=k)
        #  idxs = nmin_idx(deltas, n=k)# TODO: implement antigraph option, the least associated
        tokens = [lexicon[idx] for idx in idxs]
        nucle_types[source] = tokens[::-1]
    
    typelist = list()
    for nucle_type in nucle_types.keys():
        typelist.append(nucle_types[nucle_type])
    typelist = list(set(flatten(typelist)))
    typelist.sort()
    
    nucle_tokens = dict()
    for source in typelist:
        deltas = list()
        for i, target in enumerate(lexicon):
            deltas.append(1 - spatial.distance.cosine(DB[source], DB[target])) #  cosine similarity
        idxs = nmax_idx(deltas, n=m)
        #  idxs = nmin_idx(deltas, n=m)# TODO: implement antigraph option, the least associated
        tokens = [lexicon[idx] for idx in idxs]
        nucle_tokens[source] = tokens[::-1]

    nucle_token_lst = list()
    for key, val in nucle_tokens.items():
        nucle_token_lst.append(val)
    nucle_token_lst = list(set(flatten(nucle_token_lst)))
    nucle_token_lst.sort()

    # compute delta matrix
    embedding_dim = DB.popitem()[1].shape[0]
    X = np.zeros((len(nucle_token_lst), embedding_dim))
    for i, token in enumerate(nucle_token_lst):
        X[i, :] = DB[token]
    DELTA = np.zeros((X.shape[0], X.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            DELTA[i, j] = 1 - spatial.distance.cosine(x, y)
    np.fill_diagonal(DELTA, 0.)
    labels = []
    for token in nucle_token_lst:
        if token in typelist:
            labels.append(token.upper())
        else:
            labels.append(token)

    return X, DELTA, labels


def main():
    seeds = sys.argv[1:]# for single graph CMD input

    #with open(sys.argv[1], 'r') as f:# for loop folder input
    #    seeds = f.read().split()
    #print(seeds)
    with open("filename.txt", "w") as fobj:
        fobj.write(seeds[0])

    seeds = norm_seeds(seeds, lang="da")
    #print(seeds)
    
    DB = load_mdl(os.path.join("mdl", "embeddings.pcl"))
    X, DELTA, labels = build_delta(DB, seeds)
    # write query vectors
    np.savetxt(
        os.path.join("mdl", "query_mat.dat"), X, delimiter=","
        )
    # write similarity matrix
    np.savetxt(
        os.path.join("mdl", "delta_mat.dat"), DELTA, delimiter=","
        )
    # write labels (1st order are all caps)
    with open(os.path.join("mdl", "delta_labels.dat"), "w") as f:
        for label in labels:
            f.write("%s\n" % label)

if __name__ == "__main__":
    main()