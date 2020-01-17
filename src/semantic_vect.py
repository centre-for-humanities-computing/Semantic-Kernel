#!/home/knielbo/virtenvs/nuke/bin/python

import os
import pandas as pd
from nltk.tokenize.stanford import StanfordTokenizer
from nltk import word_tokenize
import lemmy as le


class Corpus(object):
    """ A basic corpus class for reading documents ('content') from vanilla or tabular files
    
    Attributes:
        name: str representing the corpus' name
        path: str representing the relative path
    """

    def __init__(self, name, path):
        self.name = name
        self.path = path
    
    def read(self):
        """ Reads filenames and content of file(s) on path

        Attributes:
            fnames: list of str containing filename on files on path 
            fpaths: list of str containing relative path and filename to files on path
            content
        """
        self.fnames = os.listdir(self.path)
        self.fpaths = [os.path.join(self.path,fname) for fname in self.fnames]
        if len(self.fnames) > 1:
            self.content = list()
            for fname in self.fpaths:
                with open(fname, "r") as f:
                    self.content.append(f.read())
        else:
            df = pd.read_csv(self.fpaths[0])
            self.content = df.content.values

    def normalize(self, lang):
    """ linguistic normalization
        lemma: list of str containing 

    """
        # TODO: implement with standford NLP lemmatization pipeline
        self.read()
        self.lemma = list()
        if lang == "da":
            lemmatizer = le.load(lang)
            print("DANISH")
            for text in self.content[:2]:
                text_lemmas = list()
                tokens = word_tokenize(text)
                for token in tokens:
                    lemma = lemmatizer.lemmatize("", token)
                    text_lemmas.append(lemma[-1].lower())
                self.lemma.append(text_lemmas)
        #    for i, token in enumerate(tokens):
        #        print(token, "->", text_lemmas[i])

        #    #tokens = word_tokenize(self.content)


        else:
            print(lang)


    




def main():
    print("\n\n test main() for semantic_vect \n\n") 
    fpath = os.path.join("dat","tabular")
    DATA = Corpus("foobar", fpath)# instantiate a corpus object
    DATA.read()
    DATA.normalize("da")
    
    print(DATA.lemma[2])
if __name__ == "__main__":
    main()