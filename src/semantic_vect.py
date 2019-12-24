#!/home/knielbo/virtenvs/nuke/bin/python

import os
import pandas as pd
from nltk.tokenize.stanford import StanfordTokenizer
from nltk import word_tokenize


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
        #tokenizer = StanfordTokenizer()
        #tokenizer = word_tokenize()
        self.read()
        if lang == "da":
            print("DANISH")
            print(word_tokenize(self.content[0]))
        else:
            print(lang)


    




def main():
    print("\n\n test main() for semantic_vect \n\n") 
    fpath = os.path.join("dat","tabular")
    DATA = Corpus("foobar", fpath)
    DATA.read()
    #print(DATA.content[0])

    print(DATA.normalize("da"))

if __name__ == "__main__":
    main()