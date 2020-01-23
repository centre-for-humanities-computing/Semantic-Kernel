import os
import pandas as pd
import stanfordnlp
from nltk import sent_tokenize

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
            content: list of str containing content of files on path
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
        

    def normalize(self, lang="da"):
        """ linguistic normalization
        Attributes:
            lemma: list of str containing lemmas of content whitespace tokenized

        """
        self.read()
        self.lemma = list()
        nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma',lang=lang)
        for i, text in enumerate(self.content[:10]):# TODO: to full index
            if type(text) == str:
                doc = nlp(text)
                lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
                self.lemma.append(" ".join(lemmas))
                #print(lemmas)
            else:#TODO: move test for sring to read() method
                self.lemma.append("NA")
                print("file {} is corrupt".format(i))
    
    def sentsplit(self, source, lang="da"):
        """ sentence tokenization
            Attributes:
                sentences: list of str containing either lemmatized or original sentences of content
        """
        self.sentences = list()
        if source == "lemma":
            self.normalize(lang)
            for lemma in self.lemma:
                self.sentences.append(sent_tokenize(lemma))
        else:
            self.read()
            for text in self.content:
                if type(text) == str:
                    self.sentences.append(sent_tokenize(text))

        self.sentences = [sent for text in self.sentences for sent in text]