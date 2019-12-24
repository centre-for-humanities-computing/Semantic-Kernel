#!/home/knielbo/virtenvs/nuke/bin/python

import os


class Corpus(object):
    """ A basic corpus class for reading documents ('content') from vanilla or tabular files
    
    Attributes:
        name: A string representing the corpus' name
        path: A string representing the relative path
    """

    def __init__(self, name, path):
        self.name = name
        self.path = path
    
    def read_files(self):
        self.fnames = os.listdir(self.path)




def main():
    print("\n test main() for semantic_vect \n") 
    fpath = os.path.join("dat","tabular")
    DATA = Corpus("foobar", fpath)
    DATA.get_fnames()
    print(DATA.fnames)

if __name__ == "__main__":
    main()