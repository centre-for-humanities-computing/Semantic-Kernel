# Semantic Kernel - Visualization of Neural Concept Graphs #

Semantic kernel trains neural embeddings of a plain text data set either as vanilla texts or tabular data, and generate a conceptual graph based on a query list. The graph is hierarchical such that the first level consists of the $m$ strongest associated terms with the query list (displayed in caps), and the second level consists of the $n$ strongest associated terms with the first level.   


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

For running in virtual environment (recommended) and assuming python3.6+ is installed.

```
sudo pip3 install virtualenv
virtualenv -p /usr/bin/python3.6 nuke
source nuke/bin/activate
```

### Installing

Clone repository and install requirements

```
git clone https://github.com/centre-for-humanities-computing/Semantic-Kernel.git
pip install requirements.txt
```

To run train model and generate graph

```
./main.sh
```

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

test that neural embeddings are trained by `semantic_vect`
```
./test/test1.sh
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With


## Contributing


## Versioning


## Authors
Kristoffer L. Nielbo

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
