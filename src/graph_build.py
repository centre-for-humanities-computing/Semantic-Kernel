#!/home/knielbo/virtenvs/nuke/bin/python
import os
import numpy as np
import string
import matplotlib.pyplot as plt

import networkx as nx
import community
from networkx.drawing.nx_agraph import graphviz_layout
from community import community_louvain


def dist_prune(DELTA, prune=True):
    """ transform similarity matrix to distance matrix
    - prune matrix by removing edges that have a distance larger
        than condition cond (default mean distance)
    """
    w = np.max(DELTA)
    DELTA = np.abs(DELTA - w)
    np.fill_diagonal(DELTA, 0.)
    if prune:
        cond = np.mean(DELTA)  # + np.std(DELTA)# TODO: transform to parameter with choice between models
        for i in range(DELTA.shape[0]):
            for j in range(DELTA.shape[1]):
                val = DELTA[i, j]
                if val > cond:
                    DELTA[i, j] = 0.
                else:
                    DELTA[i, j] = DELTA[i, j]

    return DELTA


def gen_graph(DELTA, labels, figname="nucleus_graph"):
    """ generate graph and plot from DELTA distance matrix
    - labels is list of node labels corresponding to columns/rows in DELTA
    """
    DELTA = DELTA * 10  # scale
    dt = [("len", float)]
    DELTA = DELTA.view(dt)

    #  Graphviz
    G = nx.from_numpy_matrix(DELTA)
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), labels)))
    pos = graphviz_layout(G)

    G = nx.drawing.nx_agraph.to_agraph(G)
    G = nx.from_numpy_matrix(DELTA)
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), labels)))
    #nx.draw(G, pos=pos, with_labels=True, node_size=100)
    #plt.savefig(os.path.join("fig", os.path.basename(figname).split(".")[0] + "_0.png"))
    #plt.close()

    np.random.seed(seed=1234)
    parts = community_louvain.best_partition(G)
    values = [parts.get(node) for node in G.nodes()]

    plt.figure(figsize=(10, 10), dpi=300, facecolor='w', edgecolor='k')
    plt.axis("off")
    nx.draw_networkx(
        G, pos=pos, cmap=plt.get_cmap("Set3"), node_color=values,
        node_size=500, font_size=12, width=1.5, font_weight="bold",
        font_color="k", alpha=1, edge_color="gray"
        )

    plt.tight_layout()
    plt.savefig(figname + "_1.png")
    plt.close()



def main():
    delta = np.loadtxt(
        os.path.join("mdl", "delta_mat.dat"), delimiter=","
        )
    DELTA = dist_prune(delta)

    with open(os.path.join("mdl", "delta_labels.dat"), "r") as f:
        labels = f.read().split("\n")
    labels = labels[:-1]
    
    #print(labels)

    with open("filename.txt", "r") as fobj:
        filename = fobj.readlines()[0].lower()

    outname = os.path.join("fig", "concept_graph_{}".format(filename))
    gen_graph(DELTA, labels, figname=outname)
    
    #print(filename)
    os.remove("filename.txt")
if __name__ == "__main__":
    main()
