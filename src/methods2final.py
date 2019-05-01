import pandas as pd 
import csv
import numpy as np
import sklearn
import gzip
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import scprep
import scanpy as sc


mk_genes = ["PTPRC", "PAX6", "PDGRFA", "NEUROD2", "GAD1", "AQP4"]
major_cell_types = ["Microglia", "NPCs", "OPCs", "Excitatory_neurons", "Interneurons", "Astrocytes"]
gene2celltype = dict(zip(mk_genes, major_cell_types))
