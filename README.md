## GLAMOUR: Graph Learning over Macromolecule Representations
#### Somesh Mohapatra, Joyce An, Rafael Gómez-Bombarelli
#### Department of Materials Science and Engineering, Massachusetts Institute of Technology

The repository and the [Tutorial](https://github.com/learningmatter-mit/GLAMOUR/blob/main/Tutorial.ipynb) accompanies [Chemistry-informed Macromolecule Graph Representation for Similarity Computation, Unsupervised and Supervised Learning](https://iopscience.iop.org/article/10.1088/2632-2153/ac545e).<br>

<img src="https://github.com/learningmatter-mit/GLAMOUR/blob/main/overview.svg" width="100%" height="400"><br>

In this work, we developed a graph representation for macromolecules. Leveraging this representation, we developed methods for - <br>
<ul>
<li><b>Similarity Computation:</b> Using chemical similarity between monomers through cheminformatic fingerprints and exact graph edit distances (GED) or graph kernels to compare topologies, it allows for quantification of the chemical and structural similarity of two arbitrary macromolecule topologies. <br>
<li><b>Unsupervised Learning:</b> Dimensionality reduction of the similarity matrices, followed by coloration using the labels shows distinct regions for different classes of macromolecules. <br>
<li><b>Supervised learning:</b> The representation was coupled to supervised GNN models to learn structure-property relationships in glycans and anti-microbial peptides. <br>
<li><b>Attribution:</b> These methods highlight the regions of the macromolecules and the substructures within the monomers that are most responsible for the predicted properties. <br>
</ul>

### Using the codebase
To use the code with an Anaconda environment, follow the installation procedure here - 
```
conda create -n GLAMOUR python=3.11
conda activate GLAMOUR
mamba install -c anaconda -c dglteam -c pytorch -c conda-forge pytorch=2.4 cudatoolkit matplotlib rdkit dglteam/label/th24_cu121::dgl captum scikit-learn networkx seaborn svglib umap-learn notebook
wget https://anaconda.org/conda-forge/grakel/0.1.10/download/<os>/grakel-0.1.10-py311h<hash>_1.conda
mamba install grakel-0.1.10-py311h<hash>_1.conda
mamba install -c conda-forge future
pip install dgllife
```
When downloading grakel, make sure to select the correct os (e. g., `linux-64`, `win-64`, `osx-64`, ...) and the correct hash-string. It is important that it is grakel for python 3.11 ending with `_1`!

If you are new to Anaconda, you can install it from [here](https://www.anaconda.com/).

### How to cite
```
@article{mohapatra2022chemistry,
  title={Chemistry-informed Macromolecule Graph Representation for Similarity Computation, Unsupervised and Supervised Learning},
  author={Mohapatra, Somesh and An, Joyce and G{\'o}mez-Bombarelli, Rafael},
  journal={Machine Learning: Science and Technology},
  year={2022},
  publisher={IOP Publishing}
}
```

### License
MIT License
