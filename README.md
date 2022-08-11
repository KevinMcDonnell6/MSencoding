# CNN-GNN for MS ion encoding

All <i>de novo</i> peptide identification algorithms use machine leanrning to encode spectrum ions in a bid to identify the underlying amino acid sequence. Here we present a CNN-GNN hybrid model for encoding spectrum ions. The model as presented can be used to identify peptide ions in MS spectra but can be modified for inclusion into a <i>de novo</i> peptide sequencing algorithm.

## How to run:

To train a model, edit the path to the training and validation files and run the following:
```
python run_graph_model.py --logging_dir <training_directory> --org <organism_type>
```


To predict the b-ions and y-ions in the spectra, edit the path to the test file and run the following:
```
python test_graph_model.py --logging_dir <training_directory> --org <organism_type>
```

