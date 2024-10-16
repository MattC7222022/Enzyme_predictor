This is a convolutional feed-forward NN with an adjustable input size. It takes in sequences of amino acids and outputs an array of 87 numbers. The first number is the probability of being an enzyme, the next 7 are probablities of each enzyme class
the next 79 are probabilities of each enzyme subclass.

It takes a maximum input size of 2000 residues


Here is how I got the training data:

To get gene lists for each organism:
Use the biomart script or just go to Genbank and search in genome (get the genbank annotated genome)

To get sequences:

1. go to 
https://www.uniprot.org/id-mapping

2. Add refseq numbers. do from refseq protein to Uniprot. For protein accession numbers, do Genbank CDS to uniprot
For Entrez_geneID, scroll down to "Genome annotation" then click Gene ID

3. Format: excel, not compressed, all genes:
Entry(not entry name)	Gene Names (primary)	Organism	EC number	Length	Sequence

4. Use make_training_sets.py (this is a custom script). It reads in all the training sets, and then randomizes them all together.
