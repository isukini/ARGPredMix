
# ARGPredMix
Antibiotic resistance gene prediction tool based on sequence features and protein language model


## Steps to Run Locally
### 1. Clone the Repository
First, clone the ARGPredMix GitHub repository to your local machine:
```bash
git clone https://github.com/isukini/ARGPredMix.git
```
### 2. Download the ESM2 Model
ARGPredMix relies on the protein language model ESM for feature encoding. You need to download the ESM2 model and place it in the model folder. You can automatically download the model using the following command:
```bash
cd model 
bash get_esm2_model.sh
```
Alternatively, you can manually download the model from
https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt

### 3. Run ARGPredMix
After completing the previous steps, you can run ARGPredMix to predict antibiotic resistance genes. Here is the command to run:
```
python ARGpredMix.py -i test.fasta -o result/test.csv
```
### 4. Output Explanation

The resulting CSV file will contain the following columns:

- **Column 1**: File name

- **Column 2**: Sequence name

- **Column 3**: Sequence

- **Column 4**: Whether it is an ARG (Antibiotic Resistance Gene)

- **Column 5**: Prediction score

- **Column 6**: Predicted resistance category

- **Column 7 and Column 8:** Sequence alignment percentage with the training dataset and CARD dataset

- The remaining columns are the predicted probabilities for each category.

![Test Result](https://raw.githubusercontent.com/isukini/ARGPredMix/main/result/pic/testresult.png)
## Dependency Installation
To predict nucleotide sequences, you need to install Prodigal:
```
conda install prodigal
```
Additionally, if you need to perform BLAST alignment, you should install BLAST:
```
conda install blast
```
### Add Necessary Conda Channels
Before installing Prodigal and BLAST, make sure to add the following channels:
```
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
```

## ARGPredMix Usage Instructions
```
usage:python ARGpredMix.py [-h] -i INPUT -o OUTPUT [-blast] [-n] [-all] [-t THRESHOLD] [-r] [-clean]

ARGPredMix - Predict resistance genes and output results to CSV

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input FASTA file path
  -o OUTPUT, --output OUTPUT
                        Output CSV file path
  -blast, --blast       Enable BLAST for antibiotic resistance gene alignment
  -n, --nucleotide      Input is nucleotide sequence or whole genome sequence
  -all, --all           Enable writing all prediction results to CSV
  -t THRESHOLD, --threshold Set the prediction threshold, default: 0.8
  -r, --radical         Enable radical mode
  -clean, --clean       Clean up temporary files

```
##  Dependencies
Make sure to install the following dependencies:
```
torch==1.9.0+cu111
numpy==1.24.4
biopython==1.83
tqdm==4.61.2
pandas==2.0.3
fair-esm==2.0.0
faiss-gpu==1.7.2
scikit-learn==1.3.2
xgboost==2.1.1
```

