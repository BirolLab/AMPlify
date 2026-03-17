# AMPlify

AMPlify is an attentive deep learning model for antimicrobial peptide prediction.

<p align="center">
	<img src="AMPlify.png">
</p>

### Dependencies

* Python 3.6
* Keras 2.2.4
* Tensorflow 1.12
* Numpy <1.17
* Pandas
* Scikit-learn
* Biopython
* h5py <3

### Installation

1. Create a new `conda` environment:
```
conda create -n amplify python=3.6
```

2. Activate the environment:
```
conda activate amplify
```

3. Install AMPlify in the environment:
```
conda install -c bioconda amplify
```
`train_amplify` and `AMPlify` can now be run. See usage information below.

4. To deactivate an active environment, use:
```
conda deactivate
```

### Datasets

Datasets for training and testing are stored in the `data` folder. Please specify the directory if you would like to use those datasets for training or testing the model.
* Balanced training set: `AMPlify_AMP_train_common.fa` + `AMPlify_non_AMP_train_balanced.fa`
* Balanced test set: `AMPlify_AMP_test_common.fa` + `AMPlify_non_AMP_test_balanced.fa`
* Imbalanced training set: `AMPlify_AMP_train_common.fa` + `AMPlify_non_AMP_train_imbalanced.fa`
* Imbalanced test set: `AMPlify_AMP_test_common.fa` + `AMPlify_non_AMP_test_imbalanced.fa`

### User-specified data:

AMPlify only accepts the 20 standard amino acids. However, if the user specifies a stop codon by adding an asterisk character to the end of a peptide sequence, AMPlify will ignore the last character and allow the peptide to participate in the training or inference stage.

### Pre-trained sub-models

Weights for the pre-trained balanced/imbalanced sub-models are stored in the `models` folder.
* Balanced model: The one proposed in the AMPlify publication; Trained on the balanced training set; Suitable to use for predicting putative AMPs from a relatively curated candidate set (e.g., sequences already filtered by length/charge and through sequence homology search).
* Imbalanced model: Trained on the imbalanced training set; Suitable for situations where the number of non-AMPs in the input sequence set is far greater than that of AMPs (e.g., a large transcriptomics sequence database).

### Train

Usage: `train_amplify [-h] -amp_tr AMP_TR -non_amp_tr NON_AMP_TR [-amp_te AMP_TE] [-non_amp_te NON_AMP_TE] [-sample_ratio {balanced,imbalanced}] -out_dir OUT_DIR -model_name MODEL_NAME
`
```
optional arguments:
  -h, --help            Show this help message and exit
  -amp_tr AMP_TR        Training AMP set, fasta file
  -non_amp_tr NON_AMP_TR
                        Training non-AMP set, fasta file
  -amp_te AMP_TE        Test AMP set, fasta file (optional)
  -non_amp_te NON_AMP_TE
                        Test non-AMP set, fasta file (optional)
  -sample_ratio {balanced,imbalanced}
                        Whether the training set is balanced
                        or not (balanced by default, optional)
  -out_dir OUT_DIR      Output directory
  -model_name MODEL_NAME
                        File name of trained model weights
```
Example: `train_amplify -amp_tr ../data/AMPlify_AMP_train_common.fa -non_amp_tr ../data/AMPlify_non_AMP_train_balanced.fa -amp_te ../data/AMPlify_AMP_test_common.fa -non_amp_te ../data/AMPlify_non_AMP_test_balanced.fa -out_dir ../models/ -model_name model`

Expected output: 1) The model weights trained using the specified data; 2) Test set performance, if test sequences have been specified.

### Predict

Usage: `AMPlify [-h] [-m {balanced,imbalanced}] -s SEQS [-od OUT_DIR] [-of {txt,tsv}] [-sub {on,off}] [-att {on,off}]`
```
optional arguments:
  -h, --help            Show this help message and exit
  -m {balanced,imbalanced}, --model {balanced,imbalanced}
                        Balanced or imbalanced model (balanced by default, optional)
  -s SEQS, --seqs SEQS  Sequences for prediction, fasta file
  -od OUT_DIR, --out_dir OUT_DIR
                        Output directory (optional)
  -of {txt,tsv}, --out_format {txt,tsv}
                        Output format, txt or tsv (tsv by default, optional)
  -sub {on,off}, --sub_model {on,off}
                        Whether to output sub-model results, on or off (off by
                        default, optional)
  -att {on,off}, --attention {on,off}
                        Whether to output attention scores, on or off (off by
                        default, optional)
```
Example: `AMPlify -s ../data/AMPlify_AMP_test_common.fa`

Expected output: Predicted probability scores, AMPlify log scaled scores, and classes of the input sequences. The AMPlify log scaled score is calculated as `-10*log10(1-Probability_score)`. Results for invalid sequences will be filled with `NA`.

Note: In the default setting, sequences with `AMPlify scores > 3.01` (i.e., `AMPlify probability scores > 0.5`) are predicted as AMPs.

### AMP discovery
* Additional scripts and data for the AMP discovery pipeline that has been utilized for bullfrog genome mining are provided in `auxiliary/amp_discovery_pipeline` (added in v1.0.1). Parameters for GMAP and MAKER2 are described in the Methods section of the [manuscript](https://doi.org/10.1186/s12864-022-08310-4).

* Additional scripts and data for the AMP mining workflow that has been utilized to mine novel AMPs from the UniProtKB/Swiss-Prot database are provided in `auxiliary/amp_mining_workflow` (added in v2.0.0). Detailed methods can be found in the [manuscript](https://doi.org/10.1002/pro.70083). The input data and result files are publicly accessible through a [Zenodo repository](https://doi.org/10.5281/zenodo.8133088).

### Author

Chenkai Li (chenkai.li@ubc.ca)

### Contact

If you have any questions, comments, or would like to report a bug, please file a Github issue or contact us.

### Citation

If you use AMPlify in your work, please cite our publications:
1. The research article for AMPlify and the discovery of RaCa series peptides:
> Li, C., Sutherland, D., Hammond, S.A. et al. AMPlify: attentive deep learning model for discovery of novel antimicrobial peptides effective against WHO priority pathogens. BMC Genomics 23, 77 (2022). https://doi.org/10.1186/s12864-022-08310-4
--------------------------------------------------------------------------------

2. The data note paper introducing the imbalanced model in addition to the original balanced model:
> Li, C., Warren, R.L. & Birol, I. Models and data of AMPlify: a deep learning tool for antimicrobial peptide prediction. BMC Research Notes 16, 11 (2023). https://doi.org/10.1186/s13104-023-06279-1
--------------------------------------------------------------------------------

3. The research article for mining the UniProtKB/Swiss-Prot database for novel AMPs:
> Li, C., Sutherland, D., Salehi, A. et al. Mining the UniProtKB/Swiss-Prot database for antimicrobial peptides. Protein Science 34, e70083 (2025). https://doi.org/10.1002/pro.70083
