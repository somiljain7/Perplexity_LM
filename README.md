# README


## Introduction
```
Kenlm objective is to answer language model queries using less time and memory.
model was found to be 2.4 times faster and use only 57% of the memory compared to SRILM. This improvement was achieved through the use of a TRIE data structure with optimized features such as bit-level packing, sorted records, interpolation search, and optional quantization, all aimed at reducing memory consumption effectively.


Parameters required for Generate n-gram based LM using KenLM are mentioned below

top_k: KenLM will use only top_k frequency words to make its vocab.

input_txt_path: Path to your clean input text file(default finetuning/train.wrd). If the input file is generated from multiple text file then set this variable equal to 'combined_txt_file_save_path'

output_path: Folder path to store binary file.

kenlm_bins: Path to bin folder inside kenlm/build.
```
## Installation

### Install Python Dependencies
```bash
pip3 install -r requirements.txt
```

### Install Kenlm Dependencies
```bash
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
```

## Content
```
├── requirements.txt
├── script.sh
├── generate_lm.py
└── perplexity.py
```

## Usage

### `perplexity.py`
A Python script with 3 methods:
- `EDA(f_path_data)`: Explore text file and clean it.
- `train(f_path_data)`: Train LM using the Kenlm toolkit and update its parameters.
- `perplexity_task(test_sentences)`: Test trained LM with a test set of data and output the results.

## running the script 
```
- script.sh is command to run generate_lm.py with its custom hyperpater which is being called my train method in perplexity.py


python3 perplexity.py --in_txt /home/coderatwork/artpark_interview/Assignment1_all_bn.txt
```
### Observability 
When using the data_prep_for_train(f_path_data) method instead of data_prep_for_clean_train(f_path_data) in the perplexity calculation script (perplexity.py), I observed that the model achieved lower perplexity values. This indicates that by not merging clean POS-tagged data from an NLTK corpus with the training set for training the language model, the model exhibited better performance, higher confidence, and a better understanding of the language used in the sentences. This observation is documented in the README.md file for clarity and reproducibility.

## RESOURCES 
- https://deepspeech.readthedocs.io/en/master/Scorer.html?highlight=language%20model
- http://www2.statmt.org/moses/?n=FactoredTraining.BuildingLanguageModel
- https://kheafield.com/papers/avenue/kenlm.pdf

## Future Options
- trying ek-step foundation's LM pretrained on indian regional languages 
- trying GPT tokeninzations for dealing with subwords for dealing with out of vocab words .

## Contributor and Contact
- Contributor: Somil Jain (GitHub: somiljain7)
- Contact: Somil Jain (Email: [somiljain71100@gmail.com])

