# Table of Contents
*Our paper is currently under major revision. The current version already includes the implementation of all methods and experiments in the review. We will update the code as soon as the revision is completed.*


1. Work Content
2. File Contents
3. Instructions
4. Running Experiments

# Work Content

This repository contains all the materials needed to reproduce our paper: "LESS: Efficient Log Storage System Based on Learned Model and Minimum Attribute Tree." The repository includes:

1. The source code for LESS
2. Sample datasets for evaluation, along with links to the full datasets
3. Documentation on how to use our tool to run experiments

# File Contents

## 1. Datasets

This repository includes 3 sample datasets and links to the full datasets. All sample datasets are stored in the `data/raw/` directory.

1. **Leonard Dataset**

`data/raw/vertex200m.csv` and `data/raw/edge200m.csv`

We used the open-source dataset from the paper "The Case for Learned Provenance Graph Storage Systems" for our experiments. The open-source implementation of this paper can be found at: [Leonard GitHub](https://github.com/dhl123/Leonard).

2. **DARPA TC Engagement5**

`data/raw/ta1-trace-3-e5-official-1.bin.1_concatenated_edges_top_300000.csv`

`data/raw/ta1-trace-3-e5-official-1.bin.1_concatenated_vertices_top_300000.csv`

We used the DARPA TC Engagement5 - Trace dataset for experiments.

The DARPA Engagement5 dataset is publicly available at: [DARPA TC Engagement5](https://drive.google.com/drive/folders/1s2AIHZ-I9myS_tJ3FsLgz_vdzu7PBYvv)

The original dataset is compressed as `.bin.gz` files, which need to be extracted into `.bin` files.

After obtaining the `.bin` file, you can use the `ta3-java-comsumer.tar.gz` tool in the Tools directory to convert it into JSON log format. `data/raw/darpa-trace-example.json` is a sample log.

Note that the JSON log is not a provenance graph. This project provides a JSON log parser that converts raw log data into a CSV format provenance graph.

3. **DARPA OpTC**

The DARPA OpTC dataset is available at: [DARPA OpTC](https://drive.google.com/drive/folders/1sB-rPVO84iv0OqkJiCilDLKWxklh7EYm).

The sample dataset was downloaded from `OpTCNCR/ecar-bro/benign/20-23Sep19/AIA-101-125`, and converted to a CSV provenance graph format using the JSON log parser.

The sample data was derived from the first 300,000 edges. This dataset only contains provenance graph edges, without node information.

`data/raw/ta1-trace-3-e5-official-1.bin.1_concatenated_edges_top_300000.csv`

## 2. Code Structure

### LESS Implementation

The `pipeline/` directory contains the code for provenance graph preprocessing, attribute compression, and topology compression in LESS.

### Data Preprocessing

`pipeline/preprocess/` contains the preprocessing code for the three dataset types. The raw data is read from `/data/raw`.

### `preprocess_leonard.py`

* Preprocessing for the Leonard dataset.

### `preprocess_darpa_tc.py`

* Preprocessing for the DARPA TC dataset.

### `preprocess_darpa_optc.py`

* Preprocessing for the DARPA OpTC dataset.

### Attribute Compression

`pipeline/property/` — Implements attribute string compression for the provenance graph nodes and edges.

#### `encode.py`

* Node and edge attribute compression for provenance graphs.
* **Edit Distance Calculation:** Uses Levenshtein distance to measure similarity between node attribute strings.
* **Bag of Words Model:** Builds a character count vector to represent string features.
* **Transformer Vectorization:** Utilizes SentenceTransformer to extract semantic vectors of node attributes and calculate cosine similarity.
* **Similarity Measurement:** Applies a sliding window approach based on data locality to compute similarity between adjacent nodes, reducing computational complexity.
* **Attribute Tree Construction:** Uses a priority queue to merge the two closest nodes during the compression process.
* **Encoded Output:** Custom encoding strategy is applied to compress and store edit operations, reducing the size of the output file.

#### `decode.py`

* Node and edge attribute decoding for provenance graphs.
* **Recursive Decoding:** Reads the binary byte stream from the encoded file, recursively restoring parent nodes of the attribute tree first, followed by the current node.
* **Edit Operation Parsing:** Extracts specific edit operations (insert, replace, delete) and applies them to the base string to decode the original string.
* **Edge and Node Attribute Decoding:** Decodes the encoded node and edge attributes from the stored file and outputs them in CSV format.

### `encode_kdtree.py`, `encode_lsh.py`, `encode_separate.py`

* Experimental code implementing attribute encoding using KD-Tree, LSH, and other algorithms.

### Topology Compression

`pipeline/edge/`

#### `encode.py`

Encodes the topology of the provenance graph. It reads edge data from CSV files, maps node identifiers to integer indices, and builds an edge dictionary where the start node index is the key and the target node index list is the value. The script performs differential encoding on the edge dictionary, converting the data into a one-dimensional NumPy array, marked with specific delimiters. The encoded data is saved as `.npy` format files.

#### `correct.py`

The `Corrector` class builds a correction table. Its primary function is to read sequences from the encoded `.npy` data file, use a pre-trained machine learning model to predict the sequence, and compare the predictions with the actual values to find errors. A correction table is then generated to record the positions of the prediction errors and the corresponding true values. The correction table is saved as a binary file, containing information about offsets and true values. The `Re_Corrector` class in `correct.py` is used to restore the encoded provenance graph topology data based on the correction table. It reads the correction table (`calibration_table.txt`) and uses it to correct the predicted values from the pre-trained model. The corrected data is saved as `.npy` format files for further processing.

#### `decode.py`

Decodes the topology of the provenance graph. It reads encoded data from `.npy` files, reconstructs the edge dictionary from the one-dimensional NumPy array, and builds an edge dictionary where the start node index is the key and the target node index list is the value. The script performs inverse differential encoding to revert the data back to its original edge structure.

#### `query.py`

Performs breadth-first search (BFS) queries on the provenance graph data. It starts from a set of starting nodes and traverses the graph using the BFS algorithm, obtaining a set of node and edge IDs. The `query_bfs` function queries descendant node and edge IDs, while the `query_bfs2` function queries ancestor node and edge IDs. The script returns the list of node IDs and edge IDs for subsequent attribute queries.

#### `model.py`

Defines various deep learning models used in testing.

#### `train_deep.py`

Trains deep learning models during testing to predict the next element in the encoded provenance graph data. The script constructs neural network models (e.g., LSTM) defined in `model.py` and trains them using the encoded provenance graph sequences from `.npy` files. The script extracts input sequences and target labels (the next element of the sequence) using a sliding window method. During training, it outputs the loss and accuracy for each epoch. The optimal model parameters are saved in a specified directory for later use in prediction tasks.

#### `train_ml.py`

Trains traditional machine learning models to predict the next element in the encoded provenance graph data. The script loads encoded provenance graph sequences from `.npy` files and trains models like XGBoost. The trained models are saved in a specified directory for later use in prediction tasks.

#### `utils.py`

Defines the early stopping mechanism used in training deep learning models during testing.

## Dataset Tools

`tools/` contains tools for handling DARPA TC and DARPA OpTC datasets.

#### `json_schema.py`

Analyzes dataset format.
* **Maintaining JSON Structure Tree:** Recursively traverses input JSON data, maintaining a tree structure based on data types, and handles dictionaries, lists, and basic elements.
* **File Reading and Processing:** Reads JSONL files line by line, merging them into an existing tree structure, with support for parallel processing of large files.
* **Storing Structure Analysis Results:** Saves the JSON structure tree into a new file for later use and viewing.

#### `parser.py`

Parses DARPA TC and DARPA OpTC JSON datasets.

#### `parser_csv.py`

Converts datasets into CSV format.

#### `parser_neo4j.py`

Stores datasets in a Neo4j database.

## Output Directory

The `/data/` directory contains all inputs and output results of the program:

* `raw/` — Original datasets
* `preprocess/` — Preprocessed data
* `encode/` — Encoded data

# Instructions

1. **Install Python Dependencies**

First, install the Python 3.11.5 environment and then install the following dependencies:
```shell
pip install -r requirements.txt
```

2. **Unzip the Original Dataset**

On Linux:
```shell
unzip data/raw/datasets.zip -d data/raw
```

On Windows, manually extract the files to the `data/raw` directory.

3. **Set Environment Variables**

In the project root directory, set the environment variables.

On Windows:
```shell
$env:PYTHONPATH = "$env:PYTHONPATH;$PWD"
```

On Linux:
```shell
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

4. **Running Experiments**

The command to run the experiments should be executed in the project root directory.

Each script will accomplish the following tasks:

* Generate compressed datasets under `data/compress_result`
* Generate the queried nodes' topological relationships and the attribute values of the nodes and edges under `data/query_result`
* Generate intermediate files for preprocessed datasets under `data/preprocess`
* Generate encoded results of nodes and edges attributes under `data/encode`
* Generate decoded results of attributes and topology under `data/decode`
* Store the XGBoost model parameters under `data/model`
* Store the error correction table for misclassified data when predicting topological relationships under `data/correct`

```shell
python scripts/run_toy.py
```

```shell
python scripts/run_darpatc.py
```

```shell
python scripts/run_darpaoptc.py
```

# Leonard：Usage

This section reproduces the experiments from the toy dataset and part of the DARPA TC dataset from the paper "The Case for Learned Provenance Graph Storage Systems." 

1. Install Python 3.11.5 environment, and then install the following dependencies: 
```shell
pip install -r requirements.txt
```

2. Extract the original dataset 

linux
```shell
unzip data/raw/datasets.zip -d leonard/data/raw
```
Windows needs to unzip the file to the "data/raw/leonard" folder. 

3. Set environment variables 

Set environment variables in the root directory of the project.

windows
```shell
$env:PYTHONPATH = "$env:PYTHONPATH;$PWD"
```

linux
```shell
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

4. Conduct the experiment. 

The command to run the experiment needs to be executed in the root directory of the project. 

Each script will perform the following functions: 

* generate compressed data set in the leonard/data/compress directory
* generate topological relationships, node and edge attributes values of queried nodes in the leonard/data/query_result directory 
* generate intermediate files for preprocessing the data set in the leonard/data/preprocess directory 
* generate encoded results in the leonard/data/encode directory 
* generate LSTM model parameters in the leonard/data/model directory 
* generate prediction correction table in the leonard/data/correct directory 

```shell
python leonard/run_toy.py
```

```shell
python leonard/run_darpatc.py
```


For the query part, we query the neighborhood of a fixed node on the sample dataset. You can modify the query node ID and search depth to observe different results.

To modify the query node:
* LESS: Modify the global variable start_node_ids in scripts/run.py
* Leonard: Modify the global variable start_node_ids in leonard/query_withedge_vertex_removeedge_darpa_batch.py

Note: If the node only appears in the edge data and no corresponding entry is found in the node data, the query result will only contain a hash field. The Darpa OpTC dataset does not contain node data.

To switch between CPU/GPU environments for Leonard, modify the use_gpu variable in leonard/config.py. By default, GPU is used.


# Expected Results

*We have published the results obtained from our implementation on our own machine, providing a reference for reproducing the results.*


The *sample* folder stores the expected output examples of LESS and Leonard on the dataset. The table below summarizes the runtime and storage overhead on the test machine.

The hardware configuration of the test machine is as follows:

* CPU: Intel i5-13490F
* Memory: 32GB
* GPU: NVIDIA GeForce RTX 4070 Ti

## Toy Dataset

### Sample Files
* Leonard: sample/leonard_toy_output_demo.zip
* LESS: sample/less_toy_output_demo.zip

### Performance Metrics

| Solution | CPU/GPU | Preprocessing Time | Compression Time | Query Time | Total Time | Storage Space |
| :------- | :------ | :----------------- | :--------------- | :--------- | :--------- | :------------ |
| Leonard  | CPU     | 4.2s               | 102.6s           | 2.3s       | 109.1s     | 2.15MB        |
| Leonard  | GPU     | 4.2s               | 49.6s            | 2.5s       | 56.3s      | 2.13MB        |
| LESS     | CPU     | 0.7s               | 6.6s             | 4.8s       | 12.1s      | 0.62MB        |

## Darpa TC Dataset

### Sample Files
* Leonard: sample/leonard_darpatc_output_demo.zip
* LESS: sample/less_darpatc_output_demo.zip

### Performance Metrics

| Solution | CPU/GPU | Preprocessing Time | Compression Time | Query Time | Total Time | Storage Space |
| :------- | :------ | :----------------- | :--------------- | :--------- | :--------- | :------------ |
| Leonard  | CPU     | 10.7s              | 241.3s           | 5.3s       | 257.3s     | 6.04MB        |
| Leonard  | GPU     | 10.8s              | 93.9s            | 5.7s       | 110.4s     | 6.06MB        |
| LESS     | CPU     | 1.8s               | 25.9s            | 14.8s      | 42.5s      | 1.12MB        |

## Darpa OpTC Dataset

### Sample Files

* LESS: sample/less_darpaoptc_output_demo.zip

### Performance Metrics

| Solution | CPU/GPU | Preprocessing Time | Compression Time | Query Time | Total Time | Storage Space |
| :------- | :------ | :----------------- | :--------------- | :--------- | :--------- | :------------ |
| LESS     | CPU     | 2.5s               | 34.3s            | 16.1s      | 52.9s      | 6.81MB        |
