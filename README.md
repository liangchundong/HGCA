# Analyzing Heterogeneous Network with Missing Attributes by Unsupervised Contrastive Learning
This repository contains the code of the paper:
>[Analyzing Heterogeneous Network with Missing Attributes by Unsupervised Contrastive Learning]()

which has been accepted by *TNNLS*.
## Dependencies
* Python3.7
* NumPy
* SciPy
* scikit-learn
* NetworkX
* PyTorch
## Datasets
The preprocessed data are available at [Baidu Netdisk](https://pan.baidu.com/s/1VpHJh6SEVcWvOPtSfAgrTw) (password: hgca) or [Google Drive](https://drive.google.com/file/d/1UFeFQBVNLcSA5OCtzCDwuT4K5LNQl-Xp/view?usp=sharing).

Please extract the zip file to folder `data`.

## Example Usages
Before running the code，please make a directory named checkpoint.
* `python run.py --cuda --dataset ACM --metapath-weight 0.4#0.6`
* `python run.py --cuda --dataset Yelp --metapath-weight 0.2#0.6#0.2`
* `python run.py --cuda --dataset DBLP --metapath-weight 0.1#0.1#0.8`

## Start From Zero (Using ACM dataset as an example)
1. Download `data.zip` and extract it to folder `data` (you can delete folders other than `raw`).
2. Run *`python preprocess_ACM.py`* (in `data` folder) to process raw data (generate a folder named `ACM` in `data` folder).
3. Run *`python sampling.py --dataset ACM`* (in `preprocess` folder) to sample nodes for batch training (generate a folder named `indices` in `data/ACM` folder).
4. Run *`python walk.py --dataset ACM`* (in `preprocess` folder) to generate sampled node sequence for training of *metapath2vec* (generate a file named `walks_ACM.txt` in `preprocess` folder).
5. Learn joint embeddings via *metapath2vec* and generate a file named `metapath2vec_ACM_embeddings.txt` in `preprocess` folder.
    - *`cd metapath2vec`*
    - *`./metapath2vec -train ../preprocess/walks_ACM.txt -output ../preprocess/metapath2vec_ACM_embeddings -pp 0 -size 128 -window 4 -negative 10 -threads 32`*
7. Run *`python embedding.py --dataset ACM`* (in `preprocess` folder) to process `metapath2vec_ACM_embeddings.txt` (generate `metapath2vec_emb_node.npy` and `metapath2vec_emb_word.npy` in `data/ACM` folder).
8. Run *`python run.py --cuda --dataset ACM --metapath-weight 0.4#0.6`*

Please refer to the code for detailed parameters.

## Citing
Dongxiao He, et al. "Analyzing Heterogeneous Network with Missing Attributes by Unsupervised Contrastive Learning," IEEE Trans. Neural
Netw. Learn. Syst.
