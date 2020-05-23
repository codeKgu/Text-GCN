### Text-GCN PyTorch
The implementation of Text GCN in Liang Yao, Chengsheng Mao, Yuan Luo. "Graph Convolutional Networks for Text Classification." In 33rd AAAI Conference on Artificial Intelligence (AAAI-19), 7370-7377.
The task is text classification. 

For an introduction to the paper check out my [blog post](https://kenqgu.com/text-classification-with-graph-convolutional-networks/).
Also checkout my [blog post]() about running experiments specifically for classifying tweets for asian prejudice during COVID-19. 
### Requirements
This repo uses python 3.6 and the following PyTorch packages:

- torch==1.3.1
- torch-cluster==1.2.4
- torch-geometric==1.1.2
- torch-scatter==1.1.2
- torch-sparse==0.4.0
- torchvision==0.4.0

I also use [comet.ml](https://www.comet.ml/site/) for experiment tracking

### Included Datasets
The included datasets are a twitter asian prejudice [dataset](https://arxiv.org/abs/2005.03909), reuters 8, and AG's news topic classification [dataset](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv).

For a new dataset, prepare a `[dataset_name]_labels.txt` and `[dataset_name]_sentences.txt` in `/data/corpus` in which each line corresponds to a document and its corresponding label. 
Use `prep_data.py` to further clean `[dataset_name]_sentences.txt`.
The script will generate a  `[dataset_name]_sentences_clean.txt`
 
### Running the model
To run the model simply change the model and dataset configurations in `config.py`. You can also enter your own cometml information to see the results and experiment running in the browser. 
After model configuration, simply run 
```
$ python main.py
```

### Results
Some initial results I have obtained using hyperparameters from the TextGCN paper are

Dataset | F1-Weighted | Accuracy
--------|-------------|---------
twitter_asian_prejudice | 	0.723 | 0.754
r8_presplit | 0.962 | 0.963
ag_presplit | 0.907 | 0.907

Try playing around with the hyperparameters or include your own dataset!