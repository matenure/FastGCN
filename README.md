# fastgcn

FastGCN: 
train_batch_multiRank_inductive_reddit_Mixlayers_sampleA.py is the final model. (precomputated the AH in the bottom layer)

train_batch_multiRank_inductive_reddit_Mixlayers_uniform.py is the codes for uniform sampling.

train_batch_multiRank_inductive_reddit_Mixlayers_appr2layers.py is the codes for 2-layer approximation.

pubmed-original**.py means the codes are used for original Cora or Pubmed datasets. Users could also change their datasets by changing the data load function from load_data() to load_data_original().

create_Graph_forGraphSAGE.py is used to transfer the data into the GraphSAGE format, so that users can compare our method with GraphSAGE. We also include the transferred original Cora dataset in this repository (./data/cora_graphSAGE).






