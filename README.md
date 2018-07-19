# FastGCN
This is the Tensorflow implementation of our ICLR2018 paper: ["**FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling**".](https://openreview.net/forum?id=rytstxWAW&noteId=ByU9EpGSf)


Instructions of the sample codes:

[For Reddit dataset]

	train_batch_multiRank_inductive_reddit_Mixlayers_sampleA.py is the final model. (precomputated the AH in the bottom layer) The original Reddit data should be transferred into the .npz format using this function: transferRedditDataFormat.

	train_batch_multiRank_inductive_reddit_Mixlayers_uniform.py is the model for uniform sampling.

	train_batch_multiRank_inductive_reddit_Mixlayers_appr2layers.py is the model for 2-layer approximation.

	create_Graph_forGraphSAGE.py is used to transfer the data into the GraphSAGE format, so that users can compare our method with GraphSAGE. We also include the transferred original Cora dataset in this repository (./data/cora_graphSAGE).


[For pubmed or cora]

	train.py is the original GCN model.

 	pubmed_Mix_sampleA.py 	The dataset could be defined in the codes, for example: flags.DEFINE_string('dataset', 'pubmed', 'Dataset string.')

	pubmed_Mix_uniform.py and pubmed_inductive_appr2layers.py are similar to the ones for reddit.

	pubmed-original**.py means the codes are used for original Cora or Pubmed datasets. Users could also change their datasets by changing the data load function from load_data() to load_data_original().
