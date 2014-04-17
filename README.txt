This is the dataset and code in the paper:

Reasoning With Neural Tensor Networks for Knowledge Base Completion
Richard Socher, Danqi Chen, Christopher Manning, Andrew Ng
Advances in Neural Information Processing Systems (NIPS 2013)

If you use the dataset/code in your research, please cite the above paper.

@incollection{SocherEtAl2013:DeepKB,
title = {Reasoning With Neural Tensor Networks for Knowledge Base Completion},
author = {Richard Socher and Danqi Chen and Christopher Manning and Andrew Ng},
booktitle = {NIPS},
year = {2013}
}


===== DATA =====

Two datasets are included in the folder data/: WordNnet/, Freebase/.
Each dataset contains six files:

+ train.txt: training file, format (e1, rel, e2).
+ dev.txt: dev file, format (e1, rel, e2, +1/-1). We generate a negative example for each positive example.
+ test.txt: test file, same format as dev.txt.
+ entities.txt: all entities, one per line.
+ relations.txt: all relations, one per line.
+ initEmbed.mat: word list and initial embeddings, MATLAB format.


===== CODE =====
In the folder code/:

== TRAINING ==
For training, You can call the training code train.m directly.

You can also change the parameters in defaultParams.m.
data_no: 0 - WordNet, 1 - FreeBase
init_no: 0 - random initialization, 1 - Turian et al.
num_iter: number of iterations, default = 500.
train_both: 0 - only generate corrupted examples (e1, r, e'2), 1 - generate both (e1, r, e'2) and (e'1, r, e2).
batch_size: size of mini-batch.
corrupt_size: the number of generated corrupted examples for each positive example.
embedding_size: embedding size, default = 100.
slice_size: slice size in our NTN model.
reg_parameter: l2 regularization parameter.
actFunc: 0 - tanh, 1 - sigmoid, 2 - identity, default = 0.
inTensorKeepNormal: 1 - complete NTN, 0 - only keep the tensor part.

== TESTING ==
For testing, you can set the path of paramter file in paramFile and call test.m.
It will evaluate on test.txt and report classification accuracy.
