# TransNets

This is the implementation of TransNets, a rating prediction method proposed in 
[TransNets: Learning to Transform for Recommendation](https://arxiv.org/pdf/1704.02298.pdf). 
Rose Catherine and William Cohen. _In Proc. 11th ACM Conference on Recommender Systems (RecSys), 2017._

If you use code or data from this repository, please cite the above paper.

### Running the code

`python TNetMain.py batch_size review_max_len embedding_size learning_rate max_epoch dropout_keep_prob 'path/to/word_emb.pkl'  'path/to/train_epochs'  'path/to/val_epochs'  'path/to/test_epochs'  num_filters  output_embedding_size  'path/to/dict.pkl'  'path/to/training_data' num_transform_layers 'path/to/save/model' `

Example:

`python TNetMain.py 500 1000 64 0.002 30 0.5 'data/yelp_2017/word_emb.pkl' 'data/yelp_2017/rand1/train_epochs' 'data/yelp_2017/rand1/val_epochs'  'data/yelp_2017/rand1/test_epochs'  100  50  'data/yelp_2017/dict.pkl'  'data/yelp_2017/rand1/train_INT.gz' 2 'tnet_models/'  `

### Files

To construct the required files, first convert your data into the format: 
> `user_id <tab> item_id <tab> rating <tab> review`

Separate out the training data from validation & test -- typically a randomized 80:10:10 split.

1. Dictionary file: maps word to id. To construct it, see item 2 below.
2. Word Embedding File: pre-trained embedding. To construct the word embedding 
and the dictionary, you can pre-train on the training data using the auxiliary code at: 
`DatasetUtils/Word2VecBasicGZ.py`. This implementation uses TensorFlow's distribution 
of Word2Vec. The code is self explanatory. The original source is here: [https://www.tensorflow.org/tutorials/word2vec](https://www.tensorflow.org/tutorials/word2vec). 

   You may alternately use any of the freely available word embeddings like Stanford's [Glove](https://nlp.stanford.edu/projects/glove/) or Google's [Word2Vec on News data](https://code.google.com/archive/p/word2vec/)
3. Epoch files: In every epoch, for each (user,item) pair in the data, 
we need to construct their text representations using the reviews that the user has written 
previously and reviews that others wrote for the item previously. Constructing this while 
training slows down the training tremendously since the GPUs wait around for the CPU. One 
workaround is to create them before hand (this can be parallelized). So, the path to each of the 
epochs -- train, validation, test -- correspond to directories where there are many epoch files 
for training, validation and testing, already written out so that they can be simply streamed.

   * First of all, convert your text data to its int representation using your word-to-id dict. You may use the auxiliary code at: `DatasetUtils/WordToIntMain`
   * To create the epoch files, you may use the auxiliary code at: `StreamingUtils/ToDisk.py`. In the code, `dataFile` is the file with the reviews that are used to construct the text for user and item. This is always the training data. `inFile` is the file for which you want to construct an epoch. Supply the train file to construct training epochs. Similarly, validation and test for constructing their respective epochs. For the `Yelp 2017` dataset and a machine of 32 cores, this takes about 1.5 hrs to create a training epoch. You could run the `ToDisk` on multiple machines to trivially parallelize the dataset creation since each epoch is randomized and is independent of other epochs. Each epoch creation can also be trivially parallelized by splitting the `inFile` if you don't have a large number of cores, but have access to multiple machines. 
   * Streaming from the pre-created epoch files while training TransNets takes a couple of minutes at worst compared to constructing them at run time which can amount to a couple of hours per epoch.
4. Training data: is of the form:
`user_id <tab> item_id <tab> rating`






