# TransNets

This is the implementation of TransNets, a rating prediction method proposed in 
[TransNets: Learning to Transform for Recommendation](https://arxiv.org/pdf/1704.02298.pdf). 
Rose Catherine and William Cohen. _In Proc. 11th ACM Conference on Recommender Systems (RecSys), 2017._

If you use code or data from this repository, please cite the above paper.


FAQ is here:  [https://github.com/rosecatherinek/TransNets/wiki/FAQ-for-TransNets](https://github.com/rosecatherinek/TransNets/wiki/FAQ-for-TransNets)

### Running the code
(for TransNet-Ext, scroll further down)

`python TNetMain.py batch_size review_max_len embedding_size learning_rate max_epoch dropout_keep_prob 'path/to/word_emb.pkl'  'path/to/train_epochs'  'path/to/val_epochs'  'path/to/test_epochs'  num_filters  output_embedding_size  'path/to/dict.pkl'  'path/to/training_data' num_transform_layers FMk window_size 'path/to/save/model' `

Example:

`python TNetMain.py 500 1000 64 0.002 30 0.5 'data/yelp_2017/word_emb.pkl' 'data/yelp_2017/rand1/train_epochs' 'data/yelp_2017/rand1/val_epochs'  'data/yelp_2017/rand1/test_epochs'  100  50  'data/yelp_2017/dict.pkl'  'data/yelp_2017/rand1/train_INT.gz' 2 8 3 'tnet_models/'  `

### Files

To construct the required files, first convert your data into the format: 
> `user_id <tab> item_id <tab> rating <tab> review`

Separate out the training data from validation & test -- typically a randomized 80:10:10 split.

1. Dictionary file: maps word to id. To construct it, see item 2 below. The dict file that I used can be found in `/data/yelp_2017/dict.pkl.gz`. Uncompress it before using it or modify the code to read the compressed pickle file.
2. Word Embedding File: pre-trained embedding. To construct the word embedding 
and the dictionary, you can pre-train on the training data using the auxiliary code at: 
`DatasetUtils/Word2VecBasicGZ.py`. This implementation uses TensorFlow's distribution 
of Word2Vec. The code is self explanatory. The original source is here: [https://www.tensorflow.org/tutorials/word2vec](https://www.tensorflow.org/tutorials/word2vec). 

   You may alternately use any of the freely available word embeddings like Stanford's [Glove](https://nlp.stanford.edu/projects/glove/) or Google's [Word2Vec on News data](https://code.google.com/archive/p/word2vec/).
   
   The word embedding that I used can be found in `/data/yelp_2017/word_emb.pkl.gz`. Uncompress it before using it or modify the code to read the compressed pickle file.
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
   * Example: An epoch dir looks like this: `ls /data/yelp_2017/train_epochs`
   
      > `epoch1.gz  epoch2.gz  epoch3.gz  epoch4.gz`
   * Example: A single epoch file looks like this: `gunzip -c /data/yelp_2017/train_epochs/epoch1.gz | head -1`
   
      > `R-rsTFabkWqA_Sunn1E1pw	Urlg0kok4QuBYiFUnbTbDA	1.0	5 317 7 68 7 6 239 35 89 39 8 6 2012 422 131 8 216 32 515 7977 1 280 70 314 7 488 5 114 23 2 4229 12 6 149 998 744 1 6 653 285 10 924 192 41 15 20 35 8 216 3 22 343 28 6 184 130 32 20 35 17 2 301 458 3 5 12161 1 264 3 5 314 7 177 9 6 1136 1 9 11 635 239 105 2 301 334 15 33289 39 1 39 31 149 47333 4 6 149 278 1 2 533 131 61 14 803 8 136 1 5 145 42 226 19 3642 955 3 4 145 226 7 25 6 227 1149 124 1 5 25 78 138 7 13101 27 12 253 1 264 3 2 54 13 8381 11 634 1 5 72 118 68 39 124 1 14 365 4 5 180 76 267 39 131 107 13 58 47 10 366 4 30 529 169 127 115 739 1 9 8 606 23 6 1834 152 29 19 30 24 322 3 22 39 8 83 54 390 497 1 2 7797 453 87 52 716 4 534 739 4 192 87 52 261 56 57 87 221 1 98 3 9 180 52 261 3812 181 7 92 3632 52 8121 1 5 28 7 424 63 7 125 4 755 311 45 18 224 68 974 312 1 2 51 50 114 23 196 1033 2 397 55 18 30 39 1 75 954 63 52 34 3 61 24 463 87 352 428 75 453 2 34 47 3 4 32 2 380 10 9 44 60 18 1375 7 2 370 67 2 537 15 9 28 708 87 66 257 7 57 378 144 309 10 254 32 44 344 70 2236 7 376 87 15 2 390 8 149 1 69 1417 126 282 8 75 2 83 393 497 1417 7 129 4138 1158 52 34 8 459 60 18 392 90 9 1 20 653 83 28 404 401 4 19 30 26 310 1976 1 5 72 118 68 7 20 194 163 124 1 29 3 2 51 11 634 344 4 5 199 50 57 15 47 39 148 255 1 264 3 67 6 398 517 14 365 4 5 114 39 1395 523 4 71 18 1253 8 341 1158 105 144 91 184 51 85 15 5 28 1253 32 20 112 1429 1 18 518 13 2 538 1569 12 2 422 7 120 23 2 16540 4 162 52 113 344 70 118 61 1 14 365 454 3178 12 2 422 4 70 451 23 4 858 91 77 51 12 3812 181 503 16 85 4 244 312 1 5 3935 7 68 39 507 89 9 805 1213 7 57 79 34 39 4 19 48 26 333 45 19 1394 63 1 166 3 92 14 365 9558 7 68 39 89 9 11 1395 47 10 52 140 4 166 3 18 204 24 376 45 18 146 138 7 57 51 1	81 248 47 15 2 143 5 106 14 175 40 1433 38 19 25 535 54 57 54 263 175 45 16 535 6 310 62 53 22 89 5 61 26 2064 311 12 9 3 19 61 26 504 9 1 20 11 184 334 1272 1 1 190 494 5 72 26 113 58 88 124 102 40 563 51 163 1 180 82 105 66 257 12 54 175 1 132 1 5 92 277 2 245 1145 50 7 311 71 27 63 4 44 19 95 376 41 8 15 9 8 47 12 822 1 2827 1 4 357 5 103 5355 6968 5 421 24 36 138 64 2107 564 1 18 25 106 58 5355 1981 27 2096 187 13 2 566 4 19 1460 48 66 299 345 1 264 3 14 213 161 1434 7 5355 1981 27 2 230 51 100 78 775 4 2 175 100 78 459 3 1414 3 4 3473 1 23 20 1281 1943 5 1336 6 478 113 7 36 938 63 32 4438 1 5 297 895 617 139 3 204 24 68 444 17 15 1 29 5 106 175 12 300 1 340 310 934 843 2088 3 868 3 4 536 1 159 871 1051 843 2088 1305 1 5 479 32 2 245 32 1 2 245 8 741 22 2 30 322 1 5 518 84 2 330 620 92 3046 431 642 347 32 41 26 54 3568 5 8 39 12 210 181 1 392 3 6 1153 107 63 7 2 620 4 5553 41 1 37 6 5 567 12 14 113 4 2433 14 1305 5 776 15 2 281 2215 30 232 22 2 1168 897 8 459 1 5 205 2 1349 45 2 175 8 626 47 42 215 4 75 3114 85 5 101 26 165 22 5 48 24 156 29 85 1 5 751 2 459 897 10 175 4 2 175 347 7307 4 301 1 2 1153 629 7 129 6 175 2606 1 29 5 8 273 17 2 1927 7 154 520 181 12 54 149 175 4 1912 2 141 93 1305 7 57 459 1 29 5 133 2 4 518 1 280 14 175 8 599 5 2302 227 1 2 310 175 8 33 2 141 2088 175 30 418 634 1 2 175 30 166 459 3 1414 3 4 280 124 7307 1 77 54 572 2 175 4 19 114 1242 7 2 2400 1 39 31 6 568 10 175 411 13 14 1013 4 92 201 19 48 26 504 895 6464 139 1 5 158 1452 88 6 135 747 1 5 421 24 36 138 64 7 5355 6968 2107 564 1 340 257 4 895 181 7 1519 6 459 13267 175 1 19 156 20 11 2886 1 1632 10768 21 5 56 177 1625 246 45 5 95 1 23 3689 9317 32 761 5 1336 66 113 17 20 194 12 822 1 2 113 8 193 2405 3 2133 121 3 4 6 219 211 1 2 822 55 8 7818 181 1 32 761 5 612 6 331 58 6 1640 3690 70 95 24 195 14 200 35 3 73 100 6 229 2206 766 47 330 1 32 761 70 277 124 3076 7 195 41 1 5 492 32 2 330 767 3 77 54 8 409 7 1757 288 1 392 32 761 70 479 3 77 818 3 244 3 50 1515 14 34 3 28 41 766 4 1435 47 1 4544 181 523 268 5 114 7 178 14 211 109 3 4 225 32 3 4 2 5966 19 320 23 9 8 5322 21 5 101 765 621 4158 4 39 11 77 140 41 62 457 312 224 25 7 178 66 5322 212 19 567 12 21 14 193 2405 4 14 121 30 214 459 3 4 61 26 270 4125 1 5 277 148 255 3 2 533 5 1375 7 192 41 540 71 48 16 151 41 7 48 111 5 204 24 1704 16 79 342 1 379 5 192 259 5 228 14 342 64 1 70 142 70 95 320 6 887 23 14 2044 29 5 1698 7 15 1 447 5 222 7 35 66 828 113 306 14 2044 172 19 3853 41 6 895 617 4865 138 23 148 166 17 828 718 1 5 277 7 119 7 335 14 887 89 9 8 24 23 14 2044 1 2 2495 5 1375 7 95 24 195 144 3181 10 9 3 4 15 75 8 2 83 54 497 15 143 1 77 75 8 24 89 5 1375 7 6 2770 4 14 822 1640 8 6 2770 1 126 75 192 41 7 113 306 125 1 5 205 45 5 235 158 57 2 1231 89 5 507 28 14 2595 133 828 1 75 142 77 5 204 24 335 2 828 113 1231 4 335 14 887 1 9 27 14 342 21 16 327 41 5322 3 523 3 4 459 34 1 75 142 5 235 25 7 120 13 7 335 9 4 5 204 24 113 828 62 822 3 60 2 1051 113 19 2643 63 23 229 55 8 828 4 822 1 418 6826 4 775 906 21 5 72 162 14 334 1162 172 19 49 7 177 116 5322 459 34 4 3140 59 342 1 1336 14 113 828 605 152 4 981 7 335 54 10 59 91 296 1258 1 85 1397 730 91 340 5378 150 340 159 10403 12 74 2069 1 106 6 871 1 159 2277 12 1585 47 1 2 834 8 74 12727 1 5 277 7 376 88 5 61 24 1519 2 1231 4`
   * The above line is of the format: `user_id <tab> item_id <tab> rating <tab> other reviews by user_id with words mapped to id <tab> other reviews about item_id with words mapped to id`. Although, the `<tab>`s show up as space characters above. 
   * Each `epoch.gz` file is ~7GB compressed. So, I haven't uploaded the file in this repo. It should be straightforward to create.
4. Training data: is of the form:
`user_id <tab> item_id <tab> rating <tab> review with words mapped to id`

   * Example (the `<tab>`s show up as space characters below): `gunzip -c /data/yelp_2017/rand_train_INT.gz | head -1`
   
      > `5I3nKFOif1fBA_1DxQIvdg	G-5kEa6E6PD5fkBRuA7k9Q	5	5 8 815 7 68 4 1576 32 54 10 14 265 682 27 411 1 2 35 8 43 1 2 51 8 136 1 242 734 1 43 34 1 337 2 691 3 1195 662 3 4 2039 1 2 536 1796 2931 12 656 1 2 516 30 1060 1 714 3 5 730 6 161 364 726 2 516 30 197 4 28 6 523 206 1 77 233 7 48 15 1 2 516 31 1060 1 5 61 24 25 171 12 1226 21 279 3 6 42 715 513 468 130 1	`
   * For `Yelp 2017`, this is a ~1GB compressed file. So, I haven't uploaded the file in this repo. It should be straightforward to create.
   * You may use the auxiliary code at: `DatasetUtils/WordToIntMain` to construct this for your data.


### Reading the output

The code outputs a lot of debug statements. It is better to re-direct the output to a file for easier reference. Else, please modify the code to print only lines that interest you. 

The main lines to look for are those prefixed with: `Testing MSE Full:`. Example: `grep "Testing MSE Full:" output/yelp_2017/tn.out`

   > Testing MSE Full: Val	0 500 	1.89057151769
   
   > Testing MSE Full: Test	0 500 	1.88791528764
   
   > Testing MSE Full: Val	0 1000 	1.91673440598
   
   > Testing MSE Full: Test	0 1000 	1.91217695346
   
   > Testing MSE Full: Val	0 1500 	1.83988785228
   
   > Testing MSE Full: Test	0 1500 	1.8353025388

The above lines show the MSE values for Validation and Test as the training proceeds. The first number `0` is the epoch number and the second number `500`, `1000`, etc. is the training iteration. Each iteration processes `batch_size` number of training examples. 

In this sample run, one of the better performances is at epoch 1 (7000th iteration): 

   > Testing MSE Full: Val	1 7000 	1.651443312

   > Testing MSE Full: Test	1 7000 	1.6465206496
   
However, it reaches the best at epoch 16 (108000th iteration):

   > Testing MSE Full: Val	16 108000 	1.63820302245

   > Testing MSE Full: Test	16 108000 	1.63276760098


### Running the TransNet-Ext code

The command to run the TransNet-Ext learning is similar to that of the TransNet model, except, you need to specify a user and item shortlist for which an embedding needs to be learned. 

`python TNetExtMain.py batch_size review_max_len embedding_size learning_rate max_epoch dropout_keep_prob 'path/to/word_emb.pkl'  'path/to/train_epochs'  'path/to/val_epochs'  'path/to/test_epochs'  num_filters  output_embedding_size  'path/to/dict.pkl'  'path/to/usershortlist.txt' 'path/to/itemshortlist.txt' 'path/to/training_data' FMk window_size `

Example:

`python TNetExtMain.py 500 1000 64 0.002 30 0.5 'data/yelp_2017/word_emb.pkl' 'data/yelp_2017/rand1/train_epochs' 'data/yelp_2017/rand1/val_epochs'  'data/yelp_2017/rand1/test_epochs'  100  50  'data/yelp_2017/dict.pkl'  'data/yelp_2017/usershortlist.txt' 'data/yelp_2017/bizshortlist.txt' 'data/yelp_2017/rand1/train_INT.gz' 8 3 `

The shortlists are users and items that appear in the training data. Users and items that appear in the validation/test, but not in train, 
will be mapped to id 0 and a random embedding will be used. 

Reading the output is same as that of the TransNet output file.
