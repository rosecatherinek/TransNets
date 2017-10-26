'''
Tensorflow impl of Deep Co NN: Joint Deep Modeling of Users and Items Using Reviews for Recommendation - Univ. Ill. Chicago

TF CNN implementation adapted from: http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

With additional MF type components

@author: roseck
@date Mar 14, 2017
'''
import tensorflow as tf
import numpy as np
from DatasetUtils import DataWordUtils
from collections import namedtuple
from DatasetUtils import Vocab
import time

MFParams = namedtuple('MFParams', 'batch_size, rev_max_len, embedding_size, user_embedding_size, FM_k, '
                            ' learning_rate, max_epoch, dropout_keep_prob, '
                            ' word_embedding_file, train_data, '
                            'train_epochs, val_epochs, test_epochs, num_filters')


class TNetExtModel(object):

    """
    A CNN for text representation.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, mfp, userlist, itemlist,  sequence_length, word_vocab_size, embedding_size, filter_sizes, num_filters):
        self.mfp = mfp
        
        self.userVocab = Vocab.Vocab()
        self.userVocab._create(userlist)
        print 'User ID Map initialized with %d entries' % self.userVocab.size()

        self.itemVocab = Vocab.Vocab()
        self.itemVocab._create(itemlist)
        print 'Item ID Map initialized with %d entries' % self.itemVocab.size()
        
        
        self.word_vocab_size = word_vocab_size
        self.max_seq_len = sequence_length
        # Placeholders for input, output and dropout
        self.input_user = tf.placeholder(tf.int32, [None], name="input_user")
        self.input_item = tf.placeholder(tf.int32, [None], name="input_item")
        
        self.input_usertext = tf.placeholder(tf.int32, [None, sequence_length], name="input_usertext")
        self.input_itemtext = tf.placeholder(tf.int32, [None, sequence_length], name="input_itemtext")
        self.input_rating = tf.placeholder(tf.float32, [None], name="input_rating")
        self.input_review = tf.placeholder(tf.int32, [None, sequence_length], name="input_revABtext")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        self.act_global_step = tf.Variable(0, name='act_global_step', trainable=False)
        self.oth_global_step = tf.Variable(0, name='oth_global_step', trainable=False)
        self.full_global_step = tf.Variable(0, name='full_global_step', trainable=False)
        
        curr_batch_size = tf.shape(self.input_rating)[0]
        
        self.emb_initializer = tf.placeholder(dtype=tf.float32, shape=[self.word_vocab_size, embedding_size])
        self.word_embedding = tf.Variable(self.emb_initializer, trainable=False, collections=[], name='word_embedding')
        
        Ascope = 'A'
        Tscope = 'T'
        Fscope = 'F'
        
        
        with tf.variable_scope(Fscope):
            self.bias_global =  tf.get_variable('global_bias', initializer=tf.constant(0.1, dtype=tf.float32))
            self.user_embedding = tf.get_variable('user_embedding', initializer=tf.random_uniform([self.userVocab.size(), self.mfp.user_embedding_size],  -1.0, 1.0))
            self._check_op = tf.check_numerics(self.user_embedding, 'User Embedding has NaN')
        
            self.item_embedding = tf.get_variable('item_embedding', initializer=tf.random_uniform([self.itemVocab.size(), self.mfp.user_embedding_size],  -1.0, 1.0))
            self._check_op = tf.check_numerics(self.item_embedding, 'Item Embedding has NaN')


        #embed user & item
        self.embedded_u = tf.nn.embedding_lookup(self.user_embedding, self.input_user, name='user_embed')
        self.embedded_i = tf.nn.embedding_lookup(self.item_embedding, self.input_item, name='item_embed')
        
        #embed the text
        self.embedded_utext = tf.nn.embedding_lookup(self.word_embedding, self.input_usertext)
        self.embedded_btext = tf.nn.embedding_lookup(self.word_embedding, self.input_itemtext)
        self.embedded_utextx = tf.expand_dims(self.embedded_utext, -1)
        self.embedded_btextx = tf.expand_dims(self.embedded_btext, -1)
        
        #embed the revAB text
        self.embedded_revtext = tf.nn.embedding_lookup(self.word_embedding, self.input_review)
        self.embedded_revtextx = tf.expand_dims(self.embedded_revtext, -1)
        
        
        #embed user text
        with tf.variable_scope(Tscope):
            pooled_outputs_user = []
            for _, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-user-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W_user = tf.get_variable('W_user', initializer=tf.truncated_normal(filter_shape, stddev=0.1, dtype=tf.float32))
                    b_user = tf.get_variable('b_user', initializer=tf.constant(0.1, shape=[num_filters], dtype=tf.float32))
                    conv = tf.nn.conv2d(
                        self.embedded_utextx,
                        W_user,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h_user = tf.nn.relu(tf.nn.bias_add(conv, b_user), name="relu")
                    # Max-pooling over the outputs
                    pooled = tf.nn.max_pool(
                        h_user,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs_user.append(pooled)
             
            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            h_pool_user = tf.concat(axis=3, values=pooled_outputs_user)
            h_pool_user_flat = tf.reshape(h_pool_user, [-1, num_filters_total])
            
            #convert to the user dim sized vector
            G_user = tf.get_variable('G_user', initializer=tf.truncated_normal([num_filters_total, self.mfp.user_embedding_size], stddev=0.1, dtype=tf.float32))
            g_user = tf.get_variable('g_user', initializer=tf.constant(0.1, shape=[self.mfp.user_embedding_size], dtype=tf.float32))
            
            user_rep = tf.nn.tanh(tf.matmul(h_pool_user_flat, G_user) + g_user)
#             user_drop_rep = tf.nn.dropout(user_rep, self.dropout_keep_prob)
                    
            
            #embed items text
            pooled_outputs_item = []
            for _, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-item-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W_item = tf.get_variable('W_item', initializer=tf.truncated_normal(filter_shape, stddev=0.1, dtype=tf.float32))
                    b_item = tf.get_variable('b_item', initializer=tf.constant(0.1, shape=[num_filters], dtype=tf.float32))
                    conv = tf.nn.conv2d(
                        self.embedded_btextx,
                        W_item,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h_item = tf.nn.relu(tf.nn.bias_add(conv, b_item), name="relu")
                    # Max-pooling over the outputs
                    pooled = tf.nn.max_pool(
                        h_item,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs_item.append(pooled)
             
            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            h_pool_item = tf.concat(axis=3, values=pooled_outputs_item)
            h_pool_item_flat = tf.reshape(h_pool_item, [-1, num_filters_total])
            
            #convert to the item dim sized vector
            G_item = tf.get_variable('G_item', initializer=tf.truncated_normal([num_filters_total, self.mfp.user_embedding_size], stddev=0.1, dtype=tf.float32))
            g_item = tf.get_variable('g_item', initializer=tf.constant(0.1, shape=[self.mfp.user_embedding_size], dtype=tf.float32))
            
            item_rep = tf.nn.tanh(tf.matmul(h_pool_item_flat, G_item) + g_item)
#             item_drop_rep = tf.nn.dropout(item_rep, self.dropout_keep_prob)
            
            
        with tf.variable_scope(Ascope):
            self.act_bias_global =  tf.get_variable('act_global_bias', initializer=tf.constant(0.1, dtype=tf.float32))
            
            #embed review text
            pooled_outputs_rev = []
            for _, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-rev-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W_rev = tf.get_variable('W_rev', initializer=tf.truncated_normal(filter_shape, stddev=0.1, dtype=tf.float32))
                    b_rev = tf.get_variable('b_rev', initializer=tf.constant(0.1, shape=[num_filters], dtype=tf.float32))
                    conv = tf.nn.conv2d(
                        self.embedded_revtextx,
                        W_rev,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h_rev = tf.nn.relu(tf.nn.bias_add(conv, b_rev), name="relu")
                    # Max-pooling over the outputs
                    pooled = tf.nn.max_pool(
                        h_rev,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs_rev.append(pooled)
             
            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            h_pool_rev = tf.concat(axis=3, values=pooled_outputs_rev)
            h_pool_rev_flat = tf.reshape(h_pool_rev, [-1, num_filters_total])
            
            #convert to the required dim sized vector
            G_rev = tf.get_variable('G_rev', initializer=tf.truncated_normal([num_filters_total, self.mfp.user_embedding_size], stddev=0.1, dtype=tf.float32))
            g_rev = tf.get_variable('g_rev', initializer=tf.constant(0.1, shape=[self.mfp.user_embedding_size], dtype=tf.float32))
            
            rev_rep = tf.nn.tanh(tf.matmul(h_pool_rev_flat, G_rev) + g_rev)
            rev_drop_rep = tf.nn.dropout(rev_rep, self.dropout_keep_prob)
            
        
        
        #---------
        #-- GAN --
        #---------
        
        with tf.variable_scope(Ascope):
            #(A) predict using the actual revAB
            FM_act_W = tf.get_variable('FM_act_W', initializer=tf.constant(0.001, shape=[self.mfp.user_embedding_size,1], dtype=tf.float32))
            FM_act_V = tf.get_variable('FM_act_V', initializer=tf.truncated_normal([self.mfp.user_embedding_size, self.mfp.FM_k], stddev=0.001, dtype=tf.float32))
            
            #compute first order:
            act_fir_inter = tf.squeeze(tf.matmul(rev_drop_rep, FM_act_W), [1])
            
            #compute the second order interactions:
            act_sec_inter = 0.5 * tf.reduce_sum(tf.square(tf.matmul(rev_drop_rep, FM_act_V)) -  tf.matmul(tf.square(rev_drop_rep), tf.square(FM_act_V)), 1 )
            
            #self.prediction = self.bias_global + tf.reduce_sum(tf.mul(self.user_drop_rep, self.item_drop_rep), 1)
            
            self.act_prediction = self.act_bias_global + act_fir_inter + act_sec_inter
            
        with tf.variable_scope(Tscope):
            #(B) convert the user text & item text to otherRevAB
            G0_other = tf.get_variable('G0_other', initializer=tf.truncated_normal([2*self.mfp.user_embedding_size, self.mfp.user_embedding_size], stddev=0.1, dtype=tf.float32))
            g0_other = tf.get_variable('g0_other', initializer=tf.constant(0.1, shape=[self.mfp.user_embedding_size], dtype=tf.float32))
            
            G1_other = tf.get_variable('G1_other', initializer=tf.truncated_normal([self.mfp.user_embedding_size, self.mfp.user_embedding_size], stddev=0.1, dtype=tf.float32))
            g1_other = tf.get_variable('g1_other', initializer=tf.constant(0.1, shape=[self.mfp.user_embedding_size], dtype=tf.float32))
            
            #concat the user text, item text
            other_z = tf.concat(axis=1, values=[user_rep, item_rep])
            
            other_rep_0 = tf.nn.tanh(tf.matmul(other_z, G0_other) + g0_other)
            other_rep = tf.nn.tanh(tf.matmul(other_rep_0, G1_other) + g1_other)
            
            other_drop_rep = tf.nn.dropout(other_rep, self.dropout_keep_prob)
        
            #(C) predict using the transformed otherAB
#             FM_oth_W = tf.Variable(tf.constant(0.001, shape=[self.mfp.user_embedding_size,1], dtype=tf.float32), name="FM_oth_W")
#             FM_oth_V = tf.Variable(tf.truncated_normal([self.mfp.user_embedding_size, self.mfp.FM_k], stddev=0.001, dtype=tf.float32), name="FM_oth_V")
            
            #compute first order:
#             oth_fir_inter = tf.squeeze(tf.matmul(other_drop_rep, FM_oth_W), [1])
            #use the same FM as that of revAB
            oth_fir_inter = tf.squeeze(tf.matmul(other_drop_rep, FM_act_W), [1])
            
            #compute the second order interactions:
#             oth_sec_inter = 0.5 * tf.reduce_sum(tf.square(tf.matmul(other_drop_rep, FM_oth_V)) -  tf.matmul(tf.square(other_drop_rep), tf.square(FM_oth_V)), 1 )
            #use the same FM as that of revAB
            oth_sec_inter = 0.5 * tf.reduce_sum(tf.square(tf.matmul(other_drop_rep, FM_act_V)) -  tf.matmul(tf.square(other_drop_rep), tf.square(FM_act_V)), 1 )
            
            #self.prediction = self.bias_global + tf.reduce_sum(tf.mul(self.user_drop_rep, self.item_drop_rep), 1)
            
            self.oth_prediction = self.act_bias_global + oth_fir_inter + oth_sec_inter
        
        with tf.variable_scope(Fscope):    
            #(D) use the transformed otherAB too while predicting
            #concat the user, item, user text, item text, trans
            z = tf.concat(axis=1, values=[self.embedded_u, self.embedded_i, other_drop_rep])
            
            FM_W = tf.get_variable('FM_W', initializer=tf.constant(0.001, shape=[3*self.mfp.user_embedding_size,1], dtype=tf.float32))
            FM_V = tf.get_variable('FM_V', initializer=tf.truncated_normal([3*self.mfp.user_embedding_size, self.mfp.FM_k], stddev=0.001, dtype=tf.float32))
            
            #compute first order:
            fir_inter = tf.squeeze(tf.matmul(z, FM_W), [1])
            
            #compute the second order interactions:
            sec_inter = 0.5 * tf.reduce_sum(tf.square(tf.matmul(z, FM_V)) -  tf.matmul(tf.square(z), tf.square(FM_V)), 1 )
            
            #self.prediction = self.bias_global + tf.reduce_sum(tf.mul(self.user_drop_rep, self.item_drop_rep), 1)
            
            self.prediction = self.bias_global + fir_inter + sec_inter
            
        
        
        
        with tf.variable_scope('loss'):
            #(A) loss with act revAB
            act_loss = tf.reduce_sum(tf.abs(self.act_prediction - self.input_rating)) #L1 loss
            act_l2_loss = tf.nn.l2_loss(self.act_prediction - self.input_rating)
            self.act_rmse =  tf.sqrt(tf.reduce_sum(tf.multiply(act_l2_loss, 2.0/ tf.cast(curr_batch_size, tf.float32))), 'act_rmse')
            
            #(B) loss with transformed AB
#             oth_loss = tf.reduce_sum(tf.minimum(tf.abs(self.oth_prediction - self.input_rating), tf.abs(self.oth_prediction - self.act_prediction))) #L1 loss
            #loss for transformation: how much it differs from the revAB
            #use the dropout rep to regularize the transformation
            oth_loss = tf.nn.l2_loss(tf.subtract(rev_rep, other_drop_rep))
            
            oth_l2_loss = tf.nn.l2_loss(self.oth_prediction - self.input_rating) 
            self.oth_rmse = tf.sqrt(tf.reduce_sum(tf.multiply(oth_l2_loss, 2.0/ tf.cast(curr_batch_size, tf.float32))), 'oth_rmse')
            
            #(C) loss with the full info
            full_loss = tf.reduce_sum(tf.abs(self.prediction - self.input_rating)) #L1 loss
            full_l2_loss = tf.nn.l2_loss(self.prediction - self.input_rating)
            self.rmse =  tf.sqrt(tf.reduce_sum(tf.multiply(full_l2_loss, 2.0/ tf.cast(curr_batch_size, tf.float32))), 'full_rmse')

        #train ops
        allvars = tf.trainable_variables()
        self.names_all = [str(d.name) for d in allvars]
                    
        #train only the revAB -> pred
        act_optimizer = tf.train.AdamOptimizer(self.mfp.learning_rate)
        act_params = [v for v in allvars if v.name.startswith(Ascope + '/')]
        self.names_act = [str(d.name)  for d in act_params]
        self._act_train_op = act_optimizer.minimize(act_loss, var_list=act_params, global_step=self.act_global_step, name='act_train_step')
        
        #train only the otherAB -> revAB
        oth_optimizer = tf.train.AdamOptimizer(self.mfp.learning_rate)
        oth_params = [v for v in allvars if v.name.startswith(Tscope + '/')]
        self.names_oth = [str(d.name)  for d in oth_params]
        self._oth_train_op = oth_optimizer.minimize(oth_loss, var_list=oth_params, global_step=self.oth_global_step, name='oth_train_step')
        
        #train the full rep -> pred
        full_optimizer = tf.train.AdamOptimizer(self.mfp.learning_rate)
        full_params = [v for v in allvars if v.name.startswith(Fscope  + '/')]
        self.names_full = [str(d.name)  for d in full_params]
        self._full_train_op = full_optimizer.minimize(full_loss, var_list=full_params, global_step=self.full_global_step, name='full_train_step')
    
    
    def get_params(self):
        return self.names_all, self.names_act, self.names_oth, self.names_full
    
    def _proc_input_text(self, text_intlist):
        '''
        text_intlist: list of list of word ids for each  datapoint being trained/tested
        convert them into ndarray of max seq len
        
        '''
        
        text = DataWordUtils.ChopOrPadLists(text_intlist, maxlen=self.max_seq_len, filler = 0)
        
        #convert to a matrix that we can process
        text = np.reshape(text, (len(text), -1) )
        
        return text
    
    
    def _proc_input(self, user_idstr, item_idstr, user_intlist, item_intlist):
        '''
        user_intlist: list of list of word ids for each user in the datapoint being trained/tested
        item_intlist: do
        
        convert them into ndarray of max seq len
        
        '''
        #map to int
        input_user = [ self.userVocab.word_to_id(u) for u in user_idstr ]
        
        #map to int
        input_item = [ self.itemVocab.word_to_id(i) for i in item_idstr ]
        
        input_usertext = DataWordUtils.ChopOrPadLists(user_intlist, maxlen=self.max_seq_len, filler = 0)
        input_itemtext = DataWordUtils.ChopOrPadLists(item_intlist, maxlen=self.max_seq_len, filler=0)
        
        #convert to a matrix that we can process
        input_usertext = np.reshape(input_usertext, (len(input_usertext), -1) )
        input_itemtext = np.reshape(input_itemtext, (len(input_itemtext), -1) )
        
        return input_user, input_item, input_usertext, input_itemtext
    
    def run_init_all(self, sess, pre_trained_emb):
        '''
        Also initializes the word embedding with the pre trained emb
        ref: https://www.tensorflow.org/how_tos/reading_data/
        ''' 
        sess.run(self.word_embedding.initializer,
                 feed_dict={self.emb_initializer: pre_trained_emb})
        sess.run(tf.global_variables_initializer())
        
    
    def run_train_step(self, sess, user_idstr, item_idstr, input_rating, user_intlist, item_intlist, rev_intlist, dp):
        
        input_user, input_item, input_usertext, input_itemtext = self._proc_input(user_idstr, item_idstr, user_intlist, item_intlist)
        input_review = self._proc_input_text(rev_intlist)
        
        #Train revAB -> pred
        act_return = [self._act_train_op, self.act_rmse, self.act_global_step]
        
        start = time.time()
        _, act_rmse, _ = sess.run(act_return, 
                        feed_dict={
                                   self.input_rating: input_rating,
                                   self.input_review: input_review,
                                   self.dropout_keep_prob: dp
                                   })
        end = time.time()
        print 'act train time: ', (end - start), ' sec'
        
        #Train othAB -> revAB
        oth_return = [self._oth_train_op, self.oth_rmse, self.oth_global_step]
    
        start = time.time()
        _, oth_rmse, _ = sess.run(oth_return, 
                        feed_dict={
                                   self.input_usertext: input_usertext,
                                   self.input_itemtext: input_itemtext,
                                   self.input_rating: input_rating,
                                   self.input_review: input_review,
                                   self.dropout_keep_prob: dp
                                   })
        end = time.time()
        print 'oth train time: ', (end - start), ' sec'
        
        
        #Train full rep -> pred
        full_return = [self._full_train_op, self.rmse, self.full_global_step]
        start = time.time()
        _, full_rmse, _ = sess.run(full_return, 
                        feed_dict={
                                   self.input_user: input_user,
                                   self.input_item: input_item,
                                   self.input_usertext: input_usertext,
                                   self.input_itemtext: input_itemtext,
                                   self.input_rating: input_rating,
                                   self.dropout_keep_prob: dp
                                   })
        end = time.time()
        print 'full train time: ', (end - start), ' sec'
        
        return act_rmse, oth_rmse, full_rmse
        
    def get_test_score(self, sess, user_idstr, item_idstr, input_rating, user_intlist, item_intlist):
        
        input_user, input_item, input_usertext, input_itemtext = self._proc_input(user_idstr, item_idstr, user_intlist, item_intlist)
    
        to_return = [self.oth_rmse, self.rmse]
        
        return sess.run(to_return, 
                        feed_dict={
                                   self.input_user: input_user,
                                   self.input_item: input_item,
                                   self.input_usertext: input_usertext,
                                   self.input_itemtext: input_itemtext,
                                   self.input_rating: input_rating,
                                   self.dropout_keep_prob: 1.0
                                   })
        
        
        
