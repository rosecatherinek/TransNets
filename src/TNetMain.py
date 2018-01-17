'''
@author: rosecatherinek
'''
import traceback

import TNetModel

import tensorflow as tf
import sys
import pickle
import time

from DatasetUtils import Misc
from StreamingUtils import FromDisk
from DatasetUtils import DataPairMgr

def train(dcmf, savedir):
    
    cfg = tf.ConfigProto(allow_soft_placement=True )
    cfg.gpu_options.allow_growth = True
    
    sess = tf.Session(config=cfg)


    #read the embedding
    emb = pickle.load( open( dcmf.mfp.word_embedding_file, "rb" ) )
    
    dcmf.run_init_all(sess, emb)
    del emb
    
    step = 0
    
    #get the epoch files from the train dir
    train_epochs = Misc.get_epoch_files(dcmf.mfp.train_epochs)
    print 'Train Epochs: found ', len(train_epochs), ' files'
        
    #get the epoch files from the val dir
    val_epochs = Misc.get_epoch_files(dcmf.mfp.val_epochs)
    print 'Val Epochs: found ', len(val_epochs), ' files'
    

    #get the epoch files from the test dir
    test_epochs = Misc.get_epoch_files(dcmf.mfp.test_epochs)
    print 'Test Epochs: found ', len(test_epochs), ' files'
    
    #load the revAB from the train file
    dp_mgr = DataPairMgr.DataPairMgr(dcmf.mfp.train_data)
    
    names = True
    for epoch in range(dcmf.mfp.max_epoch):
        print 'Epoch: ', epoch
        
        train_time = 0
        
        train_file = train_epochs[epoch % len(train_epochs)]
        print 'Train file: ', train_file        
        
        trainIter = FromDisk.FromDisk(train_file)
        batch_iter = trainIter.BatchIter(dcmf.mfp.batch_size)
        
        while True:
            step += 1
            rList, retUTextInt, retBTextInt, revABList  = ([] for i in range(4))
            try:
                #read the values
                uList, bList, rList, retUTextInt, retBTextInt  = batch_iter.next()
                #get the revAB 
                revABList = [ dp_mgr.get_int_review(u, b) for u,b in zip(uList, bList) ]
                
            except StopIteration:
                #end of this data epoch
                break
            
            start = time.time()
            act_rmse, oth_rmse, full_rmse = dcmf.run_train_step(sess, rList, retUTextInt, retBTextInt, revABList, dcmf.mfp.dropout_keep_prob)
            end = time.time()
            
            if names:
                names = False
                names_all, names_act, names_oth, names_full = dcmf.get_params()
                print 'Variables - all trainable: ', names_all
                print 'Variables trained in act: ', names_act
                print 'Variables trained in oth: ', names_oth
                print 'Variables trained in full: ', names_full
            
            
            tt= end - start
            print 'Train time ', (tt), ' sec'
            train_time += tt
            
            
            print 'Step ', step, ' act: ', act_rmse
            print 'Step ', step, ' oth: ', oth_rmse
            print 'Step ', step, ' full: ', full_rmse
            
            if step % 500 == 0:
                save, mse = test_valtest(sess, dcmf, epoch, step)
            
                if save: 
                    #save the current model
                    save_model(sess, dcmf, mse, savedir)
                
            
        
        print 'End of Epoch Testing'
        save, mse = test_valtest(sess, dcmf, epoch, step)
        if save: 
            #save the current model
            save_model(sess, dcmf, mse, savedir)
        
        sys.stdout.flush()
        
def save_model(sess, dcmf, mse, savedir):
    print 'Saving the current model'
    start = time.time()
    file_name = savedir + str(round(mse,2)).replace('.', '_') + '_model.ckpt'
    dcmf.save_model(sess, file_name)
    end = time.time()
    print 'Model saved in ', str(end - start), ' sec'
    
    

def test_valtest(sess, dcmf, epoch, step):  
    val_epochs = Misc.get_epoch_files(dcmf.mfp.val_epochs)      
    print 'Testing Perf: Val\t', epoch, step
    val_file = val_epochs[0]
    print 'Val file: ', val_file
    v_val_review = FromDisk.FromDisk(val_file)
    val_iter = v_val_review.BatchIter(dcmf.mfp.batch_size)    
    
    oth_mse_val, full_mse_val = test(sess, dcmf, val_iter)

    print 'Testing MSE Other: Val\t', epoch, step, '\t', oth_mse_val
    print 'Testing MSE Full: Val\t', epoch, step, '\t', full_mse_val
    
    test_epochs = Misc.get_epoch_files(dcmf.mfp.test_epochs)
    print 'Testing Perf: Test\t', epoch, step
    test_file = test_epochs[0]
    print 'Test file: ', test_file
    v_test_review = FromDisk.FromDisk(test_file)
    test_iter = v_test_review.BatchIter(dcmf.mfp.batch_size)    

    oth_mse_test, full_mse_test = test(sess, dcmf, test_iter)
    
    print 'Testing MSE Other: Test\t', epoch, step, '\t', oth_mse_test
    print 'Testing MSE Full: Test\t', epoch, step, '\t', full_mse_test
    
    if full_mse_val < 1.7:  
        #TODO: need to find the saving criteria from the previous MSE
        return True, full_mse_val #save this model
    else:
        return False, full_mse_val
    
def test(sess, dcmf, batch_iter):
    '''test the performance using the iterator 
    '''
    oth_mse = 0.0
    full_mse = 0.0
    tot = 0
    i = 0
    
    test_time = 0
    
    while True:
        rList, retUTextInt, retBTextInt   = ([] for i in range(3))
        try:
            #read the values
            _, _, rList, retUTextInt, retBTextInt   = batch_iter.next()
        except StopIteration:
            #end of data
            break
        
        start = time.time()
        oth_rmse, full_rmse = dcmf.get_test_score(sess, rList, retUTextInt, retBTextInt  )
        end = time.time()
        
        tt = end - start
        print 'Test time ', tt, ' sec'
        test_time += tt
        
        print 'Full_RMSE', full_rmse
            
        oth_mse += oth_rmse*oth_rmse * len(rList)
        full_mse += full_rmse*full_rmse * len(rList)
        tot += len(rList)
        
        i += 1
 
    oth_mse = oth_mse / tot
    full_mse = full_mse / tot
    
#     rmse_full = math.sqrt(full_mse)
    print 'Total Test time: ', test_time/60.0, ' min'
#     print 'x rmse = ', rmse_full
    return  oth_mse, full_mse



if __name__ == '__main__':
    
    print 'Run Cmd: ', sys.argv
    
    batchsize = 500
    revmaxlen = 1000
    embeddingsize = 64
    learningrate = 0.002
    maxepoch = 30
    dropoutkeepprob = 0.5
    wordembeddingfile = 'data/yelp_2017/word_emb.pkl'
    trainep = 'data/yelp_2017/rand1/train_epochs'
    valep = 'data/yelp_2017/rand1/val_epochs'
    testep = 'data/yelp_2017/rand1/test_epochs'  
    numfilters = 100
    userembeddingsize = 50
    dict_file = 'data/yelp_2017/dict.pkl' #word -> id
    traindata = 'data/yelp_2017/rand1/train_INT.gz' #UserBizRatingRev format: gives userA,bizB -> revAB
    transLayers = 2
    FMk = 8
    filtersizes = [3] #t = 3
    savedir = 'tnet_models/'
    
    
    if len(sys.argv) > 1:
        batchsize = int(sys.argv[1])  #100
        revmaxlen = int(sys.argv[2])  #n == ? max? avg?
        embeddingsize = int(sys.argv[3])  #300 google news pre trained
        learningrate = float(sys.argv[4])   # 0.002
        maxepoch = int(sys.argv[5])  
        dropoutkeepprob = float(sys.argv[6]) #?
        wordembeddingfile = sys.argv[7]
        trainep = sys.argv[8]
        valep = sys.argv[9]
        testep = sys.argv[10]    
        numfilters = int(sys.argv[11]) #n1 = 100
        userembeddingsize = int(sys.argv[12]) #|x| = |y| = 50
        dict_file = sys.argv[13] # word -> id mapping file
        traindata = sys.argv[14]
        transLayers = int(sys.argv[15])
        FMk = int(sys.argv[16])
        fs = int(sys.argv[17])
        filtersizes = [fs]
        savedir = sys.argv[18]
        
        
    
    

    mfp = TNetModel.MFParams(
        batch_size = batchsize,
        rev_max_len = revmaxlen,
        embedding_size = embeddingsize,
        user_embedding_size = userembeddingsize,
        FM_k = FMk,
        learning_rate = learningrate,
        max_epoch = maxepoch,
        dropout_keep_prob = dropoutkeepprob,
        word_embedding_file = wordembeddingfile,
        train_data = traindata,
        train_epochs = trainep,
        val_epochs = valep,
        test_epochs = testep,
        num_filters = numfilters)

    print 'Settings = ', mfp
    
    #create the word vocab
    dict_word_to_id = pickle.load( open( dict_file, "rb" ) )
    vocab_size = len(dict_word_to_id)
    del dict_word_to_id
    
    dcmf = TNetModel.TNetModel(mfp, sequence_length=mfp.rev_max_len, word_vocab_size=vocab_size, embedding_size=mfp.embedding_size, filter_sizes=filtersizes, num_filters=mfp.num_filters, trans_layers=transLayers)
    
    try:
        train(dcmf, savedir)
    except Exception:
        traceback.print_exc()
    
    print 'Done'
