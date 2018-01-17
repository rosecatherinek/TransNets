'''

@author: roseck
@date Mar 15, 2017
'''
import traceback

import TNetExtModel

import tensorflow as tf
import sys
import pickle
import time

import DatasetUtils.FileReader as fr
from DatasetUtils.Misc import get_epoch_files
from StreamingUtils import FromDisk
from DatasetUtils import DataPairMgr

def train(dcmf):
    
    cfg = tf.ConfigProto(allow_soft_placement=True )
    cfg.gpu_options.allow_growth = True
    
    sess = tf.Session(config=cfg)


    #read the embedding
    emb = pickle.load( open( dcmf.mfp.word_embedding_file, "rb" ) )
    
    dcmf.run_init_all(sess, emb)
    del emb
    
    step = 0
    
    #get the epoch files from the train dir
    train_epochs = get_epoch_files(dcmf.mfp.train_epochs)
    print 'Train Epochs: found ', len(train_epochs), ' files'
        
    #get the epoch files from the val dir
    val_epochs = get_epoch_files(dcmf.mfp.val_epochs)
    print 'Val Epochs: found ', len(val_epochs), ' files'
    

    #get the epoch files from the test dir
    test_epochs = get_epoch_files(dcmf.mfp.test_epochs)
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
            uList, bList, rList, retUTextInt, retBTextInt, revABList  = ([] for i in range(6))
            try:
                #read the values
                uList, bList, rList, retUTextInt, retBTextInt  = batch_iter.next()
                #get the revAB 
                revABList = [ dp_mgr.get_int_review(u, b) for u,b in zip(uList, bList) ]
                
            except StopIteration:
                #end of this data epoch
                break
            
            start = time.time()
            act_rmse, oth_rmse, full_rmse = dcmf.run_train_step(sess, uList, bList, rList, retUTextInt, retBTextInt, revABList, dcmf.mfp.dropout_keep_prob)
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
            
            if step % 1000 == 0:
                test_valtest(sess, dcmf, epoch, step)
            
        
        print 'End of Epoch Testing'
        test_valtest(sess, dcmf, epoch, step)
        
        sys.stdout.flush()
        

def test_valtest(sess, dcmf, epoch, step):  
    val_epochs = get_epoch_files(dcmf.mfp.val_epochs)      
    print 'Testing Perf: Val\t', epoch, step
    val_file = val_epochs[0]
    print 'Val file: ', val_file
    v_val_review = FromDisk.FromDisk(val_file)
    val_iter = v_val_review.BatchIter(dcmf.mfp.batch_size)    
    
    oth_mse_val, full_mse_val = test(sess, dcmf, val_iter)

    print 'Testing MSE Other: Val\t', epoch, step, '\t', oth_mse_val
    print 'Testing MSE Full: Val\t', epoch, step, '\t', full_mse_val
    
    test_epochs = get_epoch_files(dcmf.mfp.test_epochs)
    print 'Testing Perf: Test\t', epoch, step
    test_file = test_epochs[0]
    print 'Test file: ', test_file
    v_test_review = FromDisk.FromDisk(test_file)
    test_iter = v_test_review.BatchIter(dcmf.mfp.batch_size)    

    oth_mse_test, full_mse_test = test(sess, dcmf, test_iter)
    
    print 'Testing MSE Other: Test\t', epoch, step, '\t', oth_mse_test
    print 'Testing MSE Full: Test\t', epoch, step, '\t', full_mse_test
    
def test(sess, dcmf, batch_iter):
    '''test the performance using the iterator 
    '''
    oth_mse = 0.0
    full_mse = 0.0
    tot = 0
    i = 0
    
    test_time = 0
    
    while True:
        uList, bList, rList, retUTextInt, retBTextInt   = ([] for i in range(5))
        try:
            #read the values
            uList, bList, rList, retUTextInt, retBTextInt   = batch_iter.next()
        except StopIteration:
            #end of data
            break
        
        start = time.time()
        oth_rmse, full_rmse = dcmf.get_test_score(sess, uList, bList, rList, retUTextInt, retBTextInt  )
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
    
    batchsize = 10
    revmaxlen = 10
    embeddingsize = 64
    learningrate = 0.002
    maxepoch = 5
    dropoutkeepprob = 0.7
    wordembeddingfile = '/Users/roseck/Documents/RecoNNData/sl50/w2v/x_embeddings.pkl'
    trainep = '/Users/roseck/Documents/RecoNNData/sl50/x_INT_Train.txt'
    valep = '/Users/roseck/Documents/RecoNNData/sl50/x_INT_Test.txt'
    testep = '/Users/roseck/Documents/RecoNNData/sl50/x_INT_Test.txt'  
    numfilters = 100
    userembeddingsize = 50
    dict_file = '/Users/roseck/Documents/RecoNNData/sl50/w2v/x_dict.pkl' #word -> id
    userfile = '/usr1/public/rkanjira/RecoNN/sl50/usershortlist.txt' #use only train ids
    itemfile = '/usr1/public/rkanjira/RecoNN/sl50/bizshortlist.txt' #use only train ids
    traindata = '/Users/roseck/Documents/RecoNNData/sl50/x_INT_Train.txt' #UBRR format
    FMk = 8
    filtersizes = [3] #t = 3
    
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
        userfile = sys.argv[14]
        itemfile = sys.argv[15]
        traindata = sys.argv[16]
        FMk = int(sys.argv[17])
        fs = int(sys.argv[18])
        filtersizes = [fs]
        
        

    mfp = TNetExtModel.MFParams(
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
    
    dcmf = TNetExtModel.TNetExtModel(mfp,userlist=fr.readFileToSet(userfile), itemlist=fr.readFileToSet(itemfile), sequence_length=mfp.rev_max_len, word_vocab_size=vocab_size, embedding_size=mfp.embedding_size, filter_sizes=filtersizes, num_filters=mfp.num_filters)
    
    try:
        train(dcmf)
    except Exception:
        traceback.print_exc()
    
    print 'Done'