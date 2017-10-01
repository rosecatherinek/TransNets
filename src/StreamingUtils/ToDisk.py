'''
read a UBRR file (train) into mem.

Then for the input file, for every uA iB rAB docAB entry, write to disk: uA iB rAB UText BText 
where UText and BText does not have docAB

@author: roseck
Created on Mar 10, 2017
'''

from DatasetUtils.DataMgr import DataMgr
from DatasetUtils.TestDataMeta import TestDataMeta
from DatasetUtils.MissingDataMgrIter import ReviewHelperIter

import time, sys, gzip

if __name__ == '__main__':
    
    print sys.argv
    
    dataFile = '/Users/roseck/Documents/RecoNNData/sl50/x_INT_Train.txt.gz'  #mem data for uText bText
    inFile = '/Users/roseck/Documents/RecoNNData/sl50/x_INT_Test.txt.gz'  #to generate for
    outFile = '/Users/roseck/Documents/RecoNNData/sl50/disk_INT_Test.txt.gz' #output 
    batch_size = 100
    max_len = 1000
    num_workers = 4
    timeout = 1.0 #1 sec for 30 cpus 100 batch size, larger if larger batch
    
    if len(sys.argv) > 1:
        dataFile = sys.argv[1]
        inFile = sys.argv[2]
        outFile = sys.argv[3]
        batch_size = int(sys.argv[4])
        max_len = int(sys.argv[5])
        num_workers = int(sys.argv[6])
        timeout = float(sys.argv[7])
    
    start = time.time()
    trainHelper = DataMgr(dataFile, empty_user =  '')
    end = time.time()
    print 'Data Helper init in ', (end - start), ' sec'
    
    
    fullDataMeta = TestDataMeta([trainHelper], max_len, num_workers)  # 4 workers
    
    v_train_review = ReviewHelperIter(fullDataMeta, inFile)
    train_iter = v_train_review.BatchFullIter(batch_size)    
    
    gout = gzip.open(outFile, 'w')
    nw = 0
    
    totBatchGen = 0
    totDiskWrite = 0
    c = 0
    while True:
        retUList, retBList, retRList, retUTextInt, retBTextInt = ([] for i in range(5))
        try:
            #read the values
            start = time.time()
            retUList, retBList, retRList, retUTextInt, retBTextInt = train_iter.next()
            end = time.time()
            genTime = end - start
            print 'Batch gen time ', genTime, ' sec'
            totBatchGen += genTime
            c += 1
            if c % 100 == 0:
                print '(', c, ')' 
                sys.stdout.flush()
            
            #write to disk
            start= time.time()
            for u,b, r, ut, bt in zip(retUList, retBList, retRList, retUTextInt, retBTextInt):
                #ut & bt are int arrays
                u_text = [str(x) for x in ut]
                b_text = [str(x) for x in bt]
                
                u_text = ' '.join(u_text)
                b_text = ' '.join(b_text)
                
                gout.write(u)
                gout.write('\t')
                gout.write(b)
                gout.write('\t')
                gout.write(str(r))
                gout.write('\t') 
                gout.write(u_text)
                gout.write('\t')
                gout.write(b_text)
                
                gout.write('\n')
                nw +=1
            
            end = time.time()
            dTime = (end - start)
            print 'disk write time', (end - start), ' sec'
            totDiskWrite += dTime
            
            if genTime > timeout:
                #time to gen > 1 sec -> thrashing
                #restart the meta data proc
                print 'restarting Test Meta Handler'
                start = time.time()
                fullDataMeta.close()
                end = time.time()
                print 'earlier handler closed in ', (end - start), ' sec'
                del fullDataMeta
                start = time.time()
                fullDataMeta = TestDataMeta([trainHelper], max_len, num_workers)  # 4 workers
                v_train_review.update_helper(fullDataMeta)
                end = time.time()
                print 'new handler started in ', (end - start), ' sec'
            
        except StopIteration:
            #end of data
            break
    totend = time.time()
    
    fullDataMeta.close()
    print 'Total records written ', nw
    print 'Total Batch gen time ', totBatchGen/60.0, ' min'
    print 'Total Disk write time ', totDiskWrite/60.0, ' min'
    
    
    
    
     

    
