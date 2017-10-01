'''
A meta level review helper: has a list of review helpers internally to choose data from

all review helpers are treated like test helpers (including the train)
for all requests, revAB is skipped when data for (userA, itemB) is requested

@author: roseck
@date mar 9, 2017
'''
from random import shuffle
import multiprocessing
from DatasetUtils.DataMgr import DataMgr

class TestDataMeta():
    
    def __init__(self, test_helpers, max_len, num_workers):
        '''
        test helpers: a list of DataMgr objs
        max_len: approx max len (words/ids) of the merged review
        num_workers: set to max num cpus
        '''

        self.test_helpers = test_helpers
        self.max_len = max_len
        self.num_workers = num_workers
        
        print 'Meta Review Helper Initialized with ', len(test_helpers), ' Data Helpers'
        
        self.processes = []
        self.inQueue = multiprocessing.Queue()
        self.outQueue = multiprocessing.Queue()
        
        #start the workers
        for _ in range(self.num_workers):
            p = multiprocessing.Process(target=self.rev_from_dmgr, args=(self.inQueue, self.outQueue))
            p.start()
            self.processes.append(p)
        
        print 'multi processor ready with ',  len(self.processes), ' worker threads'
        print "don't forget to call close"
        
    def close(self):
        #terminate the workers
        for p in self.processes:
            p.terminate()
        
        print 'terminated ', len(self.processes), ' workers'
        
    def _int_list(self,int_str):
        '''utility fn for converting an int string to a list of int
        '''
        return [int(w) for w in int_str.split()]
    
    def _shuffle(self, inlist):
        shuffle(inlist)
        return inlist
            
    def rev_from_dmgr(self, inQueue, outQueue):
        '''
        process a single (u,b) tuple
        return: for each u,b, int list rep of rev by u, int list rep of rev for b, except rev(u,b)
        output written to outQueue
        '''
        
        for (u,b,r) in iter(inQueue.get, 'STOP'):
            
            uR = []
            bR = []
            for t in self.test_helpers:
                umap = t.user_map
                bmap = t.biz_map
        
                uR.extend(umap[u] if u in umap else [])
                bR.extend(bmap[b] if b in bmap else [])
                
            #done_queue.put("%s - %s got %s." % (current_process().name, url, status_code))

        
            #remove reviews of the (ub) pair
            uRev = [x for x in uR if x.biz != b]
            bRev = [x for x in bR if x.user != u] 
        
            #shuffle each
            shuffle(uRev)
            shuffle(bRev)
            uShuff = uRev[:min(len(uRev), self.max_len)]
            bShuff = bRev[:min(len(bRev), self.max_len)] 
            
            #get the docs
            uRevDoc = [x.doc for x in uShuff]
            bRevDoc = [x.doc for x in bShuff]
            
            #join the docs and convert to int
            uRevInt = self._int_list(' '.join(uRevDoc))
            bRevInt = self._int_list(' '.join(bRevDoc))
            
            uRevIntMax = uRevInt[: min(self.max_len, len(uRevInt))]
            bRevIntMax = bRevInt[: min(self.max_len, len(bRevInt))]

            
            #save
            outQueue.put( (u, b, r, uRevIntMax, bRevIntMax))
        
        return True
        
        

    def get_reviews(self, uList, bList, rList):    
        '''
        given the list of users and list of biz, return the list of reviews by each user and for each item
        the order of uList & bList is different in the returned value
        '''
        
        #feed the inQueue
        for u,b,r in zip(uList, bList, rList):
            self.inQueue.put((u,b,r))
            
        
        retUList = []
        retBList = []
        retRList = []
        retUTextInt = []
        retBTextInt = []
        #get the outputs
        for _ in range(len(uList)):
            u, b, r, uRevIntMax, bRevIntMax = self.outQueue.get()
            retUList.append(u)
            retBList.append(b)
            retRList.append(r)
            retUTextInt.append(uRevIntMax)
            retBTextInt.append(bRevIntMax)
        
        return retUList, retBList, retRList, retUTextInt, retBTextInt
    
if __name__ == '__main__':
    
    traindata = '/Users/roseck/Documents/RecoNNData/sl50/x_INT_Train.txt.gz'
    testdata = '/Users/roseck/Documents/RecoNNData/sl50/x_INT_Test.txt.gz'
    
    trainHelper = DataMgr(traindata, empty_user =  '')
    testHelper = DataMgr(testdata, empty_user =  '')
    
    fullDataMeta = TestDataMeta([trainHelper, testHelper], 1000, 4)  #all data
    
    uList = ['MrPOzC4Fz_xEwJkA051V8g', 'eBFm-lABQiKpaUcPDfYOgA']
    bList = ['KPoTixdjoJxSqRSEApSAGg', 'KPoTixdjoJxSqRSEApSAGg']
    rList = [4,3]
    
    retUList, retBList, retRList, retUTextInt, retBTextInt = fullDataMeta.get_reviews(uList, bList, rList)
    
    for u,b, r, uR, bR in zip(retUList, retBList, retRList, retUTextInt, retBTextInt):
        print u,b,r, uR, bR
    
      
