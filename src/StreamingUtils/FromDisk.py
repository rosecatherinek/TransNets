'''
stream from disk, records of the form: uA iB rAB UText BText 

@author: roseck
Created on Mar 11, 2017
'''

import time, gzip
from DatasetUtils import Misc

class FromDisk():
    
    def __init__(self, filename):
        '''
        filename is the dataset to iterate on
        '''
    
        #read the file
        if filename.endswith('.gz'):
            self.fin = gzip.open(filename, 'r')
        else:
            self.fin = open(filename, 'r')
        
        self.tot_batch = 0
    
    def _close(self):
        self.fin.close()
    
    
    def BatchIter(self, batch_size):    
        '''
        batch size = number of training u,b,r examples in the batch
        returns:
        uList = uA useres
        bList = iB items
        rList: rAB (float)
        user_revlist: the UText converted to int list
        item_revlist: the BText converted to int list
         
        '''
        while True:
            #one batch
            start = time.time()
            uList = []
            bList = []
            rList = []
            uTextList = []
            bTextList = []
            
            
            for line in self.fin:
                vals = line.split("\t")
                if len(vals) == 0:
                    continue
            
                u = vals[0]
                b = vals[1]
                r = float(vals[2])
                uText = vals[3]
                bText = vals[4]
                
                uList.append(u)
                bList.append(b)
                rList.append(r)
                uTextList.append(Misc.int_list(uText))
                bTextList.append(Misc.int_list(bText))
                
                
                if len(uList) >= batch_size:
                    break
            
            if len(uList) == 0:
                #end of data
                self._close()
                print 'Total Batch gen time = ', (self.tot_batch/60.0), ' min'
                raise StopIteration
            
            end = time.time()
            
            bg = (end - start)
            
            print 'Batch gen time = ', bg, ' sec'
            
            self.tot_batch += bg
            
            yield uList, bList, rList, uTextList, bTextList

 
     

    

