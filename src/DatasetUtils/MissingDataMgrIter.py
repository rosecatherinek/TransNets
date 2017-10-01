'''

given a file of the form: user<tab>biz<tab>rating<tab>review (optional)

yields batches of the form user, biz, rating, 

@author: roseck
@date Feb 28, 2017
'''
import gzip

class ReviewHelperIter():
    
    def __init__(self, review_helper, filename):
        '''
        review_helper has function of the form: get_reviews(uList, bList)
        filename is the UBR dataset to iterate on
        '''
        self.review_helper = review_helper
    
        #read the file
        if filename.endswith('.gz'):
            self.fin = gzip.open(filename, 'r')
        else:
            self.fin = open(filename, 'r')
        
    
    def update_helper(self,review_helper):
        self.review_helper = review_helper
    
    def _close(self):
        self.fin.close()
    
    def BatchIDIter(self, batch_size):    
        '''
        gives the text of only the biz reviews, for users, gives the str id
        batch size = number of training u,b,r examples in the batch
        returns: 
        user_revlist: the concat reviews by user
        item_revlist: the concat reviews for a item 
        rList: the list of corresponding ratings (float) 
        '''
        while True:
            #one batch
            uList = []
            bList = []
            rList = []
            
            for line in self.fin:
                vals = line.split("\t")
                if len(vals) == 0:
                    continue
            
                u = vals[0]
                b = vals[1]
                r = float(vals[2])
                
                uList.append(u)
                bList.append(b)
                rList.append(r)
                
                if len(uList) >= batch_size:
                    break
            
            if len(uList) == 0:
                #end of data
                self._close()
                raise StopIteration
            
            _, bRev = self.review_helper.get_reviews(uList, bList)
                
            yield uList, bRev, rList
    
            
    def BatchIter(self, batch_size):    
        '''
        batch size = number of training u,b,r examples in the batch
        returns: 
        user_revlist: the concat reviews by user
        item_revlist: the concat reviews for a item 
        rList: the list of corresponding ratings (float) 
        '''
        while True:
            #one batch
            uList = []
            bList = []
            rList = []
            
            for line in self.fin:
                vals = line.split("\t")
                if len(vals) == 0:
                    continue
            
                u = vals[0]
                b = vals[1]
                r = float(vals[2])
                
                uList.append(u)
                bList.append(b)
                rList.append(r)
                
                if len(uList) >= batch_size:
                    break
            
            if len(uList) == 0:
                #end of data
                self._close()
                raise StopIteration
            
            uRev, bRev = self.review_helper.get_reviews(uList, bList)
                
            yield uRev, bRev, rList
            
    
    def BatchFullIter(self, batch_size):    
        '''
        batch size = number of training u,b,r examples in the batch
        returns: 
        user_revlist: the concat reviews by user
        item_revlist: the concat reviews for a item 
        rList: the list of corresponding ratings (float) 
        '''
        while True:
            #one batch
            uList = []
            bList = []
            rList = []
            
            for line in self.fin:
                vals = line.split("\t")
                if len(vals) == 0:
                    continue
            
                u = vals[0]
                b = vals[1]
                r = float(vals[2])
                
                uList.append(u)
                bList.append(b)
                rList.append(r)
                
                if len(uList) >= batch_size:
                    break
            
            if len(uList) == 0:
                #end of data
                self._close()
                raise StopIteration
            
            #don't process the format of the return value
            retval = self.review_helper.get_reviews(uList, bList, rList)
                
            yield retval
            
