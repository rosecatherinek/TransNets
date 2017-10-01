'''
read the input data, parse to int list; 
create mappings of (user,item) -> review int list

@author: roseck
@date Mar 15, 2017
'''
from __builtin__ import dict
import gzip

class DataPairMgr():
    
    def _int_list(self,int_str):
        '''utility fn for converting an int string to a list of int
        '''
        return [int(w) for w in int_str.split()]
    
    def __init__(self, filename):
        '''
        filename: inits the UBRR data from the input file
        '''
        
        ub_map = dict()
        ub_ratings = dict()
        
        cnt = 0
        
        #read the file
        if filename.endswith('.gz'):
            f = gzip.open(filename, 'r')
        else:
            f = open(filename, 'r')
        
        for line in f:
            vals = line.split("\t")
            if len(vals) == 0:
                continue
            
            u = vals[0]
            b = vals[1]
            r = float(vals[2])
            d = vals[3].strip()
            
            ub_map[(u,b)] = self._int_list(d)
            ub_ratings[(u,b)] = r
            
            cnt += 1
            
        
        self.user_item_map = ub_map
        self.user_item_rating = ub_ratings
        
        
        f.close()
        print 'Data Pair Manager Initialized with ', cnt, ' reviews'
        
    def get_int_review(self, user, item):    
        if (user,item) in self.user_item_map:
            return self.user_item_map[(user,item)]
        else:
            return [0]
    
    
    def get_int_review_rating(self, user, item):    
        if (user,item) in self.user_item_map:
            return self.user_item_map[(user,item)], self.user_item_rating[(user,item)]
        else:
            return [0], 3.0  #average rating 
            

