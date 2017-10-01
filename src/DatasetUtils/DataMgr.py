'''
read the input data, parse to int list; 
create mappings of user -> reviews, item -> reviews

The companion iterator reads thru the input file sequentially, yielding the data of the form: user word id list, item word id list, rating (float)

@author: roseck
@date Feb 28, 2017
'''
from __builtin__ import dict
import gzip
from DatasetUtils.Review import Review

class DataMgr():
    
    def _int_list(self,int_str):
        '''utility fn for converting an int string to a list of int
        '''
        return [int(w) for w in int_str.split()]
    
    def __init__(self, filename, empty_user = set()):
        '''
        filename: inits the UBRR data from the input file
        empty_user: skip the reviews by this user (keeps the ratings)
        '''
        self.empty_user = empty_user
        
        ur_map = dict()
        br_map = dict()
        
        cnt = 0
        skipped = 0
        
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
            if u in self.empty_user:
                #we are skipping this review
                d = ''
                skipped += 1
            
            rev = Review(u, b, r, d)  #review obj
            
            
            #store biz -> list of reviews
            if not br_map.has_key(b):
                br_map[b] = []
            
            br_map[b].append(rev)
            
            #store user -> list of reviews
            if not ur_map.has_key(u):
                ur_map[u] = []
                
            ur_map[u].append(rev)
            
            cnt += 1
            
        
        self.biz_map = br_map
        self.user_map = ur_map
        
        
        f.close()
        print 'Review Data Manager Initialized with ', cnt, ' reviews'
        print 'Number of skipped users = ', len(self.empty_user)
        print 'Number of skipped reviews = ', skipped
        
        
            
