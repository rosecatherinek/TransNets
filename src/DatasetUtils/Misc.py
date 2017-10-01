'''
Created on Oct 27, 2016
Misc utilities 

@author: roseck
'''

import time, os

def int_list(int_str):
    '''
    utility fn for converting an int string to a list of int
    '''
    return [int(w) for w in int_str.split()]
    

def get_epoch_files(dir_str):
    '''
    get files of the form epochx.gz
    '''
    x = [ f for f in os.listdir(dir_str)]
    x = [ f for f in x if f.endswith('.gz') and f.startswith('epoch')]
    
    dir_str = dir_str if dir_str.endswith('/') else dir_str + '/'
    
    #get full path
    x = [ dir_str + f for f in x]
    
    return x
     

def ExecTime(startTime):
    '''
    prints the exec time from start time in hh:mm:ss
    '''
    currTime = time.time()
    elapsed = (currTime - startTime)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)



if __name__ == '__main__':
    startTime = time.time()
    print(startTime)
    
    startTime -= 60*60*3
    
    print(ExecTime(startTime))
