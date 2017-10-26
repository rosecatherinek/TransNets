'''
Created on Oct 12, 2016

@author: roseck
'''

import os.path

def readFileToSet(filename):
    '''
    read the input file into a set
    '''
    return set(line.strip() for line in open(filename))

def readFileToList(filename):
    '''
    read the input file into a list
    '''
    return [line.strip() for line in open(filename)]


def ratingsReviewReader(filename):
    '''
    read the ratings and the corresponding reviews
    '''
    rating = []
    reviews = []
    if os.path.exists(filename):
        for line in open(filename):
            vals = line.split("\t")
            rating.append(int(vals[0]))
            reviews.append(vals[1].strip())
    
    
    return rating, reviews

def TabReader(filename):
    '''
    yield each of the values
    '''
    if os.path.exists(filename):
        for line in open(filename):
            vals = line.split("\t")
            yield vals
    
    


