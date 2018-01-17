'''
Created on Jan 20, 2017

@author: roseck
'''

class Review():
    '''
    has a user, biz, rating and a review
    '''

    def __init__(self, u, b, r, d):
        self.user = u
        self.biz = b
        self.rating = r
        self.doc = d #the text of the review
        