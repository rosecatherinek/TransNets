'''
Created on Nov 29, 2016

@author: roseck
'''
UNKNOWN_TOKEN = '<UNK>'

class Vocab(object):
    '''
    Build a vocabulary with mapping to and from input text / words/ ids
    
    The unknown token is mapped to 0
    '''
    def __init__(self):
        '''dummy init method. Use the other initialization before using the class '''

    def _create(self, input_list=None):
        '''create a new vocab object using the input collection of words'''
        self._word_to_id = {}
        self._id_to_word = {}
        self._cnt = 0
        
        self.add_to_vocab([UNKNOWN_TOKEN])
        
        if input_list != None:
            self.add_to_vocab(input_list)
    
    def _load(self, dict_word_to_id, dict_id_to_word):
        '''initialize with a saved dictionary'''
        self._word_to_id = dict_word_to_id
        self._id_to_word = dict_id_to_word
        
    def add_to_vocab(self, input_list):
        for w in input_list:
            if w not in self._word_to_id:
                #Add
                self._word_to_id[w] = self._cnt
                self._id_to_word[self._cnt] = w
                self._cnt += 1
                
    def word_to_id(self, word):
        if word not in self._word_to_id:
            return 0
        return self._word_to_id[word]
    
    def id_to_word(self, wid):
        if wid not in self._id_to_word:
            raise ValueError('ID not in vocab: %d' % wid)
        return self._id_to_word[wid]
    
    def size(self):
        return len(self._word_to_id)
    
    def str_to_id(self, word_str):
        '''
        convert a string of words to a list of ids
        '''
        words = word_str.split()
        
        return [ self.word_to_id(w) for w in words]
        
        
            
                
                
                
                
                
                
                
