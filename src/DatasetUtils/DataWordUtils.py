'''
Utilities for words / embeddings
Source: https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

Created on Oct 12, 2016
'''
import collections
import re

SIZE_VOCABULARY = 5000

def PreProc(text):
    '''
    1. lowercase
    2. remove non-alpha numeric chars
    
    @text - input text
    '''
    return [ w for w in re.sub('[^0-9a-zA-Z_]+', ' ', text.lower()).split()] 
    

def build_dataset(docs):
    '''
    Build the dictionary and replace rare words with UNK token.
    @docs : list of strings - already preprocessed
    '''
    words = [wd for doc in docs for wd in PreProc(doc)]
    
    count = [['UNK', -1]]
    
    count.extend(collections.Counter(words).most_common(SIZE_VOCABULARY - 1))
    
    dictionary = dict()
    
    for word, _ in count:
        dictionary[word] = len(dictionary)
    
    intdocs = []
    unk_count = 0
    
    for d in docs:
        wid = []
        for w in d.split():
            if w in dictionary:
                index = dictionary[w]
            else:
                    index = 0  # dictionary['UNK']
                    unk_count += 1
            wid.append(index)
        intdocs.append(wid)
        
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    del words
    
    return intdocs, count, dictionary, reverse_dictionary


def build_dataset_streaming(docsIter, user=False):
    '''
    Build the dictionary and replace rare words with UNK token.
    @docsIter : a generator that gives the next doc; 
    for now, we will use the IterUtils.__FileDataIter(fileList) that reads the labels and reviews
    if user is True, then the output of the generator is expected to be user, label, doc : output of __UserFileDataIter
    '''
    words = []
    while True:
        try:
            if user:
                u, _, doc = docsIter.next()
                #prefix the username to wd
                dwds = [u + wd for wd in PreProc(doc)]
            else:
                _, doc = docsIter.next()
                dwds = [wd for wd in PreProc(doc)]
                
            words.extend(dwds)
        except StopIteration:
            break 
    
    count = [['UNK', -1]]
    
    count.extend(collections.Counter(words).most_common(SIZE_VOCABULARY - 1))
    
    dictionary = dict()
    
    for word, _ in count:
        dictionary[word] = len(dictionary)
    
    
    del words
    
    return dictionary


def ToIntDoc(doc, dictionary):
    '''
    Given a dictionary, convert the doc to its int representation
    '''
    wid = []
    for w in PreProc(doc):
        if w in dictionary:
            index = dictionary[w]
        else:
                index = 0  # dictionary['UNK']
        wid.append(index)
        
    return wid

def ToUserIntDoc(usr, doc, dictionary):
    '''
    Given a dictionary, convert the doc to its int representation
    In this, the words are prefixed with u
    '''
    wid = []
    for w in PreProc(doc):
        uw = usr + w
        if uw in dictionary:
            index = dictionary[uw]
        else:
                index = 0  # dictionary['UNK']
        wid.append(index)
        
    return wid

def ChopOrPadLists(docs, maxlen = 1000, filler = 0):
    '''
    chop rows in the input that are longer than maxlen
    and pad the rows in the input that are shorter than maxlen with the filler
    input:
    @docs list of lists
    '''
    #chop first
    docs = [item[:maxlen] for item in docs]
    #pad
    docs = [item + [filler] * (maxlen - len(item)) for item in docs]
    return docs
    
    
