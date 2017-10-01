'''
Created on Jan 24, 2017

Utilities for using a dict (word -> id) to convert an input doc in words to its corresponding rep in int

Note: skipping the unknown words instead of writing 0
@date 14 Feb 2017

@author: roseck
'''
import gzip

def convert_to_int(infile, outfile, word_to_int_dict, col):
    '''
    infile: a tab separated file
    outfile: written in tab sep format
    word_to_int_dict: a dict obj with word to int mapping. unknown words are mapped to 0
    col: the col of infile to be processed
    '''
    
    UNK = '0'
    
    if(outfile.endswith('.gz')):
        fout = gzip.open(outfile, 'w')
    else:
        fout = open(outfile, 'w')
    
    
    if infile.endswith('.gz'):
        fin = gzip.open(infile, 'r')
    else:    
        fin = open(infile, 'r')
    
    for line in fin:
        vals = line.split("\t")
        txt = vals[col]
        words = txt.split()
#         int_list = [ str( word_to_int_dict[w]) if w in word_to_int_dict else UNK for w in words ]
        int_list = [ str( word_to_int_dict[w]) if w in word_to_int_dict else '' for w in words ]
        int_txt = ' '.join(int_list)
        vals[col] = int_txt
        for v in vals:
            fout.write(v)
            fout.write('\t')
        fout.write('\n')
    
    fin.close()
    fout.close()
        
        
