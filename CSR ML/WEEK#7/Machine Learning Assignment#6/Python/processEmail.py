# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 23:00:36 2018

@author: Mohammad Wasil Saleem
"""

import csv
from stemming.porter2 import stem
import numpy as np

import re

'''
PS:- Install stemming algorithm, by typing the below in prompt.
pip install stemming==1.0.1
'''

def GetVocabList():
    vocabulary={}
    with open(r'D:\ML\ML\CSR ML\WEEK#7\Machine Learning Assignment#6\Python\vocab.txt', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', skipinitialspace=True)
        #print(spamreader)
        for row in spamreader:
            vocabulary[row[1]]=int(row[0])
            
    return vocabulary                   

def ProcessEmail(emailContent):
    # PROCESSEMAIL preprocesses a the body of an email and
    # returns a list of word_indices 
    vocabList = GetVocabList()
    emailContent=emailContent.lower()
    #https://docs.python.org/2/library/re.html
    #https://stackoverflow.com/questions/3351485/how-to-remove-all-html-tags-from-downloaded-page
    emailContent = re.sub(r'<[^<>]+>', " ", emailContent)
    #https://stackoverflow.com/questions/12851791/removing-numbers-from-string
    emailContent = re.sub(r'[0-9]+', 'number', emailContent)
    #https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
    emailContent = re.sub(r'(http|https)://[^\s]*', 'httpaddr', emailContent, flags=re.MULTILINE)
    #https://stackoverflow.com/questions/17681670/extract-email-sub-strings-from-large-document
    emailContent = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', emailContent)
    emailContent = re.sub(r'[$]+', 'dollar', emailContent)
    
    #https://stackoverflow.com/questions/1276764/stripping-everything-but-alphanumeric-chars-from-a-string-in-python
    #emailContent = re.sub(r'\W+', '', emailContent)
    
    documents = [stem(word) for word in re.findall(r"\w+", emailContent)]   
    print(" ".join(documents))
    
    #word_indices=np.zeros((len(documents),1))
    word_indices=[0]*len(documents)
    
    #https://stackoverflow.com/questions/14948900/strncmp-in-python
    for i,word in enumerate(documents,0):
        if word in vocabList:
            word_indices[i]=[vocabList[word]]
            #print('%d,%d'%(i,vocabList[word]))

    word_indices=[x for x in word_indices if x != 0 ]

    return word_indices
    
