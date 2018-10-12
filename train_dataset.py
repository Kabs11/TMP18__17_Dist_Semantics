#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Load and preprocess a corpus for idiom extraction'''

import os, time, json
import nltk.data
from bs4 import BeautifulSoup

def plain_text(corpus_file, no_split):
	'''Read in a plain text corpus, return a single document containing a list of unicode sentences.'''	

	splitter = nltk.data.load('tokenizers/punkt/english.pickle')
	# Read in corpus
	documents = []
	sentences = []
	with open(corpus_file, 'r', errors='ignore') as f:
		for line in f:
			if line.strip():
				if no_split:
					sentences.append(line.strip())
				else:
					sentences += splitter.tokenize(line.strip())
	documents.append(sentences)
	
	return documents
