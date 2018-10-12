#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Utility functions to work with morpha, PoS-tagging, parsing, and other things.'''

import subprocess, shlex, time, json, re, itertools, csv
import spacy
import en_core_web_sm as spacy_model
from stanfordcorenlp import StanfordCoreNLP
import nltk.data

###### STANFORD TO SPACY ######
class StanfordDoc:
	'''Spacy-Doc-like container for Stanford output'''

	def __init__(self):
		self.sents = []

	def __iter__(self):
		return iter(self.tokens)

	def __getitem__(self, i):
		if isinstance(i, slice):
			return StanfordSpan(self.tokens[i.start:i.stop])
		else:
			return self.tokens[i]

	# Generate list of tokens from sentences
	def set_tokens(self):
		self.tokens = [token for sent in self.sents for token in sent]

class StanfordSpan:
	'''Spacy-Span-like container for Stanford output'''

	def __init__(self, tokens):
		self.tokens = tokens
		self.start = self.tokens[0].i # Starting token index in document
		self.start_char = self.tokens[0].idx # Starting character index in document
		self.text_with_ws = ''.join([token.text_with_ws for token in self.tokens])
		self.text = ''.join([token.text_with_ws for token in self.tokens[:-1]]) + self.tokens[-1].text

	def __iter__(self):
		return iter(self.tokens)

	def __getitem__(self, i):
		return self.tokens[i]

class StanfordToken:
	'''Spacy-Token-like container for Stanford output'''

	def __init__(self, i, idx, lemma, tag, text, ws, word, doc):
		self.i = i # Token index in document
		self.idx = idx # Starting character index in document
		self.lemma_ = lemma
		self.tag_ = tag # PoS-tag inventory might differ slightly, but should not cause problems
		self.text = text
		self.text_with_ws = text + ws
		self.lower_ = word.lower()
		self.children = []
		self.doc = doc

	def __str__(self):
		return self.text

	# Recursively gets all the syntactic descendants of a token, including self
	def get_descendants(self):
		descendants = [self]
		for child in self.children:
			descendants += child.get_descendants()
		return descendants

	# Sets the subtree attribute, which is an ordered generator for all descendants of a token
	def get_subtree(self):
		return sorted(self.get_descendants(), key=lambda x: x.i)

	# Sets the rights attribute, which is an ordered generator for all children to the right of a token
	def get_rights(self):
		return [child for child in self.children if child.i > self.i]

	def __repr__(self):
		return self.text
###### POS-TAGGING ######
def load_pos_tagger():
	'''Loads Spacy PoS-tagger which takes pre-tokenized text.'''
	
	time_0 = time.time()
	print('Loading PoS-tagger...')
	pos_tagger = spacy_model.load(disable=['ner', 'parser'])
	print('Done! Loading PoS-tagger took {0:.2f} seconds'.format(time.time() - time_0))

	return pos_tagger

def pos_tag(pos_tagger, text):
	'''Takes pos_tagger and tokenized utf-8 idiom/sentence, returns list of word|POS strings.'''
	
	# Normalize quotes, ‘ ’ ❛ ❜ to ', and “ ” ❝ ❞ to ", Spacy doesn't process them well
	text = re.sub(u'‘|’|❛|❜', u"'", text)
	text = re.sub(u'“|”|❝|❞', u'"', text)
	# Make Doc
	doc = spacy.tokens.Doc(pos_tagger.vocab, text.split())
	# Set sentence boundary
	for token in doc:
		if token.i == 0:
			token.is_sent_start = True
		else:
			token.is_sent_start = False
	# Do actual tagging
	doc = pos_tagger.tagger(doc)
	# Convert into list of words and tags
	words_and_tags = []
	for token in doc:
		words_and_tags.append(token.text + u'|' + token.tag_)
		
	return words_and_tags

###### TOKENIZATION ######
def load_tokenizer():
	'''Loads Spacy tokenizer'''

	time_0 = time.time()
	print ('Loading tokenizer...')
	tokenizer = spacy_model.load(disable = ['tagger', 'ner', 'parser'])
	print ('Done! Loading tokenizer took {0:.2f} seconds'.format(time.time() - time_0))

	return tokenizer

def tokenize(tokenizer, sentence):
	'''Parses a (unicode) sentence, returns list of Spacy Tokens'''
	try:
		return tokenizer(sentence, 'utf-8')
	except TypeError:
		return tokenizer(sentence)

###### EXAMPLE SENTENCES ######
def get_example_sentences(idioms, sentences_file, cache_file):
	'''
	Takes a list of idioms, searches a large corpus for example sentences,
	extracts shortest example sentence, returns dict of format {idiom: sentence}.
	Saves extracted sentences and idioms to file, for fast re-use in subsequent runs. 
	'''

	time_0 = time.time()
	idioms_with_sentences = {}

	# If file is cached example sentences, load those, else extract sentences from corpus
	if re.search('.json$', sentences_file):
		idioms_with_sentences = json.load(open(sentences_file, 'r'))
		if set(idioms) <= set(idioms_with_sentences.keys()):
			print ('Using cached example sentences from {0}').format(sentences_file)
			# Select only the idioms part of the idiom dictionary 
			if set(idioms) < set(idioms_with_sentences.keys()):
				idioms_with_sentences = {key: idioms_with_sentences[key] for key in idioms_with_sentences if key in idioms}
			return idioms_with_sentences
		else:
			raise Exception('{0} does not contain entries for all the idioms specified in the dictionary argument, quitting.'.format(sentences_file))
	else:
		print ('{0} is not a cached json-file, extracting sentences containing idioms...').format(sentences_file)

	# Add fallback option: no example sentence
	for idiom in idioms:
		idioms_with_sentences[idiom] = '' 
	# Compile idiom regexes for efficiency and ignore meta-linguistic uses in quotes
	idiom_regexes = [re.compile('[^"\'] ' + idiom + ' [^"\']') for idiom in idioms]
	# Find shortest (in tokens) sentence containing idiom in corpus
	splitter = nltk.data.load('tokenizers/punkt/english.pickle')
	# Extract first 1000 lines containing the idiom with grep, then split and find sentences
	for idx, idiom in enumerate(idioms):
		if idx%100 == 0 and idx > 0:
			print ('\tGetting example sentences for {0} of {1} idioms took {2} seconds').format(idx, len(idioms), time.time()-time_0)
		call = shlex.split('grep -m 1000 "{0}" {1}'.format(u8(idiom), sentences_file))
		process = subprocess.Popen(call, stdin=subprocess.PIPE, stdout = subprocess.PIPE, stderr=subprocess.PIPE)
		output = process.communicate()
		output = output[0].strip()
		sentences = splitter.tokenize(output, 'utf-8')
		for sentence in sentences:
			if idiom_regexes[idx].search(sentence):
				# Should have at least 3 extra words in the 'sentence'
				if len(sentence.split(' ')) > len(idiom.split(' ')) + 3:
					if idioms_with_sentences[idiom]:
						# Replace old sentence if new sentence one is shorter
						if len(sentence.split(' ')) < len(idioms_with_sentences[idiom].split(' ')): 
							idioms_with_sentences[idiom] = sentence
					else:
						idioms_with_sentences[idiom] = sentence

	# Caching extracted example sentences
	ofn = cache_file
	with open(ofn, 'w') as of:
		json.dump(idioms_with_sentences, of)
		print ('Caching idioms and example sentences in {0}').format(ofn)

	print ('Done! took {0:.2f} seconds').format(time.time() - time_0)

	return idioms_with_sentences

def expand_indefinite_pronouns(idioms):
	'''
	When one's or someone's or someone occurs in an idiom, remove it,
	and add idioms with personal pronouns added in. Don't expand 'one',
	because it is too ambiguous.
	'''

	expanded_idioms = []
	base_form_map = {} # Maps expanded variants to base form, format: {'expanded idiom': 'base form'}
	possessive_pronouns = ['my', 'your', 'his', 'her', 'its', 'our', 'their']
	objective_pronouns = ['me', 'you', 'him', 'her', 'us', 'them', 'it']

	for idiom in idioms:
		# Add possessive pronouns only
		if re.search("one's: ", idiom):
			for possessive_pronoun in possessive_pronouns:
				expanded_idiom = re.sub("one's: ", possessive_pronoun, idiom)
				expanded_idioms.append(expanded_idiom)
				base_form_map[expanded_idiom] = idiom
		# Add possessive pronouns and a wildcard for other words
		elif re.search("someone's: ", idiom):
			for possessive_pronoun in possessive_pronouns + ["—'s", 'utf-8']:
				expanded_idiom = re.sub("someone's: ", possessive_pronoun, idiom)
				expanded_idioms.append(expanded_idiom)
				base_form_map[expanded_idiom] = idiom
		# Add objective pronouns and a wildcard for other words
		elif re.search("someone: ", idiom):
			for objective_pronoun in objective_pronouns + ["—", 'utf-8']:
				expanded_idiom = re.sub("someone: ", objective_pronoun, idiom)
				expanded_idioms.append(expanded_idiom)
				base_form_map[expanded_idiom] = idiom
		else: 
			expanded_idioms.append(idiom)
			base_form_map[idiom] = idiom

	return expanded_idioms, base_form_map

###### OUTPUT ######
def u8(u):
	'''Encode unicode string in utf-8.'''

	return u.encode('utf-8')
	
def write_csv(extracted_idioms, outfile):
	'''Writes extracted idioms to file in csv-format'''

	with open(outfile, 'w') as of:
		writer = csv.writer(of, delimiter='\t', quoting=csv.QUOTE_MINIMAL, quotechar='"')
		for extracted_idiom in extracted_idioms:
			output_row = [u8(extracted_idiom['idiom']), extracted_idiom['start'], extracted_idiom['end'],
				u8(extracted_idiom['snippet']), u8(extracted_idiom['bnc_document_id']), u8(extracted_idiom['bnc_sentence']),
				extracted_idiom['bnc_char_start'], extracted_idiom['bnc_char_end']]
			writer.writerow(output_row)
			print("Hey Your file has been stored")
