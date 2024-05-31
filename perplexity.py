############################################################################################
'''
Problem Statement 1: Sort the sentences from this file in ascending order in terms of number of characters in each line. Also, find the following:
- Total number of words
Total number of unique words
Number of unique characters
Problem Statement 2: From the given 21260 sentences, train a language model (LM) using randomly selected 20760 sentences. Compute perplexity on the 500 sentences and find 100 sentences with lowest perplexity. You may use nltk/kenlm or any tool to train the language model. Share the code for training LM, finding perplexity and how the 100 sentences are chosen. Also share the perplexity of 100 sentences. 
'''
#############################################################################################
import random
from kenlm import LanguageModel
import os
import re
import subprocess
import argparse
import gzip
import io
import os
import subprocess
import itertools
from collections import Counter
from tqdm import tqdm
from nltk.corpus import indian
from nltk.tag import tnt
import string
from joblib import Parallel, delayed

def dataload(f_path_data):
	with open(f_path_data, 'r', encoding='utf-8') as file:
	    all_sentences = file.read()
	symbols_to_remove = r"[৫০\)?,:–,♦,(,\[,\],👉,💓,{,},.,★]+"
	all_sentences = re.sub(symbols_to_remove, " ", all_sentences)
	number_pattern = r"\b\d+\b"
	url_pattern = r"https?://\S+?\.\w+"
	all_sentences = re.sub(number_pattern, "", all_sentences)
	all_sentences = re.sub(url_pattern, "", all_sentences)
	pattern = r'[A-Za-z\\{}]+'
	cleaned_string = re.sub(pattern, '', all_sentences)
	delimiters = r"[\|।\n]"
	sentences = re.split(delimiters, all_sentences)
	sentences = [sentence.strip() for sentence in sentences if sentence.strip() and len(sentence) > 10]
	return sentences

def remove_english(text):
	# Define a regex pattern to match English words
	# Filter out English parts from sentences
	# Join the remaining sentences back into a paragraph
	english_pattern = re.compile(r'[A-Za-z0-9]+')
	filtered_sentence = ' '.join(word for word in text.split() if not english_pattern.search(word))
	return filtered_sentence


def EDA(f_path_data):
	try:
		lines=dataload(f_path_data)
		words = list(itertools.chain(*[line.split() for line in tqdm(lines)]))
		counter = Counter(words)
		total_words = sum(counter.values()) # Total number of words
		uniqe_words = len(counter) # Total number of unique words
		unique_characters = len(set("".join(counter.keys()))) #Number of unique characters
		"""Sort the sentences from this file in ascending order in terms of number of characters in each line"""
		sorted_sentences = sorted(lines, key=lambda s: len(s.split()))
		print("\n TOTAL WORDS : ",total_words,"\n UNIQUE WORDS : ",uniqe_words,"\n UNIQUE CHARACTERS : ",unique_characters)
	except:
		print(f'Error while getting no. of unique/Total numbers of words or characters')	

def data_prep_for_train(f_path_data):
	"""prepares data for training by loading sentences from a file, splitting them into training and test sets, and saving them in separate text files."""
	try:
		sentences=dataload(f_path_data)
		train_sentences = random.sample(sentences, 20760)
		remaining_sentences = list(set(sentences) - set(train_sentences))
		test_sentences = random.sample(remaining_sentences, 500)
		with open("train.txt","w",encoding="utf-8") as f:
			f.write("।\n".join(train_sentences))
			f.close()
		with open("test.txt","w",encoding="utf-8") as f:
			f.write("।\n".join(test_sentences))
			f.close()
	except:
		print(f"Error with training the LM!! Please check the text file along with train.sh script")

def data_prep_for_clean_train(f_path_data):
"""It tokenizes sentences from the file, removes English sentences, shuffles and splits data into training and testing sets (500 samples for testing).
Additionally, it merges training data with some clean POS-tagged data from an NLTK corpus, writes the prepared data into separate text files (train.txt and test.txt)"""
	try:
		word_set = indian.sents(f_path_data)
		final=[]
		for sen in word_set:
		    sen = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in sen]).strip()
		    sen = remove_english(sen)
		    if sen==" ":
		    	pass
		    else:
		    	if sen not in final:
		    		final.append(sen)
		random.shuffle(final)
		test_size = 500
		test_data = final[:test_size]
		train_data = final[test_size:]
		# Merging train data with some POS tagged clean data from nltk corpus
		tagged_set = 'bangla.pos'
		word_set = indian.sents(tagged_set)
		for sen in word_set:
		    sen = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in sen]).strip()
		    if sen not in train_data:
		    	train_data.append(sen)    
		with open("train.txt","w",encoding="utf-8") as f:
			f.write("।\n".join(train_data))
			f.close()
		with open("test.txt","w",encoding="utf-8") as f:
			f.write("।\n".join(test_data))
			f.close()
	except:
		print(f"Error with data_prep_for_clean_train() Please check the text file along with data_prep_for_clean_train method in script")


def train(f_path_data):
	try:
		command = ["sh", "./train.sh", './train.txt', "./"]
		try:
			result = subprocess.run(command, capture_output=True, text=True)
			output = result.stdout
			print(output)
		except subprocess.CalledProcessError as e:
			print(f"Error running train.sh: {e}")
	except:
		print(f"Error with training the LM!! Please check the text file along with train.sh script")


def perplexity_task(test_sentences):
	try:
		if os.path.exists('./lm.binary'):
			lm = LanguageModel('lm.binary')
			perplexities = []
			for sentence in test_sentences:
			    perplexity = lm.perplexity(sentence)
			    perplexities.append((sentence, perplexity))
			sorted_perplexities = sorted(perplexities, key=lambda x: x[1])
			lowest_perplexity_sentences = [sentence for sentence, perplexity in sorted_perplexities[:100]]
			for sentence, perplexity in sorted_perplexities[:100]:
			    print(f"Perplexity: {perplexity}\n")
			lowest_100_sent="Selected 100 Sentences with Lowest Perplexity:"
			for sentence in lowest_perplexity_sentences:
			    lowest_100_sent=lowest_100_sent+"\n"+sentence
			with open("final_output.txt",'w') as f:
				f.write(lowest_100_sent)
		else:
			print("Train the LM first")
			exit()
	except:
		print(f"Error with calculating Perplexity")
with open("test.txt","r",encoding="utf-8") as f:
	D=f.read()

parser = argparse.ArgumentParser(description='perplexity computation')
parser.add_argument('--in_txt', help='input file path')
args = parser.parse_args()
f_path_data=args.in_txt
EDA(f_path_data)
data_prep_for_train(f_path_data)
#data_prep_for_clean_train(f_path_data)
train(f_path_data)   
D=D.split("\n")
print(len(D))
perplexity_task(D)
print(" Output saved into file name final_output.txt")

