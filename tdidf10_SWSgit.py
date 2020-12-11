from __future__ import division, unicode_literals		 
## Sheryl Winston Smith, BI Norwegian Business School
# Oslo 2020
import math
import nltk
import string
import re
from nltk.corpus import stopwords	#IMPORT STOPWORDS CORPUS (have to get into lowercase)
from textblob import Word
import codecs
codecs.open
import io
io.open 
import glob
import os
from text.blob import TextBlob as tb
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

print "Text analysis: SWS 2020"
print "============================="
print "Text files in directory:"
flist=[]
for file in glob.glob("data/*.txt"):
	print file
	flist.append(file)
	print flist
path = '/Users/sws/textmatch/data'
for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        shakes = open(file_path, 'r')
        text = shakes.read()
        lowers = text.lower()
        #no_punctuation = lowers.translate(None, string.punctuation)
        #token_dict[file] = no_punctuation
print "+++++++++++++++++++++++++"
# ++++++++++++++++++++++++++
# Text processing functions
# ++++++++++++++++++++++++++
def wf(word, blob):									# compute wf-# of times word appears in a document blob
    return blob.words.count(word) 					# not normalized by # words in blob 
def tf(word, blob):									# compute tf-# of times word appears in a document blob
    return blob.words.count(word) / len(blob.words)  # normalize by # words in blob 
def n_containing(word, bloblist):					#return # documents containing word
    return sum(1 for blob in bloblist if word in blob)
def idf(word, bloblist):							# compute idf (how common word is in bloblist)
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist))) #log of # docs in corpus(bloblist)/ # docs where word appears
def tfidf(word, blob, bloblist):					# compute tf*idf 
    return tf(word, blob) * idf(word, bloblist)
def get_cosine(blob1, blob2, bloblist):
     s1 = {word: tfidf(word, blob, bloblist) for word in blob1.filter}
     s2 = {word: tfidf(word, blob, bloblist) for word in blob2.filter}
     intersection = set(s1.keys()) & set(s2.keys())
     print "Computing cosine similarity:"
     print intersection
     numerator = sum([s1[x] * s2[x] for x in intersection])
     sum1 = sum([s1[x]**2 for x in s1.keys()])
     sum2 = sum([s2[x]**2 for x in s2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)
     print("Numerator: {} Denominator: {}".format(numerator,denominator))
     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator
print "All stopwords:"
print stopwords.words("english")
print "===================="
print "++++++++++++++"
# create TEST DOCS TO SEE WHAT IS GOING ON
print "++++++++++++++"
filelist = ["testabstract", "document5", "document6"]
for doc in filelist:
	print doc+".txt"
# ++++++++++++++++++++++++++
# Open and pre-process files
# ++++++++++++++++++++++++++
with open ("testabstract.txt", "r") as myfile:
	document4=myfile.read().replace('\n','')
document4=''.join(i for i in document4 if not i.isdigit())
document4=''.join([x for x in document4 if ((ord(x)==32) or (47 <= ord(x) < 127))])	# remove non-ascii characters
document4=re.sub(r'\s\d+',' ', document4)	# substitute matching regexp space w/digit(s)
document4=re.sub(r'[^\x00-\xF5]',' ', document4)				# substitute matching regexp w/space
document4=document4.replace("'s", '').lower()					#make lower case
document4=tb(document4) 
 
with open ("testintro.txt", "r") as myfile:
	document5=myfile.read().replace('\n','')
document5=''.join(i for i in document5 if not i.isdigit())
document5=''.join([x for x in document5 if ((ord(x)==32) or (47 <= ord(x) < 127))])	# remove non-ascii characters
document5=re.sub(r'\s\d+',' ', document5)	# substitute matching regexp space w/digit(s)
document5=re.sub(r'[^\x00-\xF5]',' ', document5)				# substitute matching regexp w/space
document5=document5.replace("'s", '').lower()
document5=tb(document5) 

with open ("testconclusion.txt", "r") as myfile:
	document6=myfile.read()
#document6=''.join(i for i in document6 if not i.isdigit())
document6=''.join([x for x in document6 if ((ord(x)==32) or (47 <= ord(x) < 127))])	# remove non-ascii characters
document6=re.sub(r'\s\d+',' ', document6)	# substitute matching regexp space w/digit(s)
document6=re.sub(r'[^\x00-\xF5]',' ', document6)				# substitute matching regexp w/space
document6=document6.replace("'s", '').lower()
document6= document6.replace('\n','')
document6=tb(document6) 

print "++++++++++++++"
# Here is where we get external IPO prospectus and patent files
print "++++++++++++++"
  
# GOING THROUGH EACH TXT FILE 
bloblist = [document4, document5, document6]
blobtext=["document4", "document5", "document6"]
# PROCESS TEXT: REMOVE STOPWORDS AND LEMMATIZE BEFORE TFIDF
for i, blob in enumerate(bloblist):
	
	print "Processing:"
	print("=====\nblob words in document {}\n=====".format(i + 1))	
	print blob.words	
	# print i
	print 
	#blob=blob.lower()		# CONVERT EVERYTHING TO LOWERCASE
	#blob=blob.rstrip("\'s")
	#singular = [w.singularize() for w in blob.words]	# singularize
	#blob.words=singular
	#lemmap = [w.lemmatize() for w in blob.words]	# LEMMATIZE
	#blob.words=lemmap
	filterp = [w for w in blob.words if not w in stopwords.words("english")] #REMOVE STOPWORDS
	#blob=filterp #Problems here because blob.words no longer exists
	print "Remove stopwords..."
	print filterp
	blob.filter=filterp
	lemmap = [w.lemmatize() for w in blob.filter]	# LEMMATIZE
	print "Lemmatize..."
	blob.filter=lemmap
	print blob.filter	
	#removeapost = [w.rstrip("\'s") for w in blob.words] 
	#blob.words=removeapost
	#print "Blob words are:"	
#	print blob.words
# ++++++++++++++++++++++++++++++++
# COMPUTE SCORES ON PROCESSED TEXT
# ++++++++++++++++++++++++++++++++

for i, blob in enumerate(bloblist):		
    print("=====\nTop words in document {}\n=====".format(i + 1))			#store TF-IDF scores in a dictionary (scores) 
    nword=len(blob.words)
    print("Total words: {}\n".format(nword))
    #Word frequency
    wfscores = {word: wf(word, blob)  for word in blob.filter} #mapping word => idfscore
    wfsorted_words = sorted(wfscores.items(), key=lambda x: x[1], reverse=True) #sort & output top n
    for word, score in wfsorted_words[:150]:
    	print("Word: {}, WF: {} ".format(word, round(score, 5)))
    print "++++++++++++++++++++"
    # Term frequencies
    tfscores = {word: tf(word, blob)  for word in blob.filter} #mapping word => idfscore
    tfsorted_words = sorted(tfscores.items(), key=lambda x: x[1], reverse=True) #sort & output top n
    for word, score in tfsorted_words[:50]:
    	print("Word: {}, TF: {} ".format(word, round(score, 5)))
    print "++++++++++++++++++++"
    # ++ n_containing
    nscores = {word: n_containing(word, bloblist) for word in blob.filter} #mapping word => idfscore 
	# print nscores
    nsorted_words = sorted(nscores.items(), key=lambda x: x[1], reverse=True) #sort & output top n
    for word, score in nsorted_words[:200]:
    	print("Word: {}, nsc: {}".format(word, round(score, 5)))
    print "++++++++++++++++++++"
    # ++ Calculate TF-IDF scores
    scores = {word: tfidf(word, blob, bloblist) for word in blob.filter} #mapping word => score 
    sorted_words = sorted(scores.items(), key=lambda x: x[0], reverse=False) #sort & output top n
    for word, score in scores.items()[:150]:
    	print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))
    print "++++++++++++++++++++"
    print "++++++++++++++++++++"
print "+++++++++++++++++++++++++"
print "Computing cosine simailarity for document4&document5:"
cscore=get_cosine(document4, document5, bloblist)
print cscore
print "Computing cosine simailarity for document5&document6:"
cscore=get_cosine(document5, document6, bloblist)
print cscore
print "Computing cosine simailarity for document4&document6:"
cscore=get_cosine(document4, document6, bloblist)
print cscore
 	
