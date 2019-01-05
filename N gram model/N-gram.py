
# coding: utf-8

# In[12]:


import re
import math
import numpy as np
from nltk.corpus import brown
import matplotlib.pyplot as plt
from __future__ import print_function

import time
import nltk
nltk.download('brown')


# In[4]:


def prepare_data(data, dev_frac=0.2, tst_frac=0.2, trigram=False):
    """Splits data into train, dev, test sets
    1. We join sentences by blank space and add start and stop after each sentence based on model.
    2. We made everything lowercase and treated punctuation as separate words.
    3. We split the data into three sets of lists of sentences."""
    
    sentences = []
    for i in brown.sents():
        sentence = ' '.join(i)        
        if trigram: 
            sentence = '<s> <s> '+sentence+' </s>'
        else:
            sentence = '<s> '+sentence+' </s>'
        sentences.append(sentence.lower())

    np.random.seed(12345)
    return np.split(sentences, [int((1-tst_frac - dev_frac)*len(sentences)), 
                                int((1-tst_frac)*len(sentences))])

def writing_data(filename, data):
    with open(filename, mode="w") as outfile:  # also, tried mode="rb"
        for s in data:
            outfile.write("%s\n" % s)

def read_sentences_from_file(file_path):
    with open(file_path, "r") as f:
        return [re.split("\s+", line.rstrip('\n')) for line in f]


# # Creating Unigram, Bigram, and Trigram Classes

# In[5]:


# used for unseen words in training vocabularies
UNK = None

# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

class UnigramLanguageModel:
    def __init__(self, sentences, Lambda, smoothing=False):
        self.unigram_frequencies = dict()
        self.corpus_length = 0
        for sentence in sentences:
            for word in sentence:
                self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                if word != SENTENCE_START and word != SENTENCE_END:
                    self.corpus_length += 1
        # subtract 2 because unigram_frequencies dictionary contains values for SENTENCE_START and SENTENCE_END
        self.unique_words = len(self.unigram_frequencies) - 2
        self.smoothing = smoothing
        self.Lambda = Lambda

    def calculate_unigram_probability(self, word):
            word_probability_numerator = self.unigram_frequencies.get(word, 0)
            word_probability_denominator = self.corpus_length
            if self.smoothing:
                word_probability_numerator += self.Lambda
                # add one more to total number of seen unique words for UNK - unseen events
                word_probability_denominator += self.unique_words + self.Lambda 
            return float(word_probability_numerator) / float(word_probability_denominator)

    def calculate_sentence_probability(self, sentence, normalize_probability=True):
        sentence_probability_log_sum = 0
        for word in sentence:
            if word != SENTENCE_START and word != SENTENCE_END:
                word_probability = self.calculate_unigram_probability(word)
                sentence_probability_log_sum += math.log(word_probability, 2)
        return math.pow(2, sentence_probability_log_sum) if normalize_probability else sentence_probability_log_sum                

    def sorted_vocabulary(self):
        full_vocab = list(self.unigram_frequencies.keys())
        full_vocab.remove(SENTENCE_START)
        full_vocab.remove(SENTENCE_END)
        full_vocab.sort()
        full_vocab.append(UNK)
        full_vocab.append(SENTENCE_START)
        full_vocab.append(SENTENCE_END)
        return full_vocab

class BigramLanguageModel(UnigramLanguageModel):
    def __init__(self, sentences, Lambda, smoothing=False):
        UnigramLanguageModel.__init__(self, sentences, Lambda, smoothing)
        self.bigram_frequencies = dict()
        self.unique_bigrams = set()
        self.Lambda = Lambda
        for sentence in sentences:
            previous_word = None
            for word in sentence:
                if previous_word != None:
                    self.bigram_frequencies[(previous_word, word)] = self.bigram_frequencies.get((previous_word, word),
                                                                                                 0) + 1
                    if previous_word != SENTENCE_START and word != SENTENCE_END:
                        self.unique_bigrams.add((previous_word, word))
                previous_word = word
        # we subtracted two for the Unigram model as the unigram_frequencies dictionary
        # contains values for SENTENCE_START and SENTENCE_END but these need to be included in Bigram
        self.unique__bigram_words = len(self.unigram_frequencies)

    def calculate_bigram_probabilty(self, previous_word, word):
        bigram_word_probability_numerator = self.bigram_frequencies.get((previous_word, word), 0)
        bigram_word_probability_denominator = self.unigram_frequencies.get(previous_word, 0)
        if self.smoothing:
            bigram_word_probability_numerator += self.Lambda
            bigram_word_probability_denominator += self.unique__bigram_words+self.Lambda
        return 0.0 if bigram_word_probability_numerator == 0 or bigram_word_probability_denominator == 0 else float(
            bigram_word_probability_numerator) / float(bigram_word_probability_denominator)

    def calculate_bigram_sentence_probability(self, sentence, normalize_probability=True):
        bigram_sentence_probability_log_sum = 0
        previous_word = None
        for word in sentence:
            if previous_word != None:
                bigram_word_probability = self.calculate_bigram_probabilty(previous_word, word)
                bigram_sentence_probability_log_sum += math.log(bigram_word_probability, 2)
            previous_word = word
        return math.pow(2,bigram_sentence_probability_log_sum) if normalize_probability else bigram_sentence_probability_log_sum

    
class TrigramLanguageModel(BigramLanguageModel):
    def __init__(self, sentences, Lambda, smoothing=False):
        BigramLanguageModel.__init__(self, sentences, Lambda, smoothing)
        self.trigram_frequencies = dict()
        self.unique_trigrams = set()
        self.Lambda = Lambda
    
        from itertools import izip

        def pairwise(iterable):
            "s -> (s0, s1), (s2, s3), (s4, s5), ..."
            a = iter(iterable)
            return izip(a, a)

        for sentence in sentences:
            first_previous_word = None
            second_previous_word = None
            for word1, word2 in pairwise(sentence):
                if first_previous_word != None and second_previous_word != None:
                    self.trigram_frequencies[(second_previous_word,first_previous_word, word1)] =                     self.trigram_frequencies.get(
                        (second_previous_word,first_previous_word, word1), 0) + 1
                    if second_previous_word != SENTENCE_START and word2 != SENTENCE_END:
                        self.unique_trigrams.add((second_previous_word,first_previous_word, word1))
                first_previous_word = word2
                second_previous_word = word1
                
        # we subtracted two for the Unigram model as the unigram_frequencies dictionary
        # contains values for SENTENCE_START and SENTENCE_END but these need to be included in Bigram
        self.unique__trigram_words = len(self.trigram_frequencies)
        
    def calculate_trigram_probabilty(self, second_previous_word,first_previous_word, word):
        trigram_word_probability_numerator = self.trigram_frequencies.get((second_previous_word,first_previous_word, word), 0)
        trigram_word_probability_denominator = self.bigram_frequencies.get((second_previous_word,first_previous_word), 0)
        if self.smoothing:
            trigram_word_probability_numerator += self.Lambda
            trigram_word_probability_denominator += self.unique__trigram_words + self.Lambda
        return 0.0 if trigram_word_probability_numerator == 0 or trigram_word_probability_denominator == 0 else float(
            trigram_word_probability_numerator) / float(trigram_word_probability_denominator)

    def calculate_trigram_sentence_probability(self, sentence, normalize_probability=True):
        
        from itertools import izip
        def pairwise(iterable):
            "s -> (s0, s1), (s2, s3), (s4, s5), ..."
            a = iter(iterable)
            return izip(a, a)
        
        trigram_sentence_probability_log_sum = 0
        first_previous_word = None
        second_previous_word = None
        
        for word1, word2 in pairwise(sentence):
            if first_previous_word != None and second_previous_word != None:
                trigram_word_probability = self.calculate_trigram_probabilty(second_previous_word,first_previous_word, word1)
                trigram_sentence_probability_log_sum += math.log(trigram_word_probability, 2)
            first_previous_word = word2
            second_previous_word = word1
        return math.pow(2,
                        trigram_sentence_probability_log_sum) if normalize_probability else trigram_sentence_probability_log_sum
        


# In[6]:


# calculate number of unigrams & bigrams
def calculate_number_of_unigrams(sentences):
    unigram_count = 0
    for sentence in sentences:
        # remove two for <s> and </s>
        unigram_count += len(sentence) - 2
    return unigram_count

def calculate_number_of_bigrams(sentences):
        bigram_count = 0
        for sentence in sentences:
            # remove one for number of bigrams in sentence
            bigram_count += len(sentence) - 1
        return bigram_count

def calculate_number_of_trigrams(sentences):
        trigram_count = 0
        for sentence in sentences:
            # remove two for number of trigrams in sentence
            trigram_count += len(sentence) - 1
        return trigram_count

# print unigram and bigram probs
def print_unigram_probs(sorted_vocab_keys, model):
    for vocab_key in sorted_vocab_keys:
        if vocab_key != SENTENCE_START and vocab_key != SENTENCE_END:
            print("{}: {}".format(vocab_key if vocab_key != UNK else "UNK",
                                       model.calculate_unigram_probability(vocab_key)), end=" ")
    print("")

def print_bigram_probs(sorted_vocab_keys, model):
    print("\t\t", end="")
    for vocab_key in sorted_vocab_keys:
        if vocab_key != SENTENCE_START:
            print(vocab_key if vocab_key != UNK else "UNK", end="\t\t")
    print("")
    for vocab_key in sorted_vocab_keys:
        if vocab_key != SENTENCE_END:
            print(vocab_key if vocab_key != UNK else "UNK", end="\t\t")
            for vocab_key_second in sorted_vocab_keys:
                if vocab_key_second != SENTENCE_START:
                    print("{0:.5f}".format(model.calculate_bigram_probabilty(vocab_key, vocab_key_second)), end="\t\t")
            print("")
    print("")

# calculate perplexty
def calculate_unigram_perplexity(model, sentences):
    unigram_count = calculate_number_of_unigrams(sentences)
    sentence_probability_log_sum = 0
    for sentence in sentences:
        try:
            sentence_probability_log_sum -= math.log(model.calculate_sentence_probability(sentence), 2)
        except:
            sentence_probability_log_sum -= math.log(0.00001, 2)
    return math.pow(2, sentence_probability_log_sum / unigram_count)

def calculate_bigram_perplexity(model, sentences):
    number_of_bigrams = calculate_number_of_bigrams(sentences)
    bigram_sentence_probability_log_sum = 0
    for sentence in sentences:
        try:
            bigram_sentence_probability_log_sum -= math.log(model.calculate_bigram_sentence_probability(sentence), 2)
        except:
            bigram_sentence_probability_log_sum -= math.log(0.00001, 2)
    return math.pow(2, bigram_sentence_probability_log_sum / number_of_bigrams)

def calculate_trigram_perplexity(model, sentences):
    number_of_trigrams = calculate_number_of_trigrams(sentences)
    trigram_sentence_probability_log_sum = 0
    for sentence in sentences:
        try:
            trigram_sentence_probability_log_sum -= math.log(model.calculate_trigram_sentence_probability(sentence), 2)
        except:
            trigram_sentence_probability_log_sum -= math.log(0.00001, 2)
    return math.pow(2, trigram_sentence_probability_log_sum / number_of_trigrams)


# In[13]:


if __name__ == '__main__':
    
    time_init = time.time()
    ## Part A ##
    
    # splitting the data in 60:20:20 splits of sentences
    train, developing, test = prepare_data(brown, dev_frac=0.2, tst_frac=0.2)
    print('creating data in list of sentences for part A\n','Train: ',len(train),'Developing: ', len(developing),'Test: ', len(test))
        
    # splitting the data in 80:00:20 splits of sentences
    train_80, developing_0, test_20 = prepare_data(brown, dev_frac=0, tst_frac=0.2)
    print('creating data in list of sentences for part C\n','Train: ',len(train_80),'Developing: ', len(developing_0),'Test: ', len(test_20), end='\n\n')
    
    ## Writing each sentence with start and stop symbols on each line into txt files 
    writing_data('train_set.txt', train)
    writing_data('dev_set.txt', developing)
    writing_data('test_set.txt', test)
    writing_data('train_dev_set.txt', train_80)
    
    ## Reading Data from  formatted txt files to train the models
    train_data = read_sentences_from_file('train_set.txt')
    dev_data = read_sentences_from_file('dev_set.txt')
    test_data = read_sentences_from_file('test_set.txt')
    train_dev_data = read_sentences_from_file('train_dev_set.txt')
    
    print("== Starting part A == ")
    train_model_unsmoothed = BigramLanguageModel(train_data, Lambda = 0, smoothing=False)
    uni = calculate_unigram_perplexity(train_model_unsmoothed, train_data)
    bi = calculate_bigram_perplexity(train_model_unsmoothed, train_data)
    print("== TRAIN PERPLEXITY UNSMOOTHED == ")
    print("unigram: ", uni)
    print("bigram: ", bi, end='\n\n')

    ## Part B ##
    print("== Starting Part B ==")
    np.random.seed(12345)
    Lambdas = np.linspace(10e-2, 10e4, 100)
#     Lambdas = [10e-5,10e-4,10e-3,10e-2,0,1,10,10e+1,10e+2,10e+3,10e+4,10e+5,10e+6,10e+7,10e8,10e9,10e10]

    a = {"lambda":[], "perplexity_uni":[], "perplexity_bi":[]}
    for i in Lambdas:
        train_model_unsmoothed = BigramLanguageModel(train_data, Lambda = i, smoothing=True)
#         sorted_vocab_keys = train_model_unsmoothed.sorted_vocabulary()
        uni = calculate_unigram_perplexity(train_model_unsmoothed, dev_data)
        bi = calculate_bigram_perplexity(train_model_unsmoothed, dev_data)
#         print("== TRAIN PERPLEXITY SMOOTHED== ")
#         print("unigram: ", uni, "lambda: ",i)
#         print("bigram: ", bi, "lambda: ",i)
        
        a["lambda"].append(i)
        a["perplexity_uni"].append(uni)
        a["perplexity_bi"].append(bi)
        
    uni_plot = plt.scatter(a['lambda'], a['perplexity_uni'])
    bi_plot = plt.scatter(a['lambda'], a['perplexity_bi'])
    plt.legend([uni_plot, bi_plot],['Unigram', 'Bigram'])
    plt.xlabel('lambda')
    plt.ylabel('perplexity')
    plt.show()
    
    print("""\n we tried numerous values for lambda but after lambda = 100 we did not see
            any significant improvement in perplexity for models. So we decided to go ahead with 100""")
    print('\n\n\n')
    
    ## Part C ##
    print("== Starting part C ==")
    train_model_unsmoothed = BigramLanguageModel(train_dev_data, Lambda = 100, smoothing=False)
    uni = calculate_unigram_perplexity(train_model_unsmoothed, test_data)
    bi = calculate_bigram_perplexity(train_model_unsmoothed, test_data)
    print("== TEST PERPLEXITY SMOOTHED WITH LAMBDA 100 == ")
    print("unigram: ", uni)
    print("bigram: ", bi, end='\n\n')
    
    ## Part D ##
    print("== Starting part D ==")
    uni_dict = train_model_unsmoothed.unigram_frequencies
    bi_dict = train_model_unsmoothed.bigram_frequencies
    
    import random

    ## Unigram
    print("== Sentences generated by Unigram ==", end='\n')
    uni_sentences = []
    for j in range(5):
        sent=''
        for i in random.sample(uni_dict.keys(), 14):    
            if i !='</s>':
                sent += i+' '
            else:
                sent += i+'.'
                break
        uni_sentences.append(sent)
        print('sent_'+str(j)+'.', sent)

    ## Bigram
    print('\n\n')
    print("== Sentences generated by Bigram ==", end='\n')
    bi_sentences = []
    for j in range(5):
        sent=''
        for i in random.sample(bi_dict.keys(), 8):    
            if i[1] !='</s>' or (i[1]!='<s>' and i[0]!='<s>'):
                sent += i[0]+' '+i[1]+' '
            else:
                sent += i[0]+'.'
                break
        bi_sentences.append(sent)
        print('sent_'+str(j)+'.', sent)
        
    ## Part E ##
    print('\n\n')
    print("== Starting Part E ==")
    print("We chose two extensions to implement with our Language Models")
    print("1. Creative handling of Unkown Words, which we are doing in the all the parts from the beginning.")
    print("2. Created Trigram Language Model", end='\n')
    
    # splitting the data in 60:20:20 splits of sentences
    train_tri, developing_tri, test_tri = prepare_data(brown, dev_frac=0.2, tst_frac=0.2, trigram=True)
    print('creating data for part E\n','Train: ',len(train_tri),'Developing: ', len(developing_tri),'Test: ', len(test_tri), end='\n')
    
    # writing data with two start symbols for trigram in train, dev, and test sets
    writing_data('train_tri_set.txt', train_tri)
    writing_data('dev_tri_set.txt', developing_tri)
    writing_data('test_tri_set.txt', test_tri)
    
    ## Reading Data from  formatted txt files to train the models
    train_tri_data = read_sentences_from_file('train_tri_set.txt')
    dev_tri_data = read_sentences_from_file('dev_tri_set.txt')
    test_tri_data = read_sentences_from_file('test_tri_set.txt')

    ## Creating and calculating Unsmoothed trigram model
    trigram_model_unsmoothed = TrigramLanguageModel(train_tri_data, Lambda = 0, smoothing=False)
    tri_train_unsmoothed_prplxty = calculate_trigram_perplexity(trigram_model_unsmoothed, train_tri_data)
    tri_dev_unsmoothed_prplxty = calculate_trigram_perplexity(trigram_model_unsmoothed, dev_tri_data)
    tri_test_unsmoothed_prplxty = calculate_trigram_perplexity(trigram_model_unsmoothed, test_tri_data)
    
    ## Creating and calculating Smoothed trigram model
    trigram_model_smoothed = TrigramLanguageModel(train_tri_data, Lambda = 100, smoothing=True)
    tri_train_smoothed_prplxty = calculate_trigram_perplexity(trigram_model_unsmoothed, train_tri_data)
    tri_dev_smoothed_prplxty = calculate_trigram_perplexity(trigram_model_unsmoothed, dev_tri_data)
    tri_test_smoothed_prplxty = calculate_trigram_perplexity(trigram_model_unsmoothed, test_tri_data)
    
    ## example of ngrams generated by the model with frequency
    d1 = trigram_model_smoothed.unigram_frequencies
    d2 = trigram_model_smoothed.bigram_frequencies
    d3 = trigram_model_smoothed.trigram_frequencies

    print("Few tokens generated from ngram models created above")
    print("unigram: ", {k:d1[k] for k in d1.keys()[:3]})
    print("bigram: ", {k:d2[k] for k in d2.keys()[:3]})
    print("trigram: ", {k:d3[k] for k in d3.keys()[:3]})
    
    print('\n\n')
    print("== TRIGRAM PERPLEXITY UNSMOOTHED == ")
    print("Training: ",tri_train_unsmoothed_prplxty)
    print("Developing: ", tri_dev_unsmoothed_prplxty)
    print("Test: ", tri_test_unsmoothed_prplxty, end='\n\n')
    
    print("== TRIGRAM PERPLEXITY SMOOTHED == ")
    print("Training: ",tri_train_smoothed_prplxty)
    print("Developing: ", tri_dev_smoothed_prplxty)
    print("Test: ", tri_test_smoothed_prplxty, end='\n\n')
  
    tri_dict = trigram_model_unsmoothed.trigram_frequencies
    
    ## Trigram
    print("== Sentences generated by Trigram ==")
    tri_sentences = []
    for j in range(5):
        sent=''
        for i in random.sample(tri_dict.keys(), 4):    
            if i[2] !='</s>' or (i[2]!='<s>' and i[1]!='<s>' and i[0]!='<s>'):
                sent += i[0]+' '+i[1]+' '+i[2]+' '
            else:
                sent += i[0]+' '+i[1]+'.'
                break
        tri_sentences.append(sent)
        print('sent_'+str(j)+'.', sent)
    
    print('\n\n')
    print("Time taken to complete: ", time.time()/60 - time_init/60, "minutes")
    


# In[25]:




