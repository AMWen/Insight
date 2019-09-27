### Topic modeling with LDA

# Import necessary packages
import pandas as pd
import numpy as np
import pickle
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns

# NLP packages
# nltk.download('stopwords')
# nltk.download('vader_lexicon')
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora, models, similarities
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess

# Guided LDA tutorial from http://scignconsulting.com/2019/03/09/guided-lda/

# Load my dataset
reviews = pd.read_csv("data/yelp_reviews_averaged.csv")
df_yelp_survey = pd.read_csv("data/Medicare responses Yelp cleaned.csv")
all_sentences = pd.read_csv("data/all_sentences.csv")
curated = pd.read_csv("data/all_sentences_annotated.csv")

# Curated dataframe subsets with annotation
curated = curated[(curated['Timeliness']==1) | (curated['Cleanliness']==1) |
                  (curated['Logistics/Billing']==1) | (curated['Space']==1) |
                  (curated['Attitude/Communication']==1) | (curated['Other'])].reset_index(drop=True)

# Shuffle rows
random.seed(23)
curated2 = curated.sample(frac=1).reset_index(drop=True)
curated3 = curated.sample(frac=1).reset_index(drop=True)

### Helper functions

# Specify stop words
stop_words = stopwords.words('english')

# Add hospital names to stop words (preprocessed and dupulicates removed)
hospitals = df_yelp_survey['Hospital Name']
hospital_names = simple_preprocess(str(hospitals.tolist()), deacc=True)
hospital_names = list(dict.fromkeys(hospital_names))
stop_words.extend(hospital_names)

# Some additonal words to ignore
words_to_ignore = ["dr", "drs", "surgery", "birth", "labor", "surgeon", "delivery", "aide", "ob", "gyn",
                   "psych", "department", "psychiatrist", "er", "doctor", "nurse", "one", "dental",
                   "blood", "baby", "grandmother", "pediatric", "kid", "son", "child", "sister",
                   "brother", "relative", "psychiatric", "ambulatory", "staff", "physical", "therapy", "hs",
                   "wait", "go", "get", "stand", "say", "owe", "take", "try", "receive", "visit", "cover",
                   "pick", "obtain", "roll", "grandson", "granddaughter", "woman"]
stop_words.extend(words_to_ignore)

# Function to simplify Penn tags to n (NOUN), v (VERB), a (ADJECTIVE), or r (ADVERB)
def simplify(penn_tag):
    pre = penn_tag[0]
    if (pre == 'J'):
        return 'a'
    elif (pre == 'R'):
        return 'r'
    elif (pre == 'V'):
        return 'v'
    else:
        return 'n'

# Function to preprocess text (remove stopwords, tokenize, remove punctuation, convert to lowercase, lemmatize, keep only nouns and adjectives)
def preprocess(text):
    # Tokenize, remove punctuation, and convert to lowercase
    tokens = gensim.utils.simple_preprocess(str(text), deacc=True)
    
    # Lemmatize and keep only nouns and adjectives
    wn = WordNetLemmatizer()
    return [wn.lemmatize(tok, simplify(pos)) for tok, pos in nltk.pos_tag(tokens)
            if ((wn.lemmatize(tok, simplify(pos)) not in stop_words) and (simplify(pos) in ['n', 'a', 'v']))]

# Function to create eta matrix based on apriori selected topics, dictionary, and number of topics
def create_eta(priors, etadict, ntopics):
    # Fill matrix with 1's
    eta = np.full(shape=(ntopics, len(etadict)), fill_value=1)
    
    # For each word in the list of priors
    for word, topic in priors.items():
        
        # Look up the word in the dictionary
        keyindex = [index for index,term in etadict.items() if term==word]
        
        # If found in the dictionary and seeded topic number is less than number of total topics
        if (len(keyindex)>0) & (topic<ntopics):
            # Seed with a large number
            eta[topic, keyindex[0]] = 1e12

    # Normalize so that the probabilities sum to 1 over all topics
    eta = np.divide(eta, eta.sum(axis=0))
    return eta

# Function to perform LDA and output coherence
def test_eta(eta, dictionary, ntopics, corpus, print_topics=True):
    # BOW format for given dictionary
    bow = [dictionary.doc2bow(line) for line in corpus]
    
    # Ignore divide-by-zero warnings
    with (np.errstate(divide='ignore')):
        model = gensim.models.ldamodel.LdaModel(corpus=bow, id2word=dictionary, num_topics=ntopics,
                                                random_state=42, eta=eta,
                                                eval_every=-1, update_every=1, chunksize=8,
                                                passes=100, alpha='asymmetric', per_word_topics=False)
    
    # Coherence
    coherence_model_lda = CoherenceModel(model=model, texts=corp, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    
    if print_topics:
        # Display the top terms for each topic
        for topic in range(ntopics):
            print('Topic {}: {}'.format(topic, [dictionary[w] for w,p in model.get_topic_terms(topic, topn=6)]))
    return model, coherence_lda


### Obtain corpus and dictionary for LDA model

# Initialize list for the text of all reviews and curated reviews
review_text_list = []
review_text_list_curated = []

# Add sentences to list
for i in range(0, len(all_sentences)):
    review_text_list.append(all_sentences.loc[i]["Sentence"])

# Sentences from curated list
topics = ['Timeliness', 'Cleanliness', 'Logistics/Billing', 'Space', 'Attitude/Communication']
for topic in topics:
    curated_topic = curated[(curated[topic]==1)].reset_index(drop=True)
    curated_topic2 = curated2[(curated2[topic]==1)].reset_index(drop=True)
    curated_topic3 = curated3[(curated3[topic]==1)].reset_index(drop=True)
    # Combine every 3 sentences so have more words per "document"
    n_sent = 3
    for i in range(0, len(curated_topic), n_sent):
        review_text_list_curated.append(curated_topic.loc[i]["Sentence"]+
                                        curated_topic2.loc[i]["Sentence"]+
                                        curated_topic3.loc[i]["Sentence"])

# Generate corpus (list of reviews that are tokenized, stop word deleted, and lemmatized)
corp = [preprocess(paragraph) for paragraph in review_text_list]
corp_curated = [preprocess(paragraph) for paragraph in review_text_list_curated]

# Generate dictionary
dictionary = gensim.corpora.Dictionary(corp)

# Save corpus and dictionary
if not os.path.exists("model"):
    os.mkdir("model")
pickle.dump(dictionary, open("model/dictionary4.pickle", "wb"))
pickle.dump(corp, open("model/corpus4.pickle", "wb"))
pickle.dump(corp_curated, open("model/corpus_curated4.pickle", "wb"))

# Save cleaned up review list dataframes
corp_df = pd.DataFrame(data={"Hospital Name": all_sentences['Hospital Name'], "Review List": corp, "Sentiment": all_sentences['Sentiment'], "Sentence": all_sentences['Sentence']})
corp_df.to_csv("data/cleaned_reviews_lemmatized.csv", sep=',',index=False)
pickle.dump(corp_df, open("data/cleaned_reviews_lemmatized.pickle", "wb"))
corp_curated_df = pd.DataFrame(data={"Review List": corp_curated})
corp_curated_df.to_csv("data/cleaned_reviews_curated_lemmatized.csv", sep=',',index=False)


### LDA model training

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load saved models (train with the sentiment corpus but use full dictionary)
dictionary = pickle.load(open("model/dictionary4.pickle", "rb"))
corp = pickle.load(open("model/corpus_curated4.pickle", "rb"))

# Initialize list for LDA models with different number of topics
lda_models = []
ntopics = [5]

# Apriori seeds for different topics
apriori = {
    'bill':0, 'billing':0, 'payment':0, 'copay':0, 'insurance':0, # Billing
    'wait':1, 'late':1, 'hour':1, 'emergency':1, 'time':1, 'spend':1, 'waiting':1, # Timeliness
    'disgust':2, 'horrible':2, 'filthy':2, 'garbage':2, 'nasty':2, 'terrible':2, # Cleanliness
    'care':3, 'attentive':3, 'experience':3, 'great':3, 'good':3, 'nice':3, 'excellent':3, 'impressed':3, # Positivity
    'understaffed':4, 'overcrowded':4, 'triage':4, 'crowded':4, 'crowd':4, # Staffing
    'attitude':5, 'horrific':5, 'aggressive':5, 'atrocious':5, 'rude':5, 'complaint':5 # Politeness
}

for i in range(0, len(ntopics)):
    eta = create_eta(apriori, dictionary, ntopics[i])
    lda_models.append(test_eta(eta, dictionary, ntopics[i], corp))
    print("")

# Save models in pickle
pickle.dump(lda_models, open("model/lda_models4.pickle", "wb"))

# Graph of coherence
plt.plot(ntopics, [lda_models[i][1] for i in range(0, len(ntopics))])
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.xticks(np.arange(0, 11, 2))
plt.show()

# Find words frequency
all_words = []
for i in corp:
    for j in i:
        all_words.append(j)
fdist_all = nltk.FreqDist(all_words)

# Find words frequency for curated sentences
all_words = []
for i in corp_curated:
    for j in i:
        all_words.append(j)
fdist_curated = nltk.FreqDist(all_words)

# Plot the most common words
# sns.set(rc={'figure.figsize':(9,4)})
# sns.set(rc={'axes.labelsize':24})
# plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=18)
# fdist.plot(30)

# Plot the most common words for curated sentences
# sns.set(rc={'figure.figsize':(9,4)})
# sns.set(rc={'axes.labelsize':24})
# plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=18)
# fdist.plot(30)
