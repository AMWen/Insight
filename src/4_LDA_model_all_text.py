### Topic modeling with LDA

# Import necessary packages
import pandas as pd
import numpy as np
import pickle
import os
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
                   "brother", "relative", "psychiatric", "ambulatory", "staff", "physical", "therapy", "hs"]
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
           if ((wn.lemmatize(tok, simplify(pos)) not in stop_words) and (simplify(pos) in ['n', 'a']))]

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
def test_eta(eta, dictionary, ntopics, print_topics=True):
    # BOW format for given dictionary
    bow = [dictionary.doc2bow(line) for line in corp]
    
    # Ignore divide-by-zero warnings
    with (np.errstate(divide='ignore')):
        model = gensim.models.ldamodel.LdaModel(corpus=bow, id2word=dictionary, num_topics=ntopics,
                                                random_state=25, eta=eta,
                                                eval_every=-1, update_every=1, chunksize=250,
                                                passes=100, alpha='auto', per_word_topics=True)
    
    # Coherence
    coherence_model_lda = CoherenceModel(model=model, texts=corp, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    
    if print_topics:
        # Display the top terms for each topic
        for topic in range(ntopics):
            print('Topic {}: {}'.format(topic, [dictionary[w] for w,p in model.get_topic_terms(topic, topn=6)]))
    return model, coherence_lda


### Begin LDA modeling

# Initialize list for the text of all reviews
review_text_list = []

# Add text of each review to list
for i in range(0, len(reviews)):
    paragraphs = reviews.loc[i]["Text"]
    review_text_list.append(paragraphs)

# Generate corpus (list of reviews that are tokenized, stop word deleted, and lemmatized)
corp = [preprocess(paragraph) for paragraph in review_text_list]

# Generate dictionary
dictionary = gensim.corpora.Dictionary(corp)

# Save corpus and dictionary
if not os.path.exists("model"):
    os.mkdir("model")
pickle.dump(dictionary, open("model/dictionary1.pickle", "wb"))
pickle.dump(corp, open("model/corpus1.pickle", "wb"))

# Save cleaned up review list
corp_df = pd.DataFrame(data={"Review List": corp})
corp_df.to_csv("data/cleaned_reviews_lemmatized.csv", sep=',',index=False)

# Find words frequency
all_words = []
for i in corp:
    for j in i:
        all_words.append(j)
fdist = nltk.FreqDist(all_words)

# Plot the most common words
sns.set(rc={'figure.figsize':(9,4)})
sns.set(rc={'axes.labelsize':24})
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)
fdist.plot(30)


### Test various LDA models

# Initialize list for LDA models with different number of topics
lda_models = []
ntopics = [2, 4, 6, 8, 10]

# Apriori seeds for different topics
apriori = {
    'bill':0, 'billing':0, 'collection':0, 'information':0, 'payment':0,
    'insurance':0, 'cash':0, 'copay':0, 'cover':0, 'pay':0, 'charge':0, # Billing
    'wait':1, 'hour':1, 'emergency':1, 'time':1, 'spend':1, 'hold':1, # Timeliness
    'dirty':2, 'awful':2, 'filthy':2, 'garbage':2, 'rank':2, 'nasty':2, # Cleanliness
    'amaze':3, 'attentive':3, 'experience':3, 'great':3, 'good':3, 'nice':3, 'best':3, 'impressed':3, # Positivity
    'rude':4, 'inconsiderate':4, 'polite':4, 'caring':4, 'patient':4, 'amazing':4,
    'love':4, 'fantastic':4, 'awesome':4, # Compassion
    'expert':5, 'expertise':5, 'incompetent':5, # Expertise
    'pain':6, 'bad':6, 'severe':6, # Experience
    'understaffed':7, 'overcrowded':7, 'triage':7, 'crowded':7, # Staffing
    'attitude':8, 'horrific':8, 'aggressive':8, 'atrocious':8, 'rude':8, 'complaint':8 # Politeness
}

for i in range(0, len(ntopics)):
    eta = create_eta(apriori, dictionary, ntopics[i])
    lda_models.append(test_eta(eta, dictionary, ntopics=ntopics[i]))

# Graph of coherence
plt.plot(ntopics, [lda_models[i][1] for i in range(0, len(ntopics))])
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.xticks(np.arange(0, 11, 2))
plt.show()


### Final LDA model

# Set up LDA model parameters
ntopics = 6
passes_in = 100
num_words = 10
apriori = {
    'bill':0, 'billing':0, 'collection':0, 'information':0, 'payment':0,
    'insurance':0, 'cash':0, 'copay':0, 'cover':0, 'pay':0, 'charge':0, # Billing
    'wait':1, 'hour':1, 'emergency':1, 'time':1, 'spend':1, 'hold':1, # Timeliness
    'dirty':2, 'awful':2, 'filthy':2, 'garbage':2, 'rank':2, 'nasty':2, # Cleanliness
    'amaze':3, 'attentive':3, 'experience':3, 'great':3, 'good':3, 'nice':3, 'best':3, 'impressed':3, # Positivity
    'rude':4, 'inconsiderate':4, 'polite':4, 'caring':4, 'patient':4, 'amazing':4,
    'love':4, 'fantastic':4, 'awesome':4, # Compassion
    'understaffed':5, 'overcrowded':5, 'triage':5, 'crowded':5, # Staffing
}

# Train LDA model
bow = [dictionary.doc2bow(line) for line in corp] # Get the bow format with set dictionary
eta = create_eta(apriori, dictionary, ntopics)
with (np.errstate(divide='ignore')):  # ignore divide-by-zero warnings
    ldamodel = gensim.models.ldamodel.LdaModel(corpus=bow, id2word=dictionary, num_topics=ntopics,
                                               random_state=25, eta=eta,
                                               eval_every=-1, update_every=1, chunksize=250,
                                               passes=passes_in, alpha='auto')

# Save model results
pickle.dump(ldamodel, open("model/lda1.pickle", "wb"))

# Print topics
topic_list = ldamodel.print_topics(num_topics=ntopics, num_words=num_words)
for index, topic in enumerate(topic_list):
    print(topic[1])
