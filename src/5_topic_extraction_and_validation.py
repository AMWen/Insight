### Sentiment analysis and review highlights

# Import necessary packages
import pandas as pd
import numpy as np
import pickle
import random

# NLP packages
import nltk
from nltk import tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora, models, similarities
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load my datasets
# Yelp review data
reviews = pd.read_csv("data/yelp_reviews_averaged.csv")
review_main_text = reviews['Text']

# Preprocessed sentences from step 4
all_sentences = pickle.load(open("data/cleaned_reviews_lemmatized.pickle", "rb"))

# Hospital information and Medicare survey data
hospital_info = pd.read_csv("data/Medicare responses Yelp cleaned.csv")
hospitals = hospital_info['Hospital Name']

# Review counts
review_counts = pd.read_csv('data/review_counts_true.csv')
review_counts = review_counts['Text']

# Load saved models
dictionary = pickle.load(open("model/dictionary4.pickle", "rb"))
lda_models = pickle.load(open("model/lda_models4.pickle", "rb"))

# Want model with 5 topics, where lda_models correspond to ntopics = [5]
ldamodel = lda_models[0][0]

# Generate topic matrix with 5 general, interpretable topics from LDA
# Topic 0: ['care', 'hour', 'room', 'billing', 'insurance', 'patient']
# Topic 1: ['spend', 'hour', 'waiting', 'emergency', 'time', 'late']
# Topic 2: ['bathroom', 'filthy', 'disgust', 'floor', 'clean', 'dirty']
# Topic 3: ['attentive', 'excellent', 'nice', 'great', 'care', 'experience']
# Topic 4: ['crowded', 'understaffed', 'triage', 'crowd', 'overcrowded', 'way']
all_topics = ["Overall Experience", "Timeliness", "Cleanliness", "Friendliness", "Spaciousness"]
ntopics = 5
topics=[[all_topics[i], [i]] for i in range(0, ntopics)]

# Initialize topics matrix with -1's
topics_matrix = [ -1 for i in range(ntopics) ]

# Flip the order (LDA topic as index, location in labeled topics list as value)
for index, topics in enumerate(topics):
    for item in topics[1]:
        topics_matrix[item] = index

# Initialize dataframe to store topic statistics
all_hospital_topics = pd.DataFrame(columns = ["Hospital ID", "Hospital Name", "Review Counts"]+all_topics)

# Counter for reviews analyzed
counter = 0

# Dataframe to save all sentences with topics and their sentiment
all_hospitals = pd.DataFrame(columns = ["Sentence", "Sentiment", "Topic", "Topic Score", "Tokens"])

# Loop through all the hospitals
for hospital_id in range(0, len(hospitals)):
    hospital = hospitals[hospital_id]
    this_hospital = pd.DataFrame(columns = ["Sentence", "Sentiment", "Topic", "Topic Score", "Tokens"])
    
    hospital_reviews = all_sentences[all_sentences['Hospital Name']==hospital].reset_index(drop=True)
    
    sent_count = 0

    # For each sentence, find sentiment statistics for each topic and the 3 most positive/most negative sentences
    for review_id in range(0, len(hospital_reviews)):
        # Obtain sentiment
        sentiment = hospital_reviews.loc[review_id, 'Sentiment']
            
        # Assign topic (default is -1 if does not match with anything)
        this_topic = -1
            
        # Evaluate preprocessed sentence for topic
        lda_score = ldamodel[dictionary.doc2bow(hospital_reviews.loc[review_id, 'Review List'])]

        # Sort sentence topics by LDA score
        sent_topics =  sorted(lda_score, key=lambda x: x[1], reverse=True)
            
        # Assign the most relevant topic to a sentence only if the topic is more than 65% dominant
        if sent_topics[0][1] > 0.6:
            this_topic = topics_matrix[sent_topics[0][0]]

        # Add procressed sentence and topic information to the sentence dataframe
        this_hospital.loc[sent_count] = [hospital_reviews.loc[review_id, 'Sentence'], sentiment, this_topic, sent_topics[0][1], hospital_reviews.loc[review_id, 'Review List']]
        sent_count += 1

    # Save the 3 most positive and negative sentiments as long as fits a topic category (not -1)
    this_hosp2 = this_hospital.sort_values(['Sentiment'], ascending=False).reset_index(drop=True)
    this_hosp2 = this_hosp2.loc[this_hosp2['Topic'] != -1].reset_index(drop=True)

    # Initialize review highlights string
    highlights = ""
        
    # Save the most polarizing sentiments only if there are at least 6 sentences
    if len(this_hosp2) > 5:
        # Only keep positive sentiments if sentiment score is above 0.4
        for i in range(0, 3):
            if this_hosp2.loc[i]['Sentiment'] > 0.4:
                highlights = highlights + this_hosp2.loc[i]['Sentence'] + "SPLIT"
            else:
                highlights = highlights + "SPLIT"
    
        # Only keep negative sentiment if sentiment score is below -0.2
        this_hosp2 = this_hosp2.sort_values(['Sentiment'], ascending=True).reset_index(drop=True)
        for i in range(0, 3):
            if this_hosp2.loc[i]['Sentiment'] < -0.2:
                highlights = highlights + this_hosp2.loc[i]['Sentence'] + "SPLIT"
            else:
                highlights = highlights + "SPLIT"

        highlights = highlights + str(sent_count)

    # Add review highlights to the hospital dataframe
    hospital_info.loc[hospital_id, 'Summary'] = highlights

    # Add % positive ratings for each topic to dataframe ([total positive, total count] for each topic)
    doc_topics = [ [ 0 for i in range(2) ] for j in range(len(all_topics)) ]

    # Loop through every sentence with clear topic
    for sentence_index in range(0, len(this_hosp2)):

        # Extract topic for each sentence
        topic_index = this_hosp2.loc[sentence_index]['Topic']

        # Increase counter for topic's total counts by 1
        doc_topics[topic_index][1] += 1

        # Extract sentiment for each sentence
        topic_sentiment = this_hosp2.loc[sentence_index]['Sentiment']

        # If sentiment is positive (greater than 0.05), increase counter for positive by 1
        if topic_sentiment > 0.05:
            doc_topics[topic_index][0] += 1

    # Fill in table of % positive sentiments for each topic for this hospital
    all_hospital_topics.loc[hospital_id] = [hospital_id, hospitals[hospital_id], review_counts[hospital_id]]+[i[0]/i[1]*100 if i[1] > 0 else 0 for i in doc_topics]

    # Append this hospital's sentences with topics and their sentiment to larger dataframe
    all_hospitals = all_hospitals.append(this_hosp2)

    # Print progress
    print(counter/len(all_sentences))
    counter += len(hospital_reviews)

# Save dataframe of sentences with topics and their sentiment
all_hospitals.to_csv("data/sentence_topics.csv", index=False)

# New dataframe with ratings
hospital_reviews = reviews[["Hospital ID", "Average Rating"]]
hospital_reviews = hospital_reviews.drop_duplicates()
all_hospital_topics = pd.merge(all_hospital_topics, hospital_reviews, on=['Hospital ID'])

# Save the updated hospital dataframe containing individual information
hospital_info = pd.merge(hospital_info, all_hospital_topics.drop(['Hospital ID'], axis=1), on=['Hospital Name'])
hospital_info.to_csv("data/all_hospitals.csv", index=False)


### Topic validation

# Extract sentences from each topic
all_hospitals = pd.read_csv("data/sentence_topics.csv")

experience = all_hospitals[all_hospitals['Topic']==0].reset_index(drop=True)
timeliness = all_hospitals[all_hospitals['Topic']==1].reset_index(drop=True)
cleanliness = all_hospitals[all_hospitals['Topic']==2].reset_index(drop=True)
friendliness = all_hospitals[all_hospitals['Topic']==3].reset_index(drop=True)
spaciousness = all_hospitals[all_hospitals['Topic']==4].reset_index(drop=True)

# Select random sentences from each
nrand = 50
random_experience = experience.loc[random.sample(range(len(experience)), nrand)]
random_timeliness = timeliness.loc[random.sample(range(len(timeliness)), nrand)]
random_cleanliness = cleanliness.loc[random.sample(range(len(cleanliness)), nrand)]
random_friendliness = friendliness.loc[random.sample(range(len(friendliness)), nrand)]
random_spaciousness = spaciousness.loc[random.sample(range(len(spaciousness)), nrand)]

# Save the sentences to csv
random_experience.to_csv("data/random_experience.csv")
random_timeliness.to_csv("data/random_timeliness.csv")
random_cleanliness.to_csv("data/random_cleanliness.csv")
random_friendliness.to_csv("data/random_friendliness.csv")
random_spaciousness.to_csv("data/random_spaciousness.csv")

