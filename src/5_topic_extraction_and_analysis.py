### Sentiment analysis and review highlights

# Import necessary packages
import pandas as pd
import numpy as np
import pickle

# NLP packages
import nltk
from nltk import tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora, models, similarities
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load my datasets
# Yelp review data
reviews = pd.read_csv('data/yelp_reviews_averaged.csv')
review_main_text = reviews['Text']

# Review counts from above
review_counts = pd.read_csv('data/review_counts_true.csv')
review_counts = review_counts['Text']

# Hospital information and Medicare survey data
hospital_info = pd.read_csv("data/Medicare responses Yelp cleaned.csv")
hospitals = hospital_info['Hospital Name']

# Load saved models
dictionary = pickle.load(open("model/dictionary2.pickle", "rb"))
ldamodel = pickle.load(open("model/lda2.pickle", "rb"))

# Generate topic matrix with 5 general, interpretable topics from LDA
all_topics = ["Experience", "Billing", "Quality of Care", "Other", "Timeliness"]
topics=[[all_topics[i], [i]] for i in range(0,5)]

# Flip the order (LDA topic as index, location in labeled topics list as value)
# If LDA topic does not fit into one of the labeled topics, set its name to -1
ntopics = 6
topics_matrix = [ -1 for i in range(ntopics) ]

for index, topics in enumerate(topics):
    for item in topics[1]:
        topics_matrix[item] = index
print(topics_matrix)

# Initialize dataframe to store topic statistics
all_hospital_topics = pd.DataFrame(columns = ["Hospital ID", "Hospital Name", "Review Counts"]+all_topics)

# Counter for reviews analyzed
counter = 0

# Initialize sentiment intensity analyzer and tokenizer
sia=SentimentIntensityAnalyzer()

# Dataframe to save all sentences with topics and their sentiment
all_hospitals = pd.DataFrame(columns = ["Sentence", "Sentiment", "Topic", "Topic Score"])

# For each hospital, find sentiment statistics for each topic and the 3 most positive/most negative sentences
for hospital_id in range(0, len(hospitals)):
    # Total number of reviews for the hospital
    hospital_review_count = review_counts[hospital_id]
    
    # Analyze all sentences for the current hospital
    this_hospital = pd.DataFrame(columns = ["Sentence", "Sentiment", "Topic", "Topic Score"])
    
    # Counter for number of sentences total for each hospital
    sent_count = 0
    
    # Loop through every review
    for review_id in range(0, hospital_review_count):
        
        # Extract review and tokenize into sentences
        real_review_id = review_id + counter
        sentences = tokenize.sent_tokenize(review_main_text[real_review_id])
        
        # Loop through every sentence in the review
        for sentence in sentences:
            # Assess sentiment
            sentiment = sia.polarity_scores(sentence)['compound']
            
            # Assign topic (default is -1 if does not match with anything)
            this_topic = -1
            
            # Preprocess sentence and evaluate for topic
            cleaned_sent = gensim.utils.simple_preprocess(str(sentence), deacc=True)
            lda_score = ldamodel[dictionary.doc2bow(cleaned_sent)]
            
            # Sort sentence topics by LDA score
            sent_topics =  sorted(lda_score, key=lambda x: x[1], reverse=True)
            
            # Assign the most relevant topic to a sentence only if the topic is more than 65% dominant
            if sent_topics[0][1] > 0.65:
                this_topic = topics_matrix[sent_topics[0][0]]

            # Add procressed sentence and topic information to the sentence dataframe
            this_hospital.loc[sent_count] = [sentence, sentiment, this_topic, sent_topics[0][1]]
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
    print(counter/sum(review_counts))
    counter += hospital_review_count

# Save dataframe of sentences with topics and their sentiment
all_hospitals.to_csv("data/sentence_topics.csv", index=False)

# New dataframe with ratings
hospital_reviews = reviews[["Hospital ID", "Average Rating"]]
hospital_reviews = hospital_reviews.drop_duplicates()
all_hospital_topics = pd.merge(all_hospital_topics, hospital_reviews, on=['Hospital ID'])

# Save the updated hospital dataframe containing individual information
hospital_info = pd.merge(hospital_info, all_hospital_topics.drop(['Hospital ID'], axis=1), on=['Hospital Name'])
hospital_info.to_csv("data/all_hospitals.csv", index=False)
