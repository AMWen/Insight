### Extract sentences from reviews for LDA training

# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# NLP packages
import nltk
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer


### Sentence extraction

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

# Initialize dataframe for sentences from all hospitals
hospital_sentences = pd.DataFrame(columns = ["Hospital Name", "Sentence", "Sentiment"])

# Counter for reviews analyzed
counter = 0

# Initialize sentiment intensity analyzer and tokenizer
sia=SentimentIntensityAnalyzer()

# Find sentences for each hospital
for hospital_id in range(0, len(hospitals)):
    # Total number of reviews for the hospital
    current_hospital = hospitals[hospital_id]
    hospital_review_count = review_counts[hospital_id]
    
    # Initialize dataframe for all sentences for the current hospital
    this_hospital = pd.DataFrame(columns = ["Hospital Name", "Sentence", "Sentiment"])
    
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
            
            # Add sentence and sentiment to the dataframe
            this_hospital.loc[sent_count] = [current_hospital, sentence, sentiment]
            sent_count += 1

    # Append this hospital's sentences with topics and their sentiment to larger dataframe
    hospital_sentences = hospital_sentences.append(this_hospital)

    # Print progress
    print(counter/sum(review_counts))
    counter += hospital_review_count

# Order sentences dataframe by sentiment and save with topics and their sentiment
hospital_sentences = hospital_sentences.sort_values(["Sentiment"], ascending=False).reset_index(drop=True)
hospital_sentences.to_csv("data/all_sentences.csv", index=False)

# Plot sentiment distribution
sns.set(rc={'figure.figsize':(6,4)})
plt.hist(hospital_sentences["Sentiment"], bins = 51)
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig("figs/sentence_sentiments.png")
plt.show()


### Sentence validation

# Extract positive, negative, and neutral sentences using 'Sentiment'
positive_sentences = hospital_sentences[hospital_sentences['Sentiment']>0.2].reset_index(drop=True)
negative_sentences = hospital_sentences[hospital_sentences['Sentiment']<-0.2].reset_index(drop=True)
neutral_sentences = hospital_sentences[(hospital_sentences['Sentiment']>-0.05) & (all_sentences['Sentiment']<0.05)].reset_index(drop=True)

# Select random sentences from each bin
nrand = 200
random_positive = positive_sentences.loc[random.sample(range(len(positive_sentences)), nrand)]
random_negative = negative_sentences.loc[random.sample(range(len(negative_sentences)), nrand)]
random_neutral = neutral_sentences.loc[random.sample(range(len(neutral_sentences)), nrand)]

# Save the sentences to csv
random_positive.to_csv("data/random_positive_sentences.csv")
random_negative.to_csv("data/random_negative_sentences.csv")
random_neutral.to_csv("data/random_neutral_sentences.csv")
all = random_positive.append(random_negative).append(random_neutral)
all.to_csv("data/random_all_sentences.csv", sep=",", index=False)
