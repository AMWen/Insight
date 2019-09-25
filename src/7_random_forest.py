### Preliminary EDA of reviews data

# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Load my dataset
reviews = pd.read_csv("data/yelp_reviews.csv")

# Double-check number of reviews obtained matches expected number
review_counts_true = pd.DataFrame(reviews.groupby('Hospital Name').Text.count())
review_counts_true.to_csv("data/review_counts_true.csv", sep=',', index=False, header=True)
review_counts_df = pd.read_csv('data/review_counts.csv')
review_counts_merge = pd.merge(review_counts_df, review_counts_true, on=['Hospital Name'])
print(review_counts_merge[review_counts_merge['Review Counts'] != review_counts_merge['Text']])

# Hospitals with overall rating
hospital_reviews = reviews[['Hospital ID', 'Hospital Name', 'Hospital Overall Rating']]
hospital_reviews = hospital_reviews.drop_duplicates()
hospital_reviews.head()

# Calculate average by hand for more nuance
hospital_reviews['Average Rating'] = 0
for i in range(0, len(hospitals)):
    hospital_reviews.loc[(hospital_reviews['Hospital ID'] == i),
                         'Average Rating'] = reviews[reviews['Hospital ID'] == i]['Rating'].mean()

# Save new dataframe
reviews = pd.merge(reviews, hospital_reviews, on=['Hospital ID', 'Hospital Name', 'Hospital Overall Rating'])
reviews.to_csv("data/yelp_reviews_averaged.csv", index=False)

# Plot distribution of overall hospital reviews
sns.set(rc={'figure.figsize':(15,4)})
plt.subplot(1, 3, 1)
plt.hist(hospital_reviews['Hospital Overall Rating'], bins = 11)
plt.xlabel('Hospital Overall Rating')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
plt.hist(hospital_reviews['Average Rating'], bins = 15)
plt.xlabel('Calculated Average Rating')
plt.ylabel('Count')
plt.show()

# Plot distribution of all patient reviews
plt.subplot(1, 3, 3)
plt.hist(reviews.Rating, bins = 5)
plt.xlabel('Rating')
plt.ylabel('Count')
