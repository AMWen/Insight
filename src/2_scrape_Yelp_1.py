### Scrape Yelp reviews for hospitals

# Import necessary packages
import pandas as pd
import numpy as np
import os
import urllib
from bs4 import BeautifulSoup
import time
import random
import math

# Load Medicare responses with Yelp links (added by hand)
df_yelp_survey = pd.read_csv("data/Medicare responses Yelp.csv")

# Remove rows without Yelp links (end up with 180 hospitals total)
df_yelp_survey = df_yelp_survey.dropna(axis=0).reset_index(drop=True)
df_yelp_survey.to_csv("data/Medicare responses Yelp cleaned.csv", sep=',', index=False)

# Extract Yelp links and hospital names from dataframe
yelp_links = df_yelp_survey['Link']
hospitals = df_yelp_survey['Hospital Name']

# Scrape the first page (contains up to 20 reviews) and get the total review count
review_counts = []

# Create directory for yelp reviews
if not os.path.exists("yelp"):
    os.mkdir("yelp")

# Get page for every hospital
for i in range(0, len(hospitals)):
    try:
        url = yelp_links[i]
        page = urllib.request.urlopen(url)
        
        # Turn into BeatifulSoup object for parsing
        soup = BeautifulSoup(page, 'html.parser')
        
        # Extract the total number of reviews
        soup_review_count = soup.find_all('span', class_='review-count rating-qualifier')
        review_count = str(soup_review_count[0]).split("\n")[1].replace(" ", "").replace("reviews", "").replace("review", "")
        review_counts.append(review_count)
        
        # Check progress
        print(hospitals[i], review_count)
        
        # Save webpage for parsing later
        data = str(soup.find_all("html"))
        file = open("yelp/{0}_1.html".format(i),"w+") # open new file in binary mode
        file.writelines(data)
        file.close()
        
        # Random wait time between requests
        random_no = (random.random()-0.5) * 3.3333
        time.sleep(5+random_no)
    
    # If an error occurred
    except:
        print("ERROR:", url)

# Save review counts
review_counts_df = pd.DataFrame(data={"Hospital Name": hospitals, "Review Counts": review_counts})
review_counts_df.to_csv("data/review_counts.csv", sep=',', index=False)
