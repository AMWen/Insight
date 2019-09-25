### Scrape remaining Yelp pages

# Import necessary packages
import pandas as pd
import numpy as np
import urllib
from bs4 import BeautifulSoup
import time
import random
import math

# Initialize lists for links for the rest of the review pages
extra_links = []
extra_links_index1 = []
extra_links_index2 = []

# Load Medicare responses with Yelp links
df_yelp_survey = pd.read_csv("data/Medicare responses Yelp cleaned.csv")

# Extract Yelp links and hospital names from dataframe
yelp_links = df_yelp_survey['Link']
hospitals = df_yelp_survey['Hospital Name']

# Load review_counts data
review_counts_df = pd.read_csv('data/review_counts.csv')
review_counts = review_counts_df['Review Counts']

# Store the URLs for review pages in list for each hospital
for i in range(0, len(review_counts)):
    if review_counts[i] > 20:
        
        # Calculate the remaining number of pages
        no_of_remaining_pages = math.ceil(int(review_counts[i])/20)-1
        for j in range(1, no_of_remaining_pages+1):
            
            # Generate the extra Yelp links
            extra_link = "{0}?start={1}".format(yelp_links[i], j*20)
            extra_links.append(extra_link)
            extra_links_index1.append(i)
            extra_links_index2.append(j+1)

# Scrape Yelp for the extra pages
for i in range(0, len(extra_links)):
    try:
        url = extra_links[i]
        page = urllib.request.urlopen(url)
        
        # Turn into BeatifulSoup object for parsing
        soup = BeautifulSoup(page, 'html.parser')
        
        # Check progress
        print(hospitals[extra_links_index1[i]], 'page', extra_links_index2[i])
        
        # Save webpage for processing later
        data = str(soup.find_all("html"))
        file = open("yelp/{0}_{1}.html".format(extra_links_index1[i], extra_links_index2[i]), "w+") # open new file in binary mode
        file.writelines(data)
        file.close()
        
        # Random wait time between requests
        random_no = (random.random()-0.5) * 3.333
        time.sleep(5.12+random_no)

    # If an error occurred
    except:
        print("ERROR:", index)
