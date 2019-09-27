### Scrape remaining Yelp pages

# Import necessary packages
import pandas as pd
import numpy as np
import os
import math
import urllib
from bs4 import BeautifulSoup

# Initialize list of file names and number of reviews in each file
file_names = []
number_reviews = []
hospital_id = []    # Index of hospital corresponding to hospitals dataframe

# Hospital information and Medicare survey data
hospital_info = pd.read_csv("data/Medicare responses Yelp cleaned.csv")
hospitals = hospital_info['Hospital Name']

# Load review counts
review_counts = pd.read_csv("data/review_counts.csv")
review_counts = review_counts['Review Counts']

# Loop through all hospitals
for i in range(0, len(review_counts)):
    # Append first page file name
    name = "file://" + os.getcwd() + "/yelp/{0}_1.html".format(i) # open new file in binary mode
    file_names.append(name)
    
    # Append hospital ID index
    hospital_id.append(i)
    
    # Unless number of reviews is 20 or less, first page has 20 reviews
    if review_counts[i] < 21:
        number_reviews.append(review_counts[i])
    else:
        number_reviews.append(20)
    
    # Subsequent pages
    if review_counts[i] > 20:
        # Calculate number of extra pages
        extra_pages = math.ceil(int(review_counts[i])/20)-1
        
        for j in range(1, extra_pages+1):
            # Append file name
            name = "file://" + os.getcwd() + "/yelp/{0}_{1}.html".format(i,j+1)
            file_names.append(name)
            
            # Page has 20 reviews unless last page of reviews
            if j < extra_pages:
                number_reviews.append(20)
            else:
                if review_counts[i]%20 == 0:
                    number_reviews.append(20)
                else:
                    number_reviews.append(review_counts[i]%20)
    
            # Append hospital ID index
            hospital_id.append(i)

# Sanity check that all file names, number of reviews, and hospital IDs are correct
# print(file_names)
# print(number_reviews)
# print(hospital_id)

# Read in saved html pages and obtain review data

# Initialize the reviews dataframe
reviews = pd.DataFrame(columns = ["Hospital ID", "Hospital Name", "Hospital Overall Rating", "Rating", "Text", "Date","Useful","Funny","Cool","Page URL","Reviewer URL","Reviewer Location","Reviewer Friend Count","Reviewer Review Count","Reviewer Photo Count"])

count = 0

# For every html page
for i in range(0, len(file_names)):
    page = urllib.request.urlopen(file_names[i])
    soup = BeautifulSoup(page, 'html.parser')
    url_loc = soup.find_all("script")[5].text.find("full_url")
    page_url0 = soup.find_all("script")[5].text[url_loc:].split("\"")[2]
    hospital = hospital_id[i]
    
    try:
        rating = str(soup.find_all("div", class_="biz-rating biz-rating-very-large clearfix")[0].find("img")).split("\"")[1].split(" ")[0]
    
    except:
        print(i)

    # For every review on the page
    for j in range(0, number_reviews[i]):
        try:
            review_rating = float(str(soup.find_all("div", itemprop="reviewRating")[j].find("meta")).split("\"")[1])
            review_text = soup.find_all("p", itemprop="description")[j].text
            review_date = soup.find_all("div", class_="review-content")[j].find("span").text.split("\n")[1].replace(" ", "")
            review_useful_count = str(soup.find_all("a", class_="ybtn ybtn--small ybtn--secondary useful js-analytics-click")[j].find_all("span", class_="count")).replace("<", ">").split(">")[2]
            review_funny_count = str(soup.find_all("a", class_="ybtn ybtn--small ybtn--secondary funny js-analytics-click")[j].find_all("span", class_="count")).replace("<", ">").split(">")[2]
            review_cool_count = str(soup.find_all("a", class_="ybtn ybtn--small ybtn--secondary cool js-analytics-click")[j].find_all("span", class_="count")).replace("<", ">").split(">")[2]
            page_url = page_url0
            reviewer_url = str(soup.find_all("div", class_="review review--with-sidebar")[j].find("a", class_="js-analytics-click")).split("\"")[5]
            reviewer_location = str(soup.find_all("div", class_="review review--with-sidebar")[j].find("li", class_="user-location responsive-hidden-small").text).replace("\n", "")
            reviewer_friend_count = soup.find_all("div", class_="review review--with-sidebar")[j].find("li", class_="friend-count responsive-small-display-inline-block")
            reviewer_review_count = soup.find_all("div", class_="review review--with-sidebar")[j].find("li", class_="review-count responsive-small-display-inline-block")
            reviewer_photo_count = soup.find_all("div", class_="review review--with-sidebar")[j].find("li", class_="photo-count responsive-small-display-inline-block")
            
            if review_useful_count == "":
                review_useful_count = 0
            else:
                review_useful_count = int(review_useful_count)
            
            if review_funny_count == "":
                review_funny_count = 0
            else:
                review_funny_count = int(review_funny_count)
            
            if review_cool_count == "":
                review_cool_count = 0
            else:
                review_cool_count = int(review_cool_count)
            
            if reviewer_friend_count == None:
                reviewer_friend_count = 0
            else:
                reviewer_friend_count = int(str(reviewer_friend_count.text).split(" ")[0].replace("\n", ""))
            
            if reviewer_review_count == None:
                reviewer_review_count = 0
            else:
                reviewer_review_count = int(str(reviewer_review_count.text).split(" ")[0].replace("\n", ""))
            
            if reviewer_photo_count == None:
                reviewer_photo_count = 0
            else:
                reviewer_photo_count = int(str(reviewer_photo_count.text).split(" ")[0].replace("\n", ""))
            
            reviews.loc[count] = [hospital, hospitals[hospital], rating, review_rating, review_text, review_date,review_useful_count,review_funny_count,review_cool_count,page_url,reviewer_url,reviewer_location,reviewer_friend_count,reviewer_review_count,reviewer_photo_count]
            count += 1
            # Progress tracking
            if (count % 500) == 0:
                print(count)

        except:
            # Several hospitals have fewer reviews (usually 1) than Yelp's reported total:
            # Southside Hospital Bay, NewYork-Presbyterian/Queens, Ellis Hospital,
            # Good Samaritan Hospital Of Suffern, Long Island Jewish Medical Center,
            # Bellevue Hospital Center, Woodhull Medical And Mental Health Center
            print(i, j+1, page_url0)

# Double-check the reviews are in order
reviews.tail()

# Save the reviews to the hard drive.
reviews.to_csv("data/yelp_reviews.csv", index=False)
