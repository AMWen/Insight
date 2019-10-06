# Hospitable Hospitals

This is a web application that assists hospital administrators with identifying key target areas to focus on to most efficiently improve patient satisfaction and public ratings.

Random forest regression was performed to predict a hospital's overall public rating based on parameters obtained from Yelp reviews that were processed using NLP topic modeling and sentiment analysis techniques as well as patient survey data from Medicare (HCAHPS).

## Project pipeline
1. Obtain Medicare Hospital Consumer Assessment of Healthcare Providers and Systems (HCAHPS) survey data of hospitals, pivot the table so columns are the various metrics (cleanliness, communication, etc.), remove hospitals without any data, and select MA and NY hospitals.
2. Find Yelp links for the various hospitals and use BeautifulSoup to perform web scraping of Yelp for their reviews.
3. Perform some preliminary exploratory data analysis of the ratings distributions.
4. Train a guided LDA model using curated reviews to obtain topics (removed neutral sentences such as "couple of years ago i had to take my wife there and a male nurse that was there at the time was working on her" to better isolate sentences pertaining to actionable topics).
5. Use NLTK VADER to perform sentence-by-sentence sentiment analysis to determine overall sentiment on each topic and extract the most positive and most negative reviews.
6. Perform linear regression on survey and Yelp data and analyze results.
7. Perform random forest regression on survey and yelp data and analyze results.
8. Create a web app using Flask that integrates the regression results and provides a dashboard so hospitals can use to determine how changes in the various parameters would affect their overall rating (http://hospitable-hospitals.com/).
