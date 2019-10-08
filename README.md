# Hospitable Hospitals
Hospitable Hospitals is a [web application](http://hospitable-hospitals.com) that assists hospital administrators with identifying key target areas to improve to most efficiently increase patient satisfaction and public ratings. By improving public perception, hospitals can then broaden their customer base.

Predictive modeling of patient ratings was performed using machine learning tools such as natural language processing and random forest regression. The results of the modeling provide hospitals with information about their performance on various measures compared to other hospitals, the relative importance of these measures in general, and what representative content online concerning these topics are like. Using this data, hospitals would then be able to make informed decisions on which aspects they would like to focus on.

This project was completed in 3 weeks as part of the [Insight Health Data Science program](https://www.insighthealthdata.com).

The back-end modeling code can be found in [src](https://github.com/AMWen/Insight/tree/master/src) and the front-end web application development files can be found in [webapp](https://github.com/AMWen/Insight/tree/master/webapp).

## 1. Datasets
- ~900,000 words from ~9,000 reviews pertaining to 180 hospitals within MA and NY.
- 2017-2018 Hospital Consumer Assessment of Healthcare Providers and Systems (HCAHPS) survey data from Medicare 

## 2. Survey data processing
Medicare patient survey data was obtained from [Medicare's website](https://data.medicare.gov/data/archives/hospital-compare). Using pandas dataframes, the table was pivoted so the columns are the various metrics (cleanliness, communication, etc.), hospitals without any data were removed (generally children's hospitals), and MA and NY hospitals were selected. 'Always' and 'Usually' survey responses were combined to obtain a measure of positive patient sentiment for various measures, and measures related to communication (for doctors, nurses, and staff) were averaged together since they were highly correlated with each other.

## 3. Yelp data processing
Yelp reviews from MA and NY hospitals were scraped and parsed using BeautifulSoup. Topic modeling was then performed on the review text using guided latent Dirichlet allocation (LDA). Removal of sentences of neutral sentiments was required to obtain more actionable topics, with sentiment analysis being performed using NLTK VADER. % positive sentiment was then determined for each of the extracted topics.

## 4. Regression model
Linear regression and random forest regression were performed to predict hopsital ratings based on Yelp and survey data. Random forest regression outperformed linear regression, likely due to multicollinearity issues, with an R^2 value of 0.78.

## 5. Acknowledgments
Thank you to Aleem Juma and Nuo Wang for helpful resources related to [guided LDA](http://scignconsulting.com/2019/03/09/guided-lda/) and [Yelp review scraping](https://github.com/nuwapi/DoctorSnapshot), respectively.
