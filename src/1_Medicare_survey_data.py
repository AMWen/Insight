### Medicare survey data

# Import necessary packages
import pandas as pd
import numpy as np

# Load Medicare survey data
survey_df = pd.read_csv('data/HCAHPS - Hospital.csv')

# Replace 'Not Available' with NaN
survey_df.replace('Not Available', np.nan, inplace=True)

# Remove hospitals with fewer than 200 completed surveys
survey_df['Number of Completed Surveys'] = survey_df['Number of Completed Surveys'].astype(float)
survey_df = survey_df[survey_df['Number of Completed Surveys']>200]

# Change capitalization of 'Hospital Name' and 'Address' to make it easier to read
survey_df['Hospital Name'] = survey_df['Hospital Name'].apply(lambda x: x.title())
survey_df['Address'] = survey_df['Address'].apply(lambda x: x.title())

# Pivot table so each row is a hospital and responses are in columns
survey_df_wide = pd.pivot(survey_df, index='Provider ID', columns='HCAHPS Answer Description',
                          values='HCAHPS Answer Percent').reset_index()

# Check that everything is in order
survey_df_wide.head()

# Add hospital name and state to the wide dataframe
survey_df_wide = pd.merge(survey_df[["Provider ID", "Hospital Name", "State", "Address"]].drop_duplicates(),
                          survey_df_wide, on=['Provider ID'])

# Look at columns and extract only desired columns
survey_df_wide.columns
survey_df_smaller = survey_df_wide[['Hospital Name', 'State', 'Address',
                                    '"Always" quiet at night', '"Usually" quiet at night',
                                    'Room was "always" clean', 'Room was "usually" clean',
                                    'Patients "always" received help as soon as they wanted', 'Patients "usually" received help as soon as they wanted',
                                    'Doctors "always" communicated well', 'Doctors "usually" communicated well',
                                    'Nurses "always" communicated well', 'Nurses "usually" communicated well',
                                    'Staff "always" explained', 'Staff "usually" explained']]
survey_df_smaller.head()

# Drop hospitals without responses and extract only hospitals in Masssachusetts and New York
survey_df_smaller = survey_df_smaller.dropna(axis=0)
survey_df_smaller = survey_df_smaller[(survey_df_smaller['State'] == 'MA') | (survey_df_smaller['State'] == 'NY')]

# Save dataframe as csv file
survey_df_smaller.to_csv("data/Medicare responses.csv", sep=',', index=False)
