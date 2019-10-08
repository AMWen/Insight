from flask import Flask, Markup, render_template, request
import pandas as pd
import Levenshtein as lev
import pickle
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load data
data = pd.read_csv("./data/all_hospitals_cleaned.csv")

# Rounded values for slider bar
def percentoverall(value, column):
    temp = (value-column.min())/(column.max()-column.min())*100
    return 25*round(temp/25)

# Extract values from slider bar
def percentresult(value, column):
    return value/100*(column.max()-column.min())+column.min()

# Load random forest regression fit
rf = pickle.load(open("./data/rf.pickle", "rb"))

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    # Initialize values to MGH
    init = 21
    hospital_name = ""
    message = " "
    sorted_data = data.loc[init:init+2].reset_index(drop=True)
    experience=round(sorted_data.loc[0]['Overall Experience'], 1)
    timeliness=round(sorted_data.loc[0]['Timeliness'], 1)
    cleanliness=round(sorted_data.loc[0]['Cleanliness'], 1)
    helpfulness=round(sorted_data.loc[0]['Helpfulness'], 1)
    communication=round(sorted_data.loc[0]['Communication'], 1)
    quietness=round(sorted_data.loc[0]['Quietness'], 1)
    overall=round(sorted_data.loc[0]['Average Rating'], 1)

    # Labels and values for bar graph
    labels = ['Overall', 'Timeliness', 'Cleanliness', 'Helpfulness', 'Communication', 'Quietness']
    values = [experience, timeliness, cleanliness, helpfulness, communication, quietness]
    
    rounded_values = [percentoverall(experience, data['Overall Experience']),
                      percentoverall(timeliness, data['Timeliness']),
                      percentoverall(cleanliness, data['Cleanliness']),
                      percentoverall(helpfulness, data['Helpfulness']),
                      percentoverall(communication, data['Communication']),
                      percentoverall(quietness, data['Quietness'])]
    
    # If search was performed
    if request.method == "POST":
        
        # Get the hospital name
        hospital_name = dict(request.form)["name"]
     
        # Initialize all string distance ratios as 0
        data['Distance Ratio'] = 0
        
        # Fuzzy string matching with hospital list with Levenshtein distance
        if hospital_name != "":
            for index, row in data.iterrows():
                data.loc[index, 'Distance Ratio'] = lev.ratio(hospital_name.lower(), row['Hospital Name'].lower())
            sorted_data = data.reset_index(drop=True)
        else:
            sorted_data = data.loc[init:init+2].reset_index(drop=True)

        # Sort the hospitals by match distance
        sorted_data = sorted_data.sort_values(['Distance Ratio'], ascending=False).reset_index(drop=True)

        if dict(request.form)["button"] == "Search/Reset":
            # Extract and round values for each topic
            experience=round(sorted_data.loc[0]['Overall Experience'], 1)
            timeliness=round(sorted_data.loc[0]['Timeliness'], 1)
            cleanliness=round(sorted_data.loc[0]['Cleanliness'], 1)
            helpfulness=round(sorted_data.loc[0]['Helpfulness'], 1)
            communication=round(sorted_data.loc[0]['Communication'], 1)
            quietness=round(sorted_data.loc[0]['Quietness'], 1)
            overall=round(sorted_data.loc[0]['Average Rating'], 1)
            
        else:
            experience = percentresult(int(dict(request.form)["Overall"]), data['Overall Experience'])
            timeliness = percentresult(int(dict(request.form)["Timeliness"]), data['Timeliness'])
            cleanliness = percentresult(int(dict(request.form)["Cleanliness"]), data['Cleanliness'])
            helpfulness = percentresult(int(dict(request.form)["Helpfulness"]), data['Helpfulness'])
            communication = percentresult(int(dict(request.form)["Communication"]), data['Communication'])
            quietness = percentresult(int(dict(request.form)["Quietness"]), data['Quietness'])
            
            # Predict rating based on values
            X = pd.DataFrame({'Overall Experience': experience, 'Timeliness': timeliness,
                             'Cleanliness': cleanliness, 'Friendliness': round(sorted_data.loc[0]['Friendliness'], 1),
                             'Quietness': quietness, 'Helpfulness': helpfulness,
                             'Communication': communication}, index=[0])
            old=round(sorted_data.loc[0]['Average Rating'], 1)
            overall = round(float(rf.predict(X)), 1)
            
            if overall > old:
                change = round((overall-old)/old*100, 1)
                message = str(change)+"% increase!"
            else:
                message = " "

        values = [experience, timeliness, cleanliness, helpfulness, communication, quietness]
            
        rounded_values = [percentoverall(experience, data['Overall Experience']),
                          percentoverall(timeliness, data['Timeliness']),
                          percentoverall(cleanliness, data['Cleanliness']),
                          percentoverall(helpfulness, data['Helpfulness']),
                          percentoverall(communication, data['Communication']),
                          percentoverall(quietness, data['Quietness'])]
    
    return render_template('index.html', hospital_name=hospital_name,
                           selected=sorted_data.loc[0],
                           name=sorted_data.loc[0]['Hospital Name'],
                           summary=sorted_data.loc[0]['Summary'],
                           labels=labels,
                           values=values,
                           overall=overall,
                           rounded_values=rounded_values,
                           message=message)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
