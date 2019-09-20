from flask import Flask, Markup, render_template, request
import pandas as pd
import Levenshtein as lev

app = Flask(__name__)

# Load data
data = pd.read_csv("./data/all_hospitals.csv")

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    # Initialize values to first hospital
    hospital_name = ""
    sorted_data = data
    parking=round(sorted_data.loc[0]['Parking'], 1)
    waiting=round(sorted_data.loc[0]['Wait Time'], 1)
    payments=round(sorted_data.loc[0]['Payments'], 1)
    communication=round(sorted_data.loc[0]['Communication'], 1)
    quality=round(sorted_data.loc[0]['Quality of Care'], 1)
    overall=round(sorted_data.loc[0]['Average Rating'], 1)

    labels = ['Logistics', 'Timeliness', 'Billing', 'Communication', 'Quality of Care']
    values = [parking, waiting, payments, communication, quality]
    
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

        # Sort the hospitals by match distance
        sorted_data = data.sort_values(['Distance Ratio'], ascending=False).reset_index(drop=True)

        if dict(request.form)["button"] == "Search/Reset":
            # Extract and round values for each topic
            parking = round(sorted_data.loc[0]['Parking'], 1)
            waiting = round(sorted_data.loc[0]['Wait Time'], 1)
            payments = round(sorted_data.loc[0]['Payments'], 1)
            communication = round(sorted_data.loc[0]['Communication'], 1)
            quality = round(sorted_data.loc[0]['Quality of Care'], 1)
            overall = round(sorted_data.loc[0]['Average Rating'], 1)
        
        else:
            parking = int(dict(request.form)["Parking"])
            waiting = int(dict(request.form)["Waiting"])
            payments = int(dict(request.form)["Payments"])
            communication = int(dict(request.form)["Communication"])
            quality = int(dict(request.form)["Quality"])
            a = .02
            b = c = d = e = .005/4
            overall = 2.5+round(a*parking+b*waiting+c*payments+d*communication+e*quality, 1)

        values = [parking, waiting, payments, communication, quality]
    
    return render_template('index.html', hospital_name=hospital_name,
                           selected=sorted_data.loc[0],
                           name=sorted_data.loc[0]['Hospital Name'],
                           summary=sorted_data.loc[0]['Summary'],
                           parking=parking,
                           waiting=waiting,
                           payments=payments,
                           communication=communication,
                           quality=quality,
                           overall=overall,
                           labels=labels,
                           values=values)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run()
