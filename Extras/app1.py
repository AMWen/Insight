from flask import Flask, Markup, render_template
import os
from flask import send_from_directory

app = Flask(__name__)

labels = [
          'JAN', 'FEB', 'MAR', 'APR',
          'MAY', 'JUN', 'JUL', 'AUG',
          'SEP', 'OCT', 'NOV', 'DEC'
          ]

values = [
          967.67, 1190.89, 1079.75, 1349.19,
          2328.91, 2504.28, 2873.83, 4764.87,
          4349.29, 6458.30, 9907, 16297
          ]

colors = [
          "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
          "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
          "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]

@app.route('/')
@app.route('/index')
def index():
    bar_labels=labels
    bar_values=values
    return render_template('index.html', title='hi', max=17000, labels=bar_labels, values=bar_values)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/bar')
def bar():
    return render_template('bar_chart.html', labels=labels, values=values, colors=colors, title='hi', max=17000)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
