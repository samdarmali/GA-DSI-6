# Flask & SQL
from flask import *
import sqlite3, hashlib, os
from werkzeug.utils import secure_filename

# Recommender libraries
from surprise import KNNBasic, KNNWithMeans, KNNBaseline, KNNWithZScore
from surprise import Dataset
from surprise import Reader
import pandas as pd
import numpy as np


''' ------- RECOMMENDER CLASS ------- '''
class Recommender():

    def __init__(self, dataset):
        '''
        Class which returns recommendations to a new customer.
        Initializes training data based on a full dataset.
        Initializes an item-item and a user-user recommender.

        Item-Item Recommender:
        - algorithm  :  KNNBaseline
        - K          :  21
        - sim        :  pearson correlation

        User-User Recommender:
        - algorithm  :  KNNwithMeans
        - K          :  12
        - sim        :  pearson correlation

        (for more information, see Surprise_CF.ipynb)
        '''
        self.dataset = dataset
        self.ii_algo = KNNBaseline(k=21, sim_options={'name': 'pearson', 'user_based': False})
        self.uu_algo = KNNWithMeans(k=12, sim_options={'name': 'pearson', 'user_based': True})

    def new_recommendations(self, new_products):
        '''
        Function that takes in a list of new products and returns recommendations.

        Arguments:
        - new_products   :  list of products chosen by new user
        - orig_data      :  original dataframe of users, items and ratings

        Returns:
        - recs_df        :  dataframe of recommendations
        '''
        # Append new customer to data
        new_data = pd.DataFrame({'customer_id':[1]*len(new_products),
                                 'product_title':new_products,
                                 'star_rating':[5]*len(new_products)})
        full_data = pd.concat([new_data, self.dataset]).reset_index(drop=True)

        # Build dataset
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(full_data[['customer_id', 'product_title', 'star_rating']], reader)
        trainset = data.build_full_trainset()

        # Train model
        self.ii_algo.fit(trainset)

        # Make recommendations
        recommendations = {'items': [], 'rating': []}
        for item in self.dataset['product_title'].unique():
            rating = self.ii_algo.predict(1, item, verbose=False)[3]
            recommendations['items'].append(item)
            recommendations['rating'].append(rating)
        recs_df = pd.DataFrame(recommendations).sort_values(by='rating', ascending=False)
        recs = recs_df.head(15)['items']

        return recs
''' ------- RECOMMENDER CLASS END------- '''


''' ------- READ IN DATA ------- '''
sqlite_db = '../datasets/amzn_vg_clean.db'
conn = sqlite3.connect(sqlite_db)
query = '''
SELECT "customer_id", "product_title", "star_rating"
FROM video_games
'''
dataset = pd.read_sql(query, con=conn)
''' ------- READ IN DATA END ------- '''


''' ------- FLASK APPLICATION ------- '''
app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/best-sellers.html/')
def best_sellers_page():
    with sqlite3.connect('../datasets/amzn_vg_clean.db') as conn:
        cur = conn.cursor()
        cur.execute('SELECT * FROM products')
        itemData = cur.fetchall()
    return render_template('best-sellers.html', itemData=itemData)


@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if request.method == 'POST':

        recommender = Recommender(dataset)

        selections = request.form.getlist('product')
        print(selections)

        recs = recommender.new_recommendations(selections)

        selected_list = []
        for id in selections:
            with sqlite3.connect('../datasets/amzn_vg_clean.db') as conn:
                full_query = 'SELECT * FROM products WHERE product_id = "{}"'.format(id)
                cur = conn.cursor()
                cur.execute(full_query)
                selected_list.extend(cur.fetchall())

        recommended_list = []
        for name in recs:
            with sqlite3.connect('../datasets/amzn_vg_clean.db') as conn:
                full_query = 'SELECT * FROM products WHERE product_title = "{}"'.format(name)
                cur = conn.cursor()
                cur.execute(full_query)
                recommended_list.extend(cur.fetchall())

        return render_template('results.html', selections=selections, recs=recs, selected_list=selected_list, recommended_list=recommended_list)


if __name__ == '__main__':
    HOST = '127.0.0.1'
    PORT = 5000
    app.run(HOST, PORT, debug=True)
''' ------- FLASK APPLICATION END ------- '''
