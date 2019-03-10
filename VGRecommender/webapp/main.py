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

    def __init__(self, dataset, new_products):
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
        self.new_products = new_products

        # Append new customer to data
        new_data = pd.DataFrame({'customer_id':[1]*len(self.new_products),
                                 'product_id': self.new_products,
                                 'star_rating':[5]*len(self.new_products)})
        full_data = pd.concat([new_data, dataset]).reset_index(drop=True)
        data = Dataset.load_from_df(full_data[['customer_id', 'product_id', 'star_rating']], Reader(rating_scale=(1, 5)))

        self.unique_products = dataset['product_id'].unique()
        self.trainset = data.build_full_trainset()
        self.ii_algo = KNNBaseline(k=21, sim_options={'name': 'pearson', 'user_based': False})
        self.uu_algo = KNNBaseline(k=99, sim_options={'name': 'msd', 'user_based': True})#KNNWithMeans(k=12, sim_options={'name': 'pearson', 'user_based': True})

    def new_recommendations(self):
        '''
        Function that takes in a list of new products and returns recommendations.

        Arguments:
        - new_products   :  list of products chosen by new user
        - orig_data      :  original dataframe of users, items and ratings
        - algo           :  algorithm for predicting ratings

        Returns:
        - recs_df        :  dataframe of recommendations
        '''

        # Train recommender systems
        self.ii_algo.fit(self.trainset)
        self.uu_algo.fit(self.trainset)

        recommendations = {'items': [], 'ii_rating': [], 'uu_rating': []}
        for item in self.unique_products:
            if item not in self.new_products:
                ii_rating = self.ii_algo.predict(1, item, verbose=False)[3]
                uu_rating = self.uu_algo.predict(1, item, verbose=False)[3]
                recommendations['items'].append(item)
                recommendations['ii_rating'].append(ii_rating)
                recommendations['uu_rating'].append(uu_rating)
        recs_df = pd.DataFrame(recommendations)
        ii_recs = recs_df.sort_values(by='ii_rating', ascending=False).head(10)['items']
        uu_recs = recs_df.sort_values(by='uu_rating', ascending=False).head(10)['items']

        return ii_recs, uu_recs

''' ------- RECOMMENDER CLASS END------- '''


''' ------- READ IN DATA ------- '''
sqlite_db = '../datasets/amzn_vg_clean.db'
conn = sqlite3.connect(sqlite_db)
query = '''
SELECT "customer_id", "product_id", "star_rating"
FROM full_dataset
'''
dataset = pd.read_sql(query, con=conn)
print(dataset.shape)
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

@app.route('/xbox-360.html/')
def xbox_360_page():
    with sqlite3.connect('../datasets/amzn_vg_clean.db') as conn:
        cur = conn.cursor()
        cur.execute('SELECT * FROM products WHERE category = "Xbox 360"')
        itemData = cur.fetchall()
    return render_template('xbox-360.html', itemData=itemData)

@app.route('/playstation-3.html/')
def playstation3_page():
    with sqlite3.connect('../datasets/amzn_vg_clean.db') as conn:
        cur = conn.cursor()
        cur.execute('SELECT * FROM products WHERE category = "PlayStation 3"')
        itemData = cur.fetchall()
    return render_template('playstation-3.html', itemData=itemData)

@app.route('/playstation-4.html/')
def playstation4_page():
    with sqlite3.connect('../datasets/amzn_vg_clean.db') as conn:
        cur = conn.cursor()
        cur.execute('SELECT * FROM products WHERE category = "PlayStation 4"')
        itemData = cur.fetchall()
    return render_template('playstation-4.html', itemData=itemData)

@app.route('/wii.html/')
def wii_page():
    with sqlite3.connect('../datasets/amzn_vg_clean.db') as conn:
        cur = conn.cursor()
        cur.execute('SELECT * FROM products WHERE category = "Wii"')
        itemData = cur.fetchall()
    return render_template('wii.html', itemData=itemData)

@app.route('/pc.html/')
def pc_page():
    with sqlite3.connect('../datasets/amzn_vg_clean.db') as conn:
        cur = conn.cursor()
        cur.execute('SELECT * FROM products WHERE category = "PC"')
        itemData = cur.fetchall()
    return render_template('pc.html', itemData=itemData)


@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if request.method == 'POST':

        selections = request.form.getlist('product')
        print(selections)

        recommender = Recommender(dataset, selections)
        ii_recs, uu_recs = recommender.new_recommendations()
        print(ii_recs)
        print(uu_recs)

        selected_list = []
        for id in selections:
            with sqlite3.connect('../datasets/amzn_vg_clean.db') as conn:
                full_query = 'SELECT * FROM products WHERE product_id = "{}"'.format(id)
                cur = conn.cursor()
                cur.execute(full_query)
                selected_list.extend(cur.fetchall())

        ii_recommended_list = []
        for name in ii_recs:
            with sqlite3.connect('../datasets/amzn_vg_clean.db') as conn:
                full_query = 'SELECT * FROM products WHERE product_id = "{}"'.format(name)
                cur = conn.cursor()
                cur.execute(full_query)
                ii_recommended_list.extend(cur.fetchall())

        uu_recommended_list = []
        for name in uu_recs:
            with sqlite3.connect('../datasets/amzn_vg_clean.db') as conn:
                full_query = 'SELECT * FROM products WHERE product_id = "{}"'.format(name)
                cur = conn.cursor()
                cur.execute(full_query)
                uu_recommended_list.extend(cur.fetchall())

        return render_template('results.html', selections=selections, selected_list=selected_list, ii_recommended_list=ii_recommended_list, uu_recommended_list=uu_recommended_list)


if __name__ == '__main__':
    HOST = '127.0.0.1'
    PORT = 5000
    app.run(HOST, PORT, debug=True)
''' ------- FLASK APPLICATION END ------- '''
