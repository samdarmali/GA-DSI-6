# Capstone Project - Video Games Recommender System

## So Why a Recommender System? (The Business Problem)

E-commerce companies today have large catalogues of products, many of which go unnoticed by consumers. Building an effective recommender system improves the consumer experience by helping them discover products that they end up loving and would not have otherwise found. This in turn leads to higher conversion rates, increased revenue and business growth.

## The Data

I used data from the Registry of Open Data on AWS (found [here](https://registry.opendata.aws/amazon-reviews/)).

## Overview of Files

1. [EDA](https://github.com/samdarmali/GA-DSI-6/blob/master/VGRecommender/EDA.ipynb) - Reading in data, cleaning the data, drawing initial insights from the data before modelling.

2. [Cosine Recommender](https://github.com/samdarmali/GA-DSI-6/blob/master/VGRecommender/Cosine_Recommender.ipynb) - This was my attempt at building my own recommender using user-based and item-based collaborative filtering from scratch. It involved an understanding of how user-based and item-based collaborative filtering worked, followed by an understanding of using cosine similarity to determine how similar user/item vectors are to each other. Further, it required an understanding of the various KNN algorithms used to calculate a predicted rating. The Surprise library was ultimately used to build the recommender system but this was good for understanding and experimentation.

3. [MF Breakdown](https://github.com/samdarmali/GA-DSI-6/blob/master/VGRecommender/MF_Breakdown.ipynb) - A breakdown of a matrix factorisation class to understand each step involved and the use of stochastic gradient descent to reach the local optimum.

4. [MF Recommender](https://github.com/samdarmali/GA-DSI-6/blob/master/VGRecommender/MF_Recommender.ipynb) - This was my attempt at building my own recommender using matrix factorization from scratch. It involved an understanding of latent factor models and an understanding of stochastic gradient descent to perform updates on user/item latent factor matrices (P & Q) and user/item biases. The Surprise library was ultimately used to build the recommender system but this was good for understanding and experimentation.

5. [Surprise CF](https://github.com/samdarmali/GA-DSI-6/blob/master/VGRecommender/Surprise_CF.ipynb) - This is the first notebook used to conduct robust testing on recommender algorithms, both user and item based collaborative filtering, using various types of similarity scoring. It was also a chance to see which similarity scoring methods worked better with different types of data (for example, with more sparse data, cosine similarity seemed to perform marginally better than pearson correlation, but reducing sparsity and using pearson correlation improved results overall). Additionally, it was a chance to see which prediction algorithms performed better, e.g. whether to factor in mean ratings, global baseline estimates or normalise with z-scores of ratings. It turned out that factoring in mean ratings and global baseline estimates worked best for both forms of collaborative filtering. The best user-based and item-based models were selected from this notebook.

6. [Surprise CF Alternative](https://github.com/samdarmali/GA-DSI-6/blob/master/VGRecommender/Surprise_CF_Alternative.ipynb) - This is a continuation of the Surprise CF notebook with more testing on other similarity scoring methods. 

7. [Surprise MF](https://github.com/samdarmali/GA-DSI-6/blob/master/VGRecommender/Surprise_MF.ipynb) - This notebook tested the different parameters to find the optimal matrix factorisation model (SVD & SVD++). Althought matrix factorisation was more effective at lowering root mean squared error of predictions, it remained poor in achieving good scores on on precision and mean average precision. The best matrix factorisation model was selected from this notebook but not used in the deployment stage.

8. [Webapp](https://github.com/samdarmali/GA-DSI-6/tree/master/VGRecommender/webapp) - This folder holds the front-end html and css files for the website and the flask backend file. However, due to the large size of the data, it is not in this repository. Hosting this webapp publicly is still in progress.

9. [Presentation](https://github.com/samdarmali/GA-DSI-6/blob/master/VGRecommender/Presentation.pdf) - This pdf file shows the final presentation given at the end of the course, outlining the purpose of the project, methodology, scoring metrics, results and further actions to be taken.

