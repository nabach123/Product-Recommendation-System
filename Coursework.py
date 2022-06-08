
#!/usr/bin/env python
# coding: utf-8

# # Product Recommendation System 

# Recommendation System Part 1:
# Product pupularity based system targetted at new customers

# Importing libraries
import numpy as np #numpy provides support for large multidimensional array objects and various tools to work with them. 
import pandas as pd # Pandas is a Python library for data analysis.
import matplotlib.pyplot as plt  #Matplotlib is a plotting library for creating static, animated, and interactive visualizations in Python. 

# %matplotlib inline
#plt.style.use("ggplot")

import sklearn  #It provides a selection of efficient tools for machine learning and statistical modeling including classification, 
#regression, clustering and dimensionality reduction via a consistence interface in Python.
from sklearn.decomposition import TruncatedSVD # it is a matrix factorization technique that factors a matrix M into the three matrices U, Σ, and V. 

# loading the dataset
ratings = pd.read_csv('ratings_Beauty.csv')
ratings = ratings.dropna() #Pandas dropna() method allows the user to analyze and drop Rows/Columns with Null values in different ways.
famous_products = pd.DataFrame(ratings.groupby('ProductId')['Rating'].count()) 
most_popular = famous_products.sort_values('Rating', ascending=False) 

# Recommendation System Part 2:
# Model based collaborative filtering system based on customer’s purchase history and ratings provided by other users who bought similar items.

# Subset of Amazon Ratings
ratings1 = ratings.head(10000)
ratings_utility_matrix = ratings1.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)
X = ratings_utility_matrix.T
X1 = X

#Decomposing the Matrix
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape

#Correlation Matrix
correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape

#Isolating Product ID 
#Assuming the customer buys Product ID 6117043058 #  (randomly chosen)
X.index[100]

# Index of product ID purchased by customer
i = "6117043058"
product_names = list(X.index)
product_ID = product_names.index(i)

# Correlation between all products purchased by this client and 
# things rated by other customers who purchased the same product
correlation_product_ID = correlation_matrix[product_ID]


# Recommending top 10 highly correlated products in sequence
Recommend = list(X.index[correlation_product_ID > 0.90])

# Removes the item already bought by the customer
Recommend.remove(i) 

Recommend[0:10]

# Recommendation System Part 3:
# Item to item based recommendation system based on product description
# Importing libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Loading the dataset
product_descriptions = pd.read_csv('product_descriptions.csv')
product_descriptions.shape

# Checking for missing values
product_descriptions = product_descriptions.dropna()
product_descriptions.shape

product_descriptions1 = product_descriptions.head(500)
# product_descriptions1.iloc[:,1]
product_descriptions1["product_description"].head(10)


# Feature extraction from product descriptions
# Converting the text in product description into numerical data for analysis
vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(product_descriptions1["product_description"])

# Visualizing product clusters in subset of data
# Fitting K-Means to the dataset
X=X1
kmeans = KMeans(n_clusters = 10, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)
plt.plot(y_kmeans, ".")
#plt.show()

def show_cluster(i):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

# Optimal clusters is 

true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X1)

#print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

# Predicting clusters based on key search words
def product_recommend(product):
   
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    #print(prediction)
    show_cluster(prediction[0])




# this file is run as main program
if __name__ == "__main__":

        # checking condition to continue next instance

    print(" Welcome !!! Enjoy your Shopping")
    print("")
       
    condition= input('Is rating data available?(y/n):')
    if  condition == "y":
        condition1 = input("Does customer has a purchased history?(y/n): ")
        if  condition1 == "y":
            print("Recommending top 10 highly correlated products in sequence")
            # recommending top 10 highly correlated products in sequence"
            print(Recommend[0:10])
            print("")
            print("Thank You")
        else:
            print("Most popular products (arranged in descending order) based on Rating")
            print(most_popular.head(10))
            print("")
            print("Thank You")

    else:
        print("Top terms per cluster:")
        for i in range(true_k):
            show_cluster(i)
         # take the item name input
        itemName = input("Enter item name: ")
        product_recommend(itemName)
        print("")
       
            



           
                
  
  
            
       
    











