# cs354-market-baseket-analysis
Our problem is to analyze customer purchase behavior, recommend products based on their purchase history, and forecast sales for an ecommerce company.

<h2>Our proposed Solution:-</h2>

 - Data Understanding and Preprocessing
 - Customer purchase behaviour
   - First, we make customer segments using clustering
   - Second, we find association rules between items
   - Third, we find probability of buying recommended product
 - Sales forecasting
   - For a basket, we add the expected cost of buying recommended product
   - Then we used forcasting model to find expected sales


<h2>Important files and folder structure:-</h2>

 - Algorithms
   - Associate_rule_mining - It contains apriori and fpgrowth algorithm which used for finding association between different products
   - Clustering - It contains implementation of k means, dbscan and hierarchical clustering used for customer segmentation
   - Forecasting - It contains code for Arima model to predict future sales
 - Data - It contains E commerce dataset used along with some intermediate generated data files
 - Utils - It contains functions to find best customer clusters and best association rules
 - Pipeline.ipynb - It contains code for main pipeline of our project which executes all other functions.

<h2>Contributers:-</h2>

 - Amit Kumar Makkad
 - Mihir Karandikar

<h2>Project Installation Guide</h2>

 - Clone this repository in your local machine
 - Open terminal in the project directory
 - Install all the dependencies
 - Now, you can run pipeline.ipynb file 

This project is part of course cs354 Computation Intelligence Lab IITI under guidance of prof Aruna Tiwari.
