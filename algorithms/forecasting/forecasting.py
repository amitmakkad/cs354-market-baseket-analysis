import pandas as pd

from fpgrowth_py import fpgrowth
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

class Predictions:

    association: any

    def __init__(self):
        pass
    
    def get_associate_ruls(self, prediction_df):
        freqItemSet, rules = fpgrowth((prediction_df['StockCode'].values), minSupRatio=0.005, minConf=0.3)
        self.association = pd.DataFrame(rules,columns =['basket','next_product','proba']) 
        selfassociation = self.association.sort_values(by='proba',ascending=False)

        print('Number of rules generated : ', len(rules))
        print('Dimensions of the association table are : ', self.association.shape)

        print(self.association.head(10))
        return self.association                


    def compute_next_best_product(self, basket_el):

        association = self.association
        
        for k in basket_el: # for each element in the consumer basket
                k={k}
                if len(association[association['basket']==k].values) !=0: # if we find a corresponding association in the fp growth table
                    next_pdt=list(association[association['basket']==k]['next_product'].values[0])[0] # we take the consequent product
                    if next_pdt not in basket_el : # We verify that the customer has not previously purchased the product
                        proba=association[association['basket']==k]['proba'].values[0] # Find the associated probability. 
                        return(next_pdt,proba)
        
        return(0,0) # return (0,0) if no product was found. 
    
    def find_next_product(self, basket):

        association = self.association

        n=basket.shape[0]
        list_next_pdt=[]
        list_proba=[]
        for i in range(n): # for each customer
            el=set(basket['StockCode'][i]) # customer's basket
            if len(association[association['basket']==el].values) !=0: # if we find a association in the fp growth table corresponding to all the customer's basket.
                next_pdt=list(association[association['basket']==el]['next_product'].values[0])[0] # We take the consequent product
                proba=association[association['basket']==el]['proba'].values[0] # Probability associated in the table
                list_next_pdt.append(next_pdt)
                list_proba.append(proba)


            elif len(association[association['basket']==el].values) ==0: # If no antecedent to all the basket was found in the table
                next_pdt,proba= self.compute_next_best_product(basket['StockCode'][i]) # previous function
                list_next_pdt.append(next_pdt)
                list_proba.append(proba)
                
        return(list_next_pdt, list_proba)

    def get_predictions(self, ecommerce_df, prediction_df):

        list_next_pdt, list_proba= self.find_next_product(prediction_df) 
        prediction_df['Recommended Product']=list_next_pdt # Set of recommended products
        prediction_df['Probability']=list_proba # Set of rprobabilities associated

        df = ecommerce_df.copy()
        df = df.loc[df["Quantity"] >0 ]
        df = df.loc[df["UnitPrice"] >0 ]
        df["TotalPrice"] = df["Quantity"]*df["UnitPrice"]
        df.dropna(axis = 0, inplace=True)
        df.isnull().sum()

        stock_codes = df['StockCode'].unique() 
        gift_stock_codes = []
        for sc in stock_codes:
            if sc[0] not in [str(i) for i in range(1,11)]: # products corresponding to gifts. 
                gift_stock_codes.append(sc)

        df=df[df['StockCode'].map(lambda x: x not in gift_stock_codes)] # delete these products

        prediction_df=prediction_df.rename(columns = {'StockCode': 'Customer basket'})
        data_stock=df.drop_duplicates(subset ="StockCode", inplace = False)
        prices=[]
        description_list=[]
        for i in range(prediction_df.shape[0]):
            stockcode=prediction_df['Recommended Product'][i]
            probability= prediction_df['Probability'][i]
            if stockcode != 0:
                unitprice=data_stock[data_stock['StockCode']==stockcode]['UnitPrice'].values[0]
                description=data_stock[data_stock['StockCode']==stockcode]['Description'].values[0]
                estim_price=unitprice*probability
                prices.append(estim_price)
                description_list.append(description)
                
            else :
                prices.append(0)
                description_list.append('Null')

            

        prediction_df['Price estimation']=prices 
        prediction_df['Product description']=description_list 
        prediction_df = prediction_df.reindex(columns=['Customer basket','Recommended Product','Product description','Probability','Price estimation'])
        return prediction_df

    def get_sales_by_date(self, sales_by_date, p, d, q, num_predictions):

        diff_data = sales_by_date.diff().dropna()

        train_size = int(len(diff_data) * 0.8)
        train, test = diff_data[0:train_size], diff_data[train_size:]

        model = ARIMA(diff_data, order=(p, d, q))
        model_fit = model.fit()

        predictions = model_fit.predict(start=len(train), end=len(diff_data)-1, typ='levels')

        rmse = np.sqrt(mean_squared_error(test, predictions))
        print('Test RMSE: %.3f' % rmse)


        diff_data = sales_by_date.diff().dropna()
        model = ARIMA(diff_data, order=(p, d, q))
        model_fit = model.fit()
        last_date = sales_by_date.index[-1]
        future_dates = pd.date_range(start=last_date, periods=num_predictions, freq='D')
        forecast_diff = model_fit.forecast(steps=num_predictions)
        forecast = sales_by_date['TotalPrice'].iloc[-1] + forecast_diff.cumsum()

        forecast_df = pd.DataFrame({'Date': future_dates, 'Sales': forecast})
        forecast_df = forecast_df.set_index('Date')

        sales = pd.concat([sales_by_date, forecast_df], axis=0)

        plt.plot(sales)
        plt.xlabel('Date')
        plt.ylabel('Predicted')
        plt.title('Dail Sales')
        plt.show()