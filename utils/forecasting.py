import pandas as pd
from math import ceil
import matplotlib.pyplot as plt
from datetime import datetime

def time_series_preprocess(ecommerce_df, price_add_by_date):

    df = ecommerce_df.copy()
    
    df = df.loc[df["Quantity"] >0 ]
    df = df.loc[df["UnitPrice"] >0 ]
    df["TotalPrice"] = df["Quantity"]*df["UnitPrice"]
    df.dropna(axis = 0, inplace=True)
    df.isnull().sum()

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Year'] = df.InvoiceDate.dt.year
    df['Month'] = df.InvoiceDate.dt.month
    df['Week'] = df.InvoiceDate.dt.isocalendar().week
    df['Day'] = df.InvoiceDate.dt.day
    df['Quarter'] = df.Month.apply(lambda m:'Q'+str(ceil(m/4)))
    df['Date'] = pd.to_datetime(df[['Year','Month','Day']])
    df = df.astype({'CustomerID':'int'})
    df = df.astype({'StockCode': 'string'})

    sales_by_date = df.groupby(by='Date')['TotalPrice'].sum().reset_index()
    sales_by_date = sales_by_date.set_index('Date')

    additional_sales = []
    for date_time in sales_by_date.index:
        datetime_obj = datetime.strptime(str(date_time), '%Y-%m-%d %H:%M:%S')
        date = datetime_obj.strftime('%Y-%m-%d')
        additional_sales.append(price_add_by_date[date])
    
    sales_by_date['AdditionalSales'] = additional_sales
    sales_by_date['TotalPrice'] += sales_by_date['AdditionalSales']
    sales_by_date.drop('AdditionalSales', axis=1, inplace=True)

    return sales_by_date

def prediction_preprocess(ecommerce_df):

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

    df = df.groupby(['InvoiceNo','CustomerID']).agg({'StockCode': lambda s: list(set(s))}) # grouping product from the same invoice. 
    print(df.head())
    return df

def get_price_add_by_date(ecommerce_df, prediction_df):

    df = ecommerce_df.copy()
    
    df = df.loc[df["Quantity"] >0 ]
    df = df.loc[df["UnitPrice"] >0 ]
    df["TotalPrice"] = df["Quantity"]*df["UnitPrice"]
    df.dropna(axis = 0, inplace=True)
    df.isnull().sum()

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Year'] = df.InvoiceDate.dt.year
    df['Month'] = df.InvoiceDate.dt.month
    df['Week'] = df.InvoiceDate.dt.isocalendar().week
    df['Day'] = df.InvoiceDate.dt.day
    df['Quarter'] = df.Month.apply(lambda m:'Q'+str(ceil(m/4)))
    df['Date'] = pd.to_datetime(df[['Year','Month','Day']])
    df = df.astype({'CustomerID':'int'})
    df = df.astype({'StockCode': 'string'})

    customer_sales_by_date = df.groupby(['CustomerID', 'Date']).agg({
        'StockCode': lambda x: list(x),
        'TotalPrice': sum
    })

    price_add_by_date = dict()

    for (customer_id, date_time), customer_basket in (zip(customer_sales_by_date.index, customer_sales_by_date['StockCode'])):

        datetime_obj = datetime.strptime(str(date_time), '%Y-%m-%d %H:%M:%S')
        date = datetime_obj.strftime('%Y-%m-%d')

        
        for precedent_basket, recommended_product, price_esitmation in zip(prediction_df['Customer basket'], prediction_df['Recommended Product'], prediction_df['Price estimation']):
            if set(precedent_basket).issubset(set(customer_basket)):
                price_add_by_date[date] = price_add_by_date.get(date, 0) + price_esitmation

    return price_add_by_date


def visualize(sales_by_date):

    plt.plot(sales_by_date)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Daily Sales')
    plt.show()
