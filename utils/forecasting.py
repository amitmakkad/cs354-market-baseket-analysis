import pandas as pd
from math import ceil
import matplotlib.pyplot as plt

def preprocess(ecommerce_df):

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
    return df


def visualize(foreceasting_df):

    sales_by_date = foreceasting_df.groupby(by='Date')['TotalPrice'].sum().reset_index()
    sales_by_date.plot(x='Date',y='TotalPrice',kind='line')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Daily Sales')
    plt.show()


    sales_by_week = foreceasting_df.groupby(by=['Year','Week'])['TotalPrice'].sum().reset_index()
    sales_by_week['index'] = sales_by_week.index
    sales_by_week.plot(x='index' ,y='TotalPrice',kind='line')
    plt.xlabel('Week')
    plt.ylabel('Sales')
    plt.title('Weekly Sales')
    plt.show()