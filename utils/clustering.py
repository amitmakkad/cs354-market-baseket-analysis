import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def check_df(df):
    customer_ids = [17850,13047,12583,12346,12347,12348,12349,12350,18280,18281,18282,18283,18287]
    for id in customer_ids:
        try:
            res = df.loc[df['CustomerID'] == id]
            print((res['Amount']).values[0],(res['Frequency']).values[0])
        except:
            print("Error")
            pass

def remove_anomalies(df):

    model = IsolationForest(n_estimators=150, max_samples='auto', contamination=float(0.1), max_features=1.0)
    model.fit(df[["Amount","Frequency"]])

    scores = model.decision_function(df[["Amount","Frequency"]])
    anomaly = model.predict(df[["Amount","Frequency"]])

    df['scores'] = scores
    df['anomaly'] = anomaly

    anomaly = df.loc[df['anomaly']==-1]
    anomaly_index = list(anomaly.index)
    print('Total number of outliers is:', len(anomaly))

    df = df.drop(anomaly_index, axis = 0).reset_index(drop=True)
    df.drop(['scores', 'anomaly'], axis = 1, inplace =True)

    return df

def preprocess(ecommerce_df):

    df = ecommerce_df.copy()

    df.drop(['StockCode', 'InvoiceDate','Description'],axis = 1, inplace =True)
    df = df.loc[df["Quantity"] >0 ]
    df = df.loc[df["UnitPrice"] >0 ]
    df["TotalPrice"] = df["Quantity"]*df["UnitPrice"]
    df.drop(['Quantity', 'UnitPrice'],axis = 1, inplace =True)
    df.dropna(axis = 0, inplace=True)
    df.isnull().sum()

    Amount = df.groupby('CustomerID')['TotalPrice'].sum()
    Amount = Amount.reset_index()
    Amount.columns=['CustomerID','Amount']

    Frequency = df.groupby('CustomerID')['InvoiceNo'].count()
    Frequency = Frequency.reset_index()
    Frequency.columns=['CustomerID','Frequency']

    df = pd.merge(Amount, Frequency, on='CustomerID', how='inner')
    df = df.astype({'CustomerID':'int'})

    df = remove_anomalies(df)

    scaler = StandardScaler()
    df[["Amount","Frequency"]] = scaler.fit_transform(df[["Amount","Frequency"]])

    return df

def get_data(clustering_df):

    X = []

    for idx, row in clustering_df.iterrows():
        X.append([row["Amount"],row["Frequency"]])
    
    X = np.array(X)
    return X