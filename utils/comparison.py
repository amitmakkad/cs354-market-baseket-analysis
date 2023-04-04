import sys
sys.dont_write_bytecode = True

import pandas as pd

from algorithms.associate_rule_mining import apriori


def get_best_clusters(ecommerce_df, customer_clusters_kmeans, customer_clusters_dbscan, customer_clusters_hierarchial):

    apriori_model = apriori.Apriori()
    
    associate_rules_kmeans = apriori_model.get_associate_rules(ecommerce_df, customer_clusters_kmeans)
    associate_rules_dbscan = apriori_model.get_associate_rules(ecommerce_df, customer_clusters_dbscan)
    associate_rules_hierarchial = apriori_model.get_associate_rules(ecommerce_df, customer_clusters_hierarchial)

    associate_rules_kmeans.rename(columns={"lift": "lift1"}, inplace=True)
    associate_rules_kmeans.drop(['leverage', 'conviction','antecedent support','consequent support','support','confidence'], axis=1, inplace=True)

    associate_rules_dbscan.rename(columns={"lift": "lift2"}, inplace=True)
    associate_rules_dbscan.drop(['leverage', 'conviction','antecedent support','consequent support', 'support','confidence'], axis=1, inplace=True)

    associate_rules_hierarchial.rename(columns={"lift": "lift3"}, inplace=True)
    associate_rules_hierarchial.drop(['leverage', 'conviction','antecedent support','consequent support','support','confidence'], axis=1, inplace=True)

    df=pd.merge(pd.merge(associate_rules_kmeans,associate_rules_dbscan,on=['antecedents','consequents'], how="inner"), associate_rules_hierarchial, on=['antecedents','consequents'], how="inner")

    kmeans = 0
    dbscan = 0
    hierar = 0

    for index, row in df.iterrows():
        l1=row['lift1']
        l2=row['lift2']
        l3=row['lift3']
        if l1>l2 and l1>l3:
            kmeans=kmeans+1
        elif l2>l1 and l2>l3:
            dbscan=dbscan+1
        elif l3>l1 and l3>l2:
            hierar=hierar+1

    if kmeans==max(kmeans,dbscan,hierar):
        print("best cluster by kmeans")
        return customer_clusters_kmeans
    elif dbscan==max(kmeans,dbscan,hierar):
        print("best cluster by dbscan")
        return customer_clusters_dbscan
    elif hierar==max(kmeans,dbscan,hierar):
        print("best cluster by hierar")
        return customer_clusters_hierarchial
    
def get_best_associate_rules(associate_rules_apriori, associate_rules_fpgrowth):
    
    associate_rules_apriori['zhang metric1'] = None

    for i, row in associate_rules_apriori.iterrows():
        supp_x = row['antecedent support']
        supp_y = row['consequent support']
        supp_xy = row['support']
        max_supp = max(supp_x, supp_y)
        
        zhang_metric = (supp_xy - supp_x * supp_y) / (max_supp - supp_x * supp_y)
        associate_rules_apriori.at[i, 'zhang metric1'] = zhang_metric

    associate_rules_apriori.drop(['leverage', 'conviction','antecedent support','consequent support','support','confidence','lift'], axis=1, inplace=True)


    associate_rules_fpgrowth['zhang metric2'] = None
    
    for i, row in associate_rules_fpgrowth.iterrows():
        supp_x = row['antecedent support']
        supp_y = row['consequent support']
        supp_xy = row['support']
        max_supp = max(supp_x, supp_y)
        
        zhang_metric = (supp_xy - supp_x * supp_y) / (max_supp - supp_x * supp_y)
        associate_rules_fpgrowth.at[i, 'zhang metric2'] = zhang_metric

    associate_rules_fpgrowth.rename(columns={"zhang metric": "zhang metric2"}, inplace=True)
    associate_rules_fpgrowth.drop(['leverage', 'conviction','antecedent support','consequent support','support','confidence','lift'], axis=1, inplace=True)

    print(associate_rules_apriori)
    print(associate_rules_fpgrowth)

    df=pd.merge(associate_rules_apriori,associate_rules_fpgrowth,on=['antecedents','consequents'],how="inner")
    print(df)

    ap=0
    fp=0
    for index, row in df.iterrows():
        z1=row['zhang metric1']
        z2=row['zhang metric2']
        if z1>z2:
            ap=ap+1
        else:
            fp=fp+1

    if ap==max(ap,fp):
        print("best rules by apriori")
        return associate_rules_apriori
    else:
        print("best rules by apriori")
        return associate_rules_fpgrowth