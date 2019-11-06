from tqdm import tqdm_notebook as tqdm
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def elbow_method(df_, k_initial=1, maxClusters=15, k_iteration=1):
    # to validing K value in K-means
    print('Executing Elbow Method to:\n...K of {} to {} from k_iteration:{}\n'.format(k_initial,maxClusters, k_iteration))
    inertia_dic = {}
    for k in tqdm(range(k_initial, maxClusters, k_iteration)):
        ## validing K value in K-means

        # print('...testing k: {}'.format(k))
        inertia_dic[k] = KMeans(n_clusters=k).fit(df_).inertia_
    return inertia_dic

def gap_statistic(df_, nrefs=3, maxClusters=15, k_initial=1, k_iteration=1):
    #### Gap
    #https://anaconda.org/milesgranger/gap-statistic/notebook
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        df: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = {}
    for k in tqdm(range(k_initial, maxClusters, k_iteration)):
        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=df_.shape)
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            refDisp = km.inertia_
            refDisps[i] = refDisp
        # Fit cluster to original data and create dispersion
        km = KMeans(k).fit(df_)
        origDisp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        # Assign this loop's gap statistic to gaps
        gaps[k] = gap

    return gaps  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal