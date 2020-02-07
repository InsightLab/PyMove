from tqdm import tqdm_notebook as tqdm
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def elbow_method(move_data, k_initial=1, max_clusters=15, k_iteration=1):
    """Determines the optimal number of clusters in the range set by the user using the elbow method.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data.
    k_initial: int, optional (1 by default).
        The initial value used in the interaction of the elbow method. Represents the maximum numbers of clusters.
    max_clusters: int, optional (15  by default).
        The maximum value used in the interaction of the elbow method. Maximum number of clusters to test for
    k_iteration: int, optional (1 by default).
        Increment value of the sequence used by the elbow method.

    Returns
    -------
    inertia_dic : dictionary
        The inertia values ​​for the different numbers of clusters
    Example
    -------
        clustering.elbow_method(move_data=move_df[['lat', 'lon']], k_initial = 2, max_clusters = 17, k_iteration = 2)
            {2: 55084.15957839036,
             4: 245.68365592382938,
             6: 92.31472644640075,
             8: 62.618599956870355,
             10: 45.59653757292055,
             12: 34.32238676029195,
             14: 26.087387367439227,
             16: 20.64369311973992}
    """

    print('Executing Elbow Method to:\n...K of {} to {} from k_iteration:{}\n'.format(k_initial,max_clusters, k_iteration))
    inertia_dic = {}
    for k in tqdm(range(k_initial, max_clusters, k_iteration)):
        # validing K value in K-means
        # print('...testing k: {}'.format(k))
        inertia_dic[k] = KMeans(n_clusters=k).fit(move_data).inertia_
    return inertia_dic

def gap_statistic(move_data, nrefs=3, k_initial=1, max_clusters=15, k_iteration=1):
    """Calculates optimal clusters numbers using Gap Statistic from Tibshirani, Walther, Hastie

    Parameters
    ----------
    move_data: ndarry of shape (n_samples, n_features).
        The input trajectory data.
    nrefs: int, optional (3 by default).
        number of sample reference datasets to create
    k_initial: int, optional (1 by default).
        The initial value used in the interaction of the elbow method. Represents the maximum numbers of clusters.
    max_clusters: int, optional (15  by default).
        Maximum number of clusters to test for.
    k_iteration:int, optional (1 by default).
        Increment value of the sequence used by the elbow method.

    Returns
    -------
    gaps :  dictionary
        The error value for each cluster number

    Notes
    -----
    https://anaconda.org/milesgranger/gap-statistic/notebook

    """
    gaps = {}
    for k in tqdm(range(k_initial, max_clusters, k_iteration)):
        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=move_data.shape)
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            refDisp = km.inertia_
            refDisps[i] = refDisp
        # Fit cluster to original data and create dispersion
        km = KMeans(k).fit(move_data)
        origDisp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        # Assign this loop's gap statistic to gaps
        gaps[k] = gap

    return gaps  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal