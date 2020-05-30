import numpy as np
from sklearn.cluster import KMeans

from pymove.utils.log import progress_bar


def elbow_method(
        move_data, k_initial=1, max_clusters=15, k_iteration=1, random_state=None
):
    """
    Determines the optimal number of clusters in the range set by the user using
    the elbow method.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data.
    k_initial: int, optional (1 by default).
        The initial value used in the interaction of the elbow method.
        Represents the maximum numbers of clusters.
    max_clusters: int, optional (15  by default).
        The maximum value used in the interaction of the elbow method.
        Maximum number of clusters to test for
    k_iteration: int, optional (1 by default).
        Increment value of the sequence used by the elbow method.
    random_state: int, RandomState instance, default=None
        Determines random number generation for centroid initialization.
        Use an int to make the randomness deterministic

    Returns
    -------
    dict
        The inertia values ​​for the different numbers of clusters

    Example
    -------
    clustering.elbow_method(move_data=move_df[['lat', 'lon']], k_iteration=3)
        {
            1: 55084.15957839036,
            4: 245.68365592382938,
            7: 92.31472644640075,
            10: 62.618599956870355,
            13: 45.59653757292055,
        }

    """

    message = 'Executing Elbow Method to:\n...K of %srs to %srs from k_iteration:%srs\n'
    message = message % (k_initial, max_clusters, k_iteration)
    print(message, flush=True)
    inertia_dic = {}
    for k in progress_bar(range(k_initial, max_clusters + 1, k_iteration)):
        km = KMeans(n_clusters=k, random_state=random_state)
        inertia_dic[k] = km.fit(move_data).inertia_
    return inertia_dic


def gap_statistic(
    move_data, nrefs=3, k_initial=1, max_clusters=15, k_iteration=1, random_state=None
):
    """
    Calculates optimal clusters numbers using Gap Statistic from Tibshirani,
    Walther, Hastie.

    Parameters
    ----------
    move_data: ndarry of shape (n_samples, n_features).
        The input trajectory data.
    nrefs: int, optional (3 by default).
        number of sample reference datasets to create
    k_initial: int, optional (1 by default).
        The initial value used in the interaction of the elbow method.
        Represents the maximum numbers of clusters.
    max_clusters: int, optional (15  by default).
        Maximum number of clusters to test for.
    k_iteration:int, optional (1 by default).
        Increment value of the sequence used by the elbow method.
    random_state: int, RandomState instance, default=None
        Determines random number generation for centroid initialization.
        Use an int to make the randomness deterministic

    Returns
    -------
    dict
        The error value for each cluster number

    Notes
    -----
    https://anaconda.org/milesgranger/gap-statistic/notebook

    """

    message = 'Executing Gap Statistic to:\n...K of %srs to %srs from k_iteration:%srs\n'
    message = message % (k_initial, max_clusters, k_iteration)
    print(message, flush=True)
    gaps = {}
    np.random.seed(random_state)
    for k in progress_bar(range(k_initial, max_clusters + 1, k_iteration)):
        # Holder for reference dispersion results
        ref_disps = np.zeros(nrefs)
        # For n references, generate random sample and perform kmeans
        # getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            random_reference = np.random.random_sample(size=move_data.shape)
            # Fit to it
            km = KMeans(n_clusters=k, random_state=random_state)
            ref_disps[i] = km.fit(random_reference).inertia_
        # Fit cluster to original data and create dispersion
        km = KMeans(k).fit(move_data)
        orig_disp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(ref_disps)) - np.log(orig_disp)
        # Assign this loop'srs gap statistic to gaps
        gaps[k] = gap

    return gaps
