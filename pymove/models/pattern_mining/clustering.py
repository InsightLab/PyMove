import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from pymove.utils.constants import EARTH_RADIUS, LATITUDE, LONGITUDE, N_CLUSTER
from pymove.utils.conversions import meters_to_eps
from pymove.utils.log import progress_bar, timer_decorator


@timer_decorator
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
    clustering.elbow_method(move_data=move_df, k_iteration=3)
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
        inertia_dic[k] = km.fit(move_data[[LATITUDE, LONGITUDE]]).inertia_
    return inertia_dic


@timer_decorator
def gap_statistic(
    move_data, nrefs=3, k_initial=1, max_clusters=15, k_iteration=1, random_state=None
):
    """
    Calculates optimal clusters numbers using Gap Statistic from Tibshirani,
    Walther, Hastie.

    Parameters
    ----------
    move_data: ndarray of shape (n_samples, n_features).
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
        km = KMeans(k).fit(move_data[[LATITUDE, LONGITUDE]])
        orig_disp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(ref_disps)) - np.log(orig_disp)
        # Assign this loop'srs gap statistic to gaps
        gaps[k] = gap

    return gaps


@timer_decorator
def dbscan_clustering(
    move_data,
    cluster_by,
    meters=10,
    min_sample=1680 / 2,
    earth_radius=EARTH_RADIUS,
    metric='euclidean',
    inplace=False
):
    """Performs density based clustering on the move_dataframe according to cluster_by

    Parameters
    ----------
    move_data : dataframe
        the input trajectory
    cluster_by : str
        the colum to cluster
    meters : int, optional
        distance to use in the clustering, by default 10
    min_sample : float, optional
        the minimum number of samples to consider a cluster, by default 1680/2
    earth_radius : int
        Y offset from your original position in meters.
    metric: string, or callable, default euclidean
        The metric to use when calculating distance between instances in a feature array
    """
    if not inplace:
        move_data = move_data[:]
    move_data.reset_index(drop=True, inplace=True)

    move_data[N_CLUSTER] = -1

    for cluster_id in progress_bar(move_data[cluster_by].unique(), desc='Clustering'):

        df_filter = move_data[move_data[cluster_by] == cluster_id]

        dbscan = DBSCAN(
            eps=meters_to_eps(meters, earth_radius),
            min_samples=min_sample,
            metric=metric
        )
        dbscan_result = dbscan.fit(df_filter[[LATITUDE, LONGITUDE]].to_numpy())

        idx = df_filter.index
        res = dbscan_result.labels_ + move_data[N_CLUSTER].max() + 1
        move_data.at[idx, N_CLUSTER] = res

    if not inplace:
        return move_data
