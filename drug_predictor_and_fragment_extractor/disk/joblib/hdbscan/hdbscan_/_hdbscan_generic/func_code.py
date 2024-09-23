# first line: 94
def _hdbscan_generic(
    X,
    min_samples=5,
    alpha=1.0,
    metric="minkowski",
    p=2,
    leaf_size=None,
    gen_min_span_tree=False,
    **kwargs
):
    if metric == "minkowski":
        distance_matrix = pairwise_distances(X, metric=metric, p=p)
    elif metric == "arccos":
        distance_matrix = pairwise_distances(X, metric="cosine", **kwargs)
    elif metric == "precomputed":
        # Treating this case explicitly, instead of letting
        #   sklearn.metrics.pairwise_distances handle it,
        #   enables the usage of numpy.inf in the distance
        #   matrix to indicate missing distance information.
        # TODO: Check if copying is necessary
        distance_matrix = X.copy()
    else:
        distance_matrix = pairwise_distances(X, metric=metric, **kwargs)

    if issparse(distance_matrix):
        # raise TypeError('Sparse distance matrices not yet supported')
        return _hdbscan_sparse_distance_matrix(
            distance_matrix,
            min_samples,
            alpha,
            metric,
            p,
            leaf_size,
            gen_min_span_tree,
            **kwargs
        )

    mutual_reachability_ = mutual_reachability(distance_matrix, min_samples, alpha)

    min_spanning_tree = mst_linkage_core(mutual_reachability_)

    # Warn if the MST couldn't be constructed around the missing distances
    if np.isinf(min_spanning_tree.T[2]).any():
        warn(
            "The minimum spanning tree contains edge weights with value "
            "infinity. Potentially, you are missing too many distances "
            "in the initial distance matrix for the given neighborhood "
            "size.",
            UserWarning,
        )

    # mst_linkage_core does not generate a full minimal spanning tree
    # If a tree is required then we must build the edges from the information
    # returned by mst_linkage_core (i.e. just the order of points to be merged)
    if gen_min_span_tree:
        result_min_span_tree = min_spanning_tree.copy()
        for index, row in enumerate(result_min_span_tree[1:], 1):
            candidates = np.where(isclose(mutual_reachability_[int(row[1])], row[2]))[0]
            candidates = np.intersect1d(
                candidates, min_spanning_tree[:index, :2].astype(int)
            )
            candidates = candidates[candidates != row[1]]
            assert len(candidates) > 0
            row[0] = candidates[0]
    else:
        result_min_span_tree = None

    # Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]

    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    return single_linkage_tree, result_min_span_tree
