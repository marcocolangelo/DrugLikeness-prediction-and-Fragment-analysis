# first line: 280
def _hdbscan_prims_balltree(
    X,
    min_samples=5,
    alpha=1.0,
    metric="minkowski",
    p=2,
    leaf_size=40,
    gen_min_span_tree=False,
    **kwargs
):
    if X.dtype != np.float64:
        X = X.astype(np.float64)

    # The Cython routines used require contiguous arrays
    if not X.flags["C_CONTIGUOUS"]:
        X = np.array(X, dtype=np.double, order="C")

    tree = BallTree(X, metric=metric, leaf_size=leaf_size, **kwargs)

    dist_metric = DistanceMetric.get_metric(metric, **kwargs)

    # Get distance to kth nearest neighbour
    core_distances = tree.query(
        X, k=min_samples + 1, dualtree=True, breadth_first=True
    )[0][:, -1].copy(order="C")

    # Mutual reachability distance is implicit in mst_linkage_core_vector
    min_spanning_tree = mst_linkage_core_vector(X, core_distances, dist_metric, alpha)
    # Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]
    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    if gen_min_span_tree:
        return single_linkage_tree, min_spanning_tree
    else:
        return single_linkage_tree, None
