#!/usr/bin/env python3

from collections import defaultdict, deque

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from loguru import logger
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

# Tolerance for exact coordinate match (0.00001 arcsec ≈ 0.05 milliarcsec)
# Well below typical astrometric precision (~10 mas for Gaia)
EXACT_DUPLICATE_TOLERANCE = 1e-5  # arcsec


def _cluster_with_agglomerative(
    coords: SkyCoord,
    is_medium_resolution: np.ndarray,
    max_separation: float,
    max_cluster_diameter: float,
    max_points_for_agglomerative: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster with strict diameter constraint using hybrid approach.

    Optimized algorithm:
    1. Pre-filter: Find points with neighbors using search_around_sky (fast)
    2. Split: Divide into connected components using BFS
    3. Cluster: Apply AgglomerativeClustering only to small subsets

    Uses complete linkage which ensures cluster diameter <= threshold.

    Parameters:
    -----------
    coords : SkyCoord
        Coordinate array
    is_medium_resolution : array-like
        Boolean array indicating medium resolution targets
    max_separation : float
        Maximum separation in degrees for nearest neighbor calculation
    max_cluster_diameter : float
        Maximum cluster diameter in degrees

    Returns:
    --------
    labels : array
        Cluster labels (-1 for isolated objects)
    nn_separations : array
        Nearest neighbor separation in arcsec
    """
    n_points = len(coords)
    labels = np.full(n_points, -1, dtype=int)
    nn_separations = np.full(n_points, np.nan)

    # Step 1: Find candidate neighbors within max_cluster_diameter
    logger.debug(
        f"Searching for neighbors within {max_cluster_diameter:.6f} deg for {n_points} points..."
    )
    idx1, idx2, seps_cluster, _ = search_around_sky(
        coords, coords, max_cluster_diameter * u.deg
    )

    # Exclude self-matches and different resolutions
    not_self = idx1 != idx2
    same_resolution = is_medium_resolution[idx1] == is_medium_resolution[idx2]
    valid = not_self & same_resolution
    idx1_cluster, idx2_cluster, seps_cluster = (
        idx1[valid],
        idx2[valid],
        seps_cluster[valid],
    )

    # Step 2: Identify points that have neighbors
    has_neighbors = np.unique(np.concatenate([idx1_cluster, idx2_cluster]))
    n_with_neighbors = len(has_neighbors)

    logger.debug(
        f"Found {n_with_neighbors} points with neighbors ({n_with_neighbors/n_points*100:.1f}%)"
    )

    if n_with_neighbors == 0:
        # All points are isolated
        logger.debug("No duplicates found - all points are isolated")
        return labels, nn_separations

    # Step 3: Find connected components using Breadth-First Search (BFS)
    logger.debug("Finding connected components using BFS...")
    adjacency = defaultdict(set)
    for i, j in zip(idx1_cluster, idx2_cluster, strict=True):
        adjacency[i].add(j)
        adjacency[j].add(i)

    visited = set()
    components = []

    for start_node in has_neighbors:
        if start_node in visited:
            continue

        # BFS
        component = set()
        queue = deque([start_node])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            queue.extend(adjacency[node] - visited)

        components.append(list(component))

    logger.info(f"Found {len(components)} connected components")

    # Step 4: Apply AgglomerativeClustering to each component
    current_label = 0

    for comp_idx, component in enumerate(components):
        comp_size = len(component)

        if comp_size == 1:
            # Single point (safety check - shouldn't happen normally)
            labels[component[0]] = -1
            continue

        # Check memory limit for this component
        if max_points_for_agglomerative is not None:
            if comp_size > max_points_for_agglomerative:
                estimated_memory_gb = (comp_size**2 * 8) / (1024**3)
                raise ValueError(
                    f"Component {comp_idx+1} has {comp_size} points, exceeding limit "
                    f"({max_points_for_agglomerative}). "
                    f"Estimated memory: ~{estimated_memory_gb:.1f} GB. "
                    f"Increase --max-points-for-agglomerative or adjust max_cluster_diameter."
                )

        logger.debug(
            f"Processing component {comp_idx+1}/{len(components)} with {comp_size} points"
        )

        # Extract coordinates for points in this component
        comp_coords = coords[component]
        comp_is_mr = is_medium_resolution[component]

        # Compute distance matrix (small since it's only within the component)
        ra_rad = comp_coords.ra.radian
        dec_rad = comp_coords.dec.radian
        coords_rad = np.column_stack([dec_rad, ra_rad])
        distances_rad = pairwise_distances(coords_rad, metric="haversine")

        # Set distance to infinity for points with different medium resolution
        # Vectorized: create mask for all pairs with different resolution
        diff_resolution_mask = comp_is_mr[:, None] != comp_is_mr[None, :]
        distances_rad[diff_resolution_mask] = np.inf

        # Cluster with complete linkage
        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=np.radians(max_cluster_diameter),
            linkage="complete",
            metric="precomputed",
        )
        comp_labels = agg.fit_predict(distances_rad)

        # Map component labels to global labels
        # Single-point clusters are set to -1
        unique_comp_labels, counts = np.unique(comp_labels, return_counts=True)
        label_mapping = {}

        for label, count in zip(unique_comp_labels, counts, strict=True):
            if count == 1:
                label_mapping[label] = -1
            else:
                label_mapping[label] = current_label
                current_label += 1

        # Assign global labels
        for i, global_idx in enumerate(component):
            labels[global_idx] = label_mapping[comp_labels[i]]

    # Step 5: Compute nearest neighbor separations (search within max_separation)
    logger.debug("Computing nearest neighbor separations...")
    idx1, idx2, seps, _ = search_around_sky(coords, coords, max_separation * u.deg)

    # Exclude self-matches and different resolutions
    not_self = idx1 != idx2
    same_resolution = is_medium_resolution[idx1] == is_medium_resolution[idx2]
    valid = not_self & same_resolution
    idx1, idx2, seps = idx1[valid], idx2[valid], seps[valid]

    # Compute nearest neighbor separation - optimized vectorized approach
    # Only compute for points that are in clusters (labels >= 0)
    clustered_mask = labels >= 0
    clustered_indices = np.where(clustered_mask)[0]

    if len(clustered_indices) > 0:
        seps_arcsec = seps.arcsec

        # Create temporary dataframe for efficient groupby operation
        df_pairs = pd.DataFrame({
            'idx1': idx1,
            'idx2': idx2,
            'sep': seps_arcsec
        })

        # For each point, find its minimum separation (considering both idx1 and idx2)
        # Stack idx1 and idx2 to treat them symmetrically
        df_stacked = pd.concat([
            df_pairs[['idx1', 'sep']].rename(columns={'idx1': 'idx'}),
            df_pairs[['idx2', 'sep']].rename(columns={'idx2': 'idx'})
        ])

        # Filter to only clustered points and compute minimum
        df_stacked = df_stacked[df_stacked['idx'].isin(clustered_indices)]
        min_seps = df_stacked.groupby('idx')['sep'].min()

        # Assign minimum separations back to the array
        nn_separations[min_seps.index] = min_seps.values

    # Statistics
    n_clusters = len(np.unique(labels[labels >= 0]))
    n_isolated = np.sum(labels == -1)
    n_in_groups = np.sum(labels >= 0)

    logger.debug(f"number of clusters: {n_clusters}")
    logger.debug(f"number of isolated objects: {n_isolated}")
    logger.debug(f"number of objects in pairs/groups: {n_in_groups}")

    return labels, nn_separations


def _find_duplicates_with_separation(
    ra: np.ndarray,
    dec: np.ndarray,
    is_medium_resolution: np.ndarray,
    max_separation: float = 0.1,
    max_cluster_diameter: float | None = None,
    max_points_for_agglomerative: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find duplicates with strict cluster diameter constraint.

    Uses AgglomerativeClustering with complete linkage for strict diameter control.

    Parameters:
    -----------
    ra : array-like
        Right ascension in degrees
    dec : array-like
        Declination in degrees
    is_medium_resolution : array-like
        Boolean array indicating medium resolution targets
    max_separation : float
        Maximum separation in degrees for linking objects
    max_cluster_diameter : float or None
        Maximum cluster diameter in degrees (converted from astropy.units.Quantity).
        If None: defaults to max_separation * 2
        Uses AgglomerativeClustering with complete linkage.
    max_points_for_agglomerative : int or None
        Maximum points allowed for AgglomerativeClustering.
        If None: no limit (may use large memory for big datasets)
        If specified: raises error when n_points > limit
        Memory estimate: 10000 pts ≈ 800MB, 50000 pts ≈ 20GB

    Returns:
    --------
    labels : array
        Cluster labels (-1 for unique objects)
    nn_separations : array
        Nearest neighbor separation in arcsec (nan for unique objects)
    """
    # Input validation
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    is_medium_resolution = np.asarray(is_medium_resolution)

    n_points = len(ra)

    if len(dec) != n_points or len(is_medium_resolution) != n_points:
        raise ValueError(
            f"Array length mismatch: ra={n_points}, dec={len(dec)}, "
            f"is_medium_resolution={len(is_medium_resolution)}"
        )

    # Initialize result arrays for all points
    labels = np.full(n_points, -1, dtype=int)
    nn_separations = np.full(n_points, np.nan)

    # Check for invalid values
    valid = np.isfinite(ra) & np.isfinite(dec)

    if not np.all(valid):
        n_invalid = np.sum(~valid)
        logger.warning(f"Found {n_invalid} non-finite coordinate values (NaN/inf)")

        if not np.any(valid):
            raise ValueError("No valid coordinates found")

        logger.warning(f"Processing {np.sum(valid)} valid coordinates, marking {n_invalid} as isolated")

        # Process only valid coordinates
        valid_ra = ra[valid]
        valid_dec = dec[valid]
        valid_is_mr = is_medium_resolution[valid]
    else:
        # All coordinates are valid
        valid_ra = ra
        valid_dec = dec
        valid_is_mr = is_medium_resolution

    # Validate coordinate ranges (only for valid coordinates)
    if np.any((valid_ra < 0) | (valid_ra > 360)):
        raise ValueError(
            f"RA must be in range [0, 360], found values in [{valid_ra.min()}, {valid_ra.max()}]"
        )
    if np.any((valid_dec < -90) | (valid_dec > 90)):
        raise ValueError(
            f"Dec must be in range [-90, 90], found values in [{valid_dec.min()}, {valid_dec.max()}]"
        )

    # Create SkyCoord objects for valid coordinates
    coords = SkyCoord(ra=valid_ra * u.deg, dec=valid_dec * u.deg, frame="icrs")

    # Set default max_cluster_diameter
    if max_cluster_diameter is None:
        # Default: 2 * max_separation
        # For PFS: fiber diameter = 1.0 arcsec
        # max_separation = fiber diameter for nearest neighbor detection
        # max_cluster_diameter = 2 × fiber diameter for detecting spatial grouping
        max_cluster_diameter = max_separation * 2
        logger.debug(
            f"Using default max_cluster_diameter = {max_cluster_diameter:.6f} deg (2 * max_separation)"
        )

    # Use AgglomerativeClustering for strict diameter constraint
    valid_labels, valid_nn_separations = _cluster_with_agglomerative(
        coords,
        valid_is_mr,
        max_separation,
        max_cluster_diameter,
        max_points_for_agglomerative,
    )

    # Map results back to original indices
    labels[valid] = valid_labels
    nn_separations[valid] = valid_nn_separations
    # Invalid coordinates remain as -1 and nan (already initialized)

    return labels, nn_separations


def dupcheck_internal(
    df: pd.DataFrame,
    sep: u.Quantity = 1.0 * u.arcsec,
    max_cluster_diameter: u.Quantity | None = None,
    max_points_for_agglomerative: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Check for internal duplicates within a single proposal.

    Identifies duplicate targets within the same proposal based on coordinate
    proximity and medium resolution mode. Optionally outputs separate CSV files
    for isolated objects, exact duplicates, and near duplicates.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing target data for a single proposal
    sep : astropy.units.Quantity
        Maximum angular separation for nearest neighbor search (default: 1.0 arcsec = PFS fiber diameter)
    max_cluster_diameter : astropy.units.Quantity or None
        Maximum cluster diameter (angular units) for grouping targets.
        If None: defaults to 2 * sep (default: 2.0 arcsec = 2 × PFS fiber diameter)
    max_points_for_agglomerative : int or None
        Maximum points allowed for AgglomerativeClustering.
        If None: no limit

    Returns:
    --------
    tuple of (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
        Returns (df_isolated, df_dups_exact, df_dups_near):
        - df_isolated: DataFrame of isolated (non-duplicate) targets
        - df_dups_exact: DataFrame of exact coordinate duplicates
        - df_dups_near: DataFrame of near duplicates within separation threshold
        All DataFrames will be empty if input df is empty.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    is_medium_resolution = df["resolution"] == "M"

    labels, nn_separations = _find_duplicates_with_separation(
        df["ra"].to_numpy(),
        df["dec"].to_numpy(),
        is_medium_resolution.to_numpy(),
        max_separation=sep.to(u.deg).value,
        max_cluster_diameter=(
            max_cluster_diameter.to(u.deg).value
            if max_cluster_diameter is not None
            else None
        ),
        max_points_for_agglomerative=max_points_for_agglomerative,
    )

    logger.debug(f"{len(labels)=}")
    logger.debug(f"{len(nn_separations)=}")

    # Group df by labels for duplicated objects
    # Create a copy to avoid modifying the original DataFrame
    df_result = df.copy()
    df_result["dup_label"] = labels

    df_dups = df_result.loc[df_result["dup_label"] >= 0, :].copy()
    # Add duplication count to df_dups
    df_dups["dup_count"] = df_dups.groupby("dup_label")["dup_label"].transform("count")
    df_dups["nn_sep"] = nn_separations[(df_result["dup_label"] >= 0).to_numpy()]
    # Sort df_dups by dup_count and dup_label
    df_dups.sort_values(by=["dup_count", "dup_label"], inplace=True)

    max_dups = df_dups["dup_count"].max()
    max_dups = max_dups if np.isfinite(max_dups) else 0

    logger.info(f"Maximum duplication count: {max_dups}")

    # Separate exact duplicates (within tolerance) from near duplicates
    df_dups_exact = df_dups.loc[df_dups["nn_sep"] < EXACT_DUPLICATE_TOLERANCE, :].copy()
    # Include rows with nn_sep >= tolerance OR nn_sep == NaN (clustered by max_cluster_diameter but no neighbor within max_separation)
    df_dups_near = df_dups.loc[
        (df_dups["nn_sep"] >= EXACT_DUPLICATE_TOLERANCE) | (pd.isna(df_dups["nn_sep"])),
        :,
    ].copy()

    df_isolated = df_result.loc[df_result["dup_label"] == -1, :].copy()
    df_isolated["dup_count"] = 1
    df_isolated["dup_label"] = -1
    df_isolated["nn_sep"] = np.nan

    logger.debug(f"Exact duplicates:\n{df_dups_exact.head(5)}")
    logger.debug(f"Near duplicates:\n{df_dups_near.head(5)}")
    logger.debug(f"Isolated objects:\n{df_isolated.head(5)}")

    return df_isolated, df_dups_exact, df_dups_near
