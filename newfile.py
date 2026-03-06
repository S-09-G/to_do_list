"""
Guided Clustering — Manager-defined clusters with cosine-similarity-based user assignment.

Instead of discovering clusters via unsupervised learning, this module lets managers
define K clusters by specifying features + weights, then assigns every user to the
nearest synthetic centroid using cosine similarity in the RAW feature space.

Lives in: src/models/guided_clustering.py
"""

import os
import json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm as sparse_norm
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from datetime import datetime


class GuidedClustering:
    """
    Assigns users to manager-defined clusters based on cosine similarity
    between user feature vectors and synthetic centroids.
    """

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.processed_dir = config["paths"]["processed_data"]
        self.reports_dir = config["paths"]["reports"]

    # ============================================================
    # FEATURE DISCOVERY — for the UI dropdowns
    # ============================================================

    def get_available_features(self) -> List[str]:
        """
        Return all feature names from the raw sparse matrix.
        The UI uses this to populate the feature selection dropdown.
        """
        from src.data_processing.config_data_loader import ConfigDataLoader

        raw_path = ConfigDataLoader.find_raw_matrix_path(self.processed_dir)
        if raw_path is None:
            raise FileNotFoundError(
                "Raw feature matrix not found. Build the matrix first via the Data page."
            )

        _, _, feature_names = ConfigDataLoader.load_sparse_raw(raw_path)
        self.logger.info(f"GuidedClustering: {len(feature_names)} features available")
        return feature_names

    # ============================================================
    # BUILD SYNTHETIC CENTROIDS
    # ============================================================

    def build_synthetic_centroids(
        self,
        cluster_definitions: List[Dict],
        feature_names: List[str],
    ) -> np.ndarray:
        """
        Convert manager-defined cluster specs into a (K, n_features) centroid matrix.

        Each cluster_definition looks like:
        {
            "name": "Comedy Lovers",
            "features": {"genre_Comedy": 0.8, "genre_Romance": 0.5, ...}
        }

        Features not mentioned get weight 0. The centroid is then the weighted
        indicator vector in raw feature space.

        Parameters
        ----------
        cluster_definitions : list of dict
            Each dict has 'name' (str) and 'features' (dict of feature->weight).
        feature_names : list of str
            Full ordered list of feature names matching the sparse matrix columns.

        Returns
        -------
        centroids : np.ndarray, shape (K, n_features)
        """
        feat_to_idx = {f: i for i, f in enumerate(feature_names)}
        n_features = len(feature_names)
        K = len(cluster_definitions)

        centroids = np.zeros((K, n_features), dtype=np.float32)

        for k, cluster_def in enumerate(cluster_definitions):
            unknown_features = []
            for feat_name, weight in cluster_def["features"].items():
                col = feat_to_idx.get(feat_name)
                if col is not None:
                    centroids[k, col] = float(weight)
                else:
                    unknown_features.append(feat_name)

            if unknown_features:
                self.logger.warning(
                    f"Cluster '{cluster_def['name']}': "
                    f"{len(unknown_features)} unknown features ignored: "
                    f"{unknown_features[:5]}"
                )

            # Warn if centroid is all zeros (no valid features matched)
            if np.all(centroids[k] == 0):
                self.logger.warning(
                    f"Cluster '{cluster_def['name']}' has zero centroid — "
                    f"no features matched the vocabulary."
                )

        self.logger.info(
            f"GuidedClustering: Built {K} synthetic centroids "
            f"({n_features}-dimensional)"
        )
        return centroids

    # ============================================================
    # ASSIGN USERS — the core algorithm
    # ============================================================

    def assign_users(
        self,
        centroids: np.ndarray,
        confidence_threshold: float = 0.0,
        batch_size: int = 500_000,
    ) -> Dict[str, Any]:
        """
        Assign every user to the nearest synthetic centroid via cosine similarity.

        Processes in batches to stay within RAM limits at 70M users.

        Parameters
        ----------
        centroids : np.ndarray, shape (K, n_features)
            Synthetic centroid matrix.
        confidence_threshold : float
            Users with max similarity below this are flagged as unassigned (-1).
        batch_size : int
            Number of users to process at a time.

        Returns
        -------
        dict with keys:
            - labels : np.ndarray of int, shape (n_users,)
            - similarities : np.ndarray of float, shape (n_users,)  (max sim per user)
            - user_ids : list of str
            - feature_names : list of str
        """
        from src.data_processing.config_data_loader import ConfigDataLoader

        raw_path = ConfigDataLoader.find_raw_matrix_path(self.processed_dir)
        if raw_path is None:
            raise FileNotFoundError("Raw feature matrix not found.")

        csr, user_ids, feature_names = ConfigDataLoader.load_sparse_raw(raw_path)
        n_users = csr.shape[0]
        K = centroids.shape[0]

        self.logger.info(
            f"GuidedClustering: Assigning {n_users:,} users to {K} clusters "
            f"(batch_size={batch_size:,})"
        )

        # Precompute centroid norms — (K,)
        centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=False)  # (K,)
        # Avoid division by zero for empty centroids
        centroid_norms = np.maximum(centroid_norms, 1e-10)

        labels = np.full(n_users, -1, dtype=np.int32)
        similarities = np.zeros(n_users, dtype=np.float32)

        n_batches = (n_users + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_users)
            batch_csr = csr[start:end]  # (batch, n_features) sparse

            # User norms — (batch, 1) for broadcasting
            user_norms = sparse_norm(batch_csr, axis=1)  # (batch,)
            user_norms = np.maximum(user_norms, 1e-10)

            # Dot products: sparse (batch, n_features) @ dense (n_features, K)
            # Result is dense (batch, K)
            dots = batch_csr.dot(centroids.T)  # (batch, K)

            # If dots came back sparse (unlikely with dense centroids), convert
            if sp.issparse(dots):
                dots = dots.toarray()

            # Cosine similarity
            # dots[i, k] / (user_norms[i] * centroid_norms[k])
            cos_sim = dots / (user_norms[:, np.newaxis] * centroid_norms[np.newaxis, :])

            # Assign to argmax
            batch_labels = np.argmax(cos_sim, axis=1)
            batch_max_sim = np.max(cos_sim, axis=1)

            # Apply confidence threshold
            below_threshold = batch_max_sim < confidence_threshold
            batch_labels[below_threshold] = -1

            labels[start:end] = batch_labels
            similarities[start:end] = batch_max_sim

            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                self.logger.info(
                    f"  Batch {batch_idx + 1}/{n_batches} done "
                    f"({end:,}/{n_users:,} users)"
                )

        n_assigned = int(np.sum(labels >= 0))
        n_unassigned = n_users - n_assigned
        self.logger.info(
            f"GuidedClustering: Assignment complete. "
            f"{n_assigned:,} assigned, {n_unassigned:,} unassigned "
            f"(threshold={confidence_threshold})"
        )

        return {
            "labels": labels,
            "similarities": similarities,
            "user_ids": user_ids,
            "feature_names": feature_names,
        }

    # ============================================================
    # METRICS
    # ============================================================

    def compute_metrics(
        self,
        labels: np.ndarray,
        similarities: np.ndarray,
        cluster_definitions: List[Dict],
    ) -> Dict[str, Any]:
        """
        Compute per-cluster and global stats for the guided assignment.
        """
        K = len(cluster_definitions)
        total_users = len(labels)

        cluster_stats = []
        for k in range(K):
            mask = labels == k
            count = int(mask.sum())
            if count > 0:
                sims = similarities[mask]
                cluster_stats.append({
                    "cluster_id": k,
                    "size": count,
                    "percentage": round(count / total_users * 100, 2),
                    "mean_similarity": round(float(np.mean(sims)), 4),
                    "min_similarity": round(float(np.min(sims)), 4),
                    "max_similarity": round(float(np.max(sims)), 4),
                    "median_similarity": round(float(np.median(sims)), 4),
                })
            else:
                cluster_stats.append({
                    "cluster_id": k,
                    "size": 0,
                    "percentage": 0.0,
                    "mean_similarity": 0.0,
                    "min_similarity": 0.0,
                    "max_similarity": 0.0,
                    "median_similarity": 0.0,
                })

        n_unassigned = int(np.sum(labels == -1))
        assigned_mask = labels >= 0
        global_mean_sim = float(np.mean(similarities[assigned_mask])) if assigned_mask.any() else 0.0

        return {
            "total_users": total_users,
            "total_assigned": total_users - n_unassigned,
            "total_unassigned": n_unassigned,
            "unassigned_percentage": round(n_unassigned / total_users * 100, 2),
            "global_mean_similarity": round(global_mean_sim, 4),
            "cluster_stats": cluster_stats,
        }

    # ============================================================
    # PROFILE GENERATION — top features per assigned cluster
    # ============================================================

    def generate_profiles(
        self,
        csr: sp.csr_matrix,
        labels: np.ndarray,
        feature_names: List[str],
        cluster_definitions: List[Dict],
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        For each cluster, compute mean feature values and find top actual features
        (not just the ones the manager defined, but what the assigned users actually have).

        This tells management "you asked for Comedy users, and here's what those
        users ACTUALLY look like" — which is extremely valuable for validation.
        """
        K = len(cluster_definitions)
        global_means = np.asarray(csr.mean(axis=0)).flatten()  # (n_features,)

        profiles = {}
        for k in range(K):
            mask = labels == k
            count = int(mask.sum())

            if count == 0:
                profiles[str(k)] = {
                    "cluster_id": k,
                    "defined_features": cluster_definitions[k]["features"],
                    "size": 0,
                    "percentage": 0.0,
                    "top_features": [],
                    "bottom_features": [],
                }
                continue

            cluster_csr = csr[mask]
            cluster_means = np.asarray(cluster_csr.mean(axis=0)).flatten()

            # Compute lift = cluster_mean / global_mean (avoid div by zero)
            safe_global = np.maximum(global_means, 1e-10)
            lifts = cluster_means / safe_global

            # Sort by lift for top features
            top_indices = np.argsort(lifts)[::-1][:top_n]
            bottom_indices = np.argsort(lifts)[:top_n]

            top_features = []
            for idx in top_indices:
                top_features.append({
                    "feature": feature_names[idx],
                    "cluster_mean": round(float(cluster_means[idx]), 4),
                    "global_mean": round(float(global_means[idx]), 4),
                    "lift": round(float(lifts[idx]), 2),
                })

            bottom_features = []
            for idx in bottom_indices:
                bottom_features.append({
                    "feature": feature_names[idx],
                    "cluster_mean": round(float(cluster_means[idx]), 4),
                    "global_mean": round(float(global_means[idx]), 4),
                    "lift": round(float(lifts[idx]), 2),
                })

            total = len(labels)
            profiles[str(k)] = {
                "cluster_id": k,
                "defined_features": cluster_definitions[k]["features"],
                "size": count,
                "percentage": round(count / total * 100, 2),
                "top_features": top_features,
                "bottom_features": bottom_features,
            }

        return profiles

    # ============================================================
    # SAVE / LOAD RESULTS
    # ============================================================

    def save_results(
        self,
        user_ids: List[str],
        labels: np.ndarray,
        similarities: np.ndarray,
        cluster_definitions: List[Dict],
        metrics: Dict[str, Any],
        profiles: Dict[str, Any],
        confidence_threshold: float = 0.0,
    ) -> Dict[str, str]:
        """
        Save a guided clustering run. Returns dict of file paths.
        """
        os.makedirs(self.reports_dir, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1) Assignments parquet
        assignments_path = os.path.join(
            self.reports_dir,
            f"guided_assignments_{run_id}.parquet",
        )
        df = pd.DataFrame({
            "user_id": user_ids,
            "cluster": labels,
            "similarity": similarities,
        })
        df.to_parquet(assignments_path, index=False)
        self.logger.info(f"Saved assignments: {assignments_path}")

        # 2) Run metadata JSON (definitions + metrics + profiles)
        meta_path = os.path.join(
            self.reports_dir,
            f"guided_run_{run_id}.json",
        )
        meta = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "n_clusters": len(cluster_definitions),
            "confidence_threshold": confidence_threshold,
            "cluster_definitions": cluster_definitions,
            "metrics": metrics,
            "profiles": profiles,
            "assignments_file": assignments_path,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        self.logger.info(f"Saved run metadata: {meta_path}")

        return {
            "run_id": run_id,
            "assignments_path": assignments_path,
            "meta_path": meta_path,
        }

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all saved guided clustering runs."""
        runs = []
        if not os.path.isdir(self.reports_dir):
            return runs

        for fname in sorted(os.listdir(self.reports_dir), reverse=True):
            if fname.startswith("guided_run_") and fname.endswith(".json"):
                path = os.path.join(self.reports_dir, fname)
                try:
                    with open(path, "r") as f:
                        meta = json.load(f)
                    runs.append({
                        "run_id": meta["run_id"],
                        "created_at": meta["created_at"],
                        "n_clusters": meta["n_clusters"],
                        "confidence_threshold": meta.get("confidence_threshold", 0.0),
                        "total_users": meta["metrics"]["total_users"],
                        "total_assigned": meta["metrics"]["total_assigned"],
                        "total_unassigned": meta["metrics"]["total_unassigned"],
                    })
                except Exception as e:
                    self.logger.warning(f"Could not read {fname}: {e}")
        return runs

    def load_run(self, run_id: str) -> Dict[str, Any]:
        """Load a saved guided clustering run by run_id."""
        meta_path = os.path.join(self.reports_dir, f"guided_run_{run_id}.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Guided run '{run_id}' not found.")

        with open(meta_path, "r") as f:
            return json.load(f)

    def delete_run(self, run_id: str) -> bool:
        """Delete a guided clustering run and its files."""
        meta_path = os.path.join(self.reports_dir, f"guided_run_{run_id}.json")
        assignments_path = os.path.join(
            self.reports_dir, f"guided_assignments_{run_id}.parquet"
        )

        deleted = False
        for path in [meta_path, assignments_path]:
            if os.path.exists(path):
                os.remove(path)
                self.logger.info(f"Deleted: {path}")
                deleted = True

        return deleted












    # ============================================================
    # GUIDED CLUSTERING
    # ============================================================
    #
    # Paste this section into pipeline_service.py right BEFORE the
    # singleton block (line ~1342, before "# Create a singleton instance").
    #
    # Also add this import at the top of the file alongside the other imports:
    #   from src.models.guided_clustering import GuidedClustering
    # ============================================================

    def get_guided_features(self) -> Dict[str, Any]:
        """Return all available feature names for the guided clustering UI."""
        self.logger.info("API: Fetching available features for guided clustering...")

        try:
            guided = GuidedClustering(self.config, self.logger)
            features = guided.get_available_features()

            return {
                "success": True,
                "features": features,
                "total": len(features),
                "message": f"Found {len(features):,} available features",
            }

        except Exception as e:
            self.logger.error(f"Guided feature fetch failed: {str(e)}")
            return {
                "success": False,
                "features": [],
                "total": 0,
                "message": f"Error: {str(e)}",
            }

    def run_guided_clustering(
        self,
        cluster_definitions: List[Dict],
        confidence_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Run guided clustering: build synthetic centroids from manager-defined
        features+weights, assign all users via cosine similarity.

        Parameters
        ----------
        cluster_definitions : list of dict
            Each dict has:
              - "features": dict of feature_name -> weight (0.0 to 1.0)
            Example:
            [
                {"features": {"genre_Comedy": 0.8, "genre_Romance": 0.5}},
                {"features": {"genre_Action": 0.9, "genre_Thriller": 0.7}},
            ]
        confidence_threshold : float
            Min cosine similarity to assign a user. Users below this
            get cluster = -1 (unassigned).

        Returns
        -------
        dict with success, metrics, profiles, run_id, etc.
        """
        self.logger.info(
            f"API: Running guided clustering with {len(cluster_definitions)} clusters, "
            f"threshold={confidence_threshold}..."
        )

        try:
            # Validate input
            if not cluster_definitions or len(cluster_definitions) == 0:
                return {
                    "success": False,
                    "message": "Must define at least 1 cluster.",
                }

            for i, cd in enumerate(cluster_definitions):
                if "features" not in cd or not cd["features"]:
                    return {
                        "success": False,
                        "message": f"Cluster {i} has no features defined.",
                    }

            guided = GuidedClustering(self.config, self.logger)

            # Step 1: Get feature names
            feature_names = guided.get_available_features()

            # Step 2: Build synthetic centroids
            centroids = guided.build_synthetic_centroids(
                cluster_definitions, feature_names
            )

            # Check for zero centroids
            zero_centroids = []
            for k in range(centroids.shape[0]):
                if np.all(centroids[k] == 0):
                    zero_centroids.append(k)

            if zero_centroids:
                return {
                    "success": False,
                    "message": (
                        f"Cluster(s) {zero_centroids} have no valid features. "
                        f"None of the specified feature names matched the vocabulary."
                    ),
                }

            # Step 3: Assign users
            result = guided.assign_users(
                centroids,
                confidence_threshold=confidence_threshold,
            )

            labels = result["labels"]
            similarities = result["similarities"]
            user_ids = result["user_ids"]
            feat_names = result["feature_names"]

            # Step 4: Compute metrics
            metrics = guided.compute_metrics(
                labels, similarities, cluster_definitions
            )

            # Step 5: Generate profiles (uses raw sparse matrix)
            from src.data_processing.config_data_loader import ConfigDataLoader

            raw_path = ConfigDataLoader.find_raw_matrix_path(
                self.config["paths"]["processed_data"]
            )
            csr, _, _ = ConfigDataLoader.load_sparse_raw(raw_path)

            profiles = guided.generate_profiles(
                csr, labels, feat_names, cluster_definitions
            )
            del csr
            gc.collect()

            # Step 6: Save everything
            save_result = guided.save_results(
                user_ids=user_ids,
                labels=labels,
                similarities=similarities,
                cluster_definitions=cluster_definitions,
                metrics=metrics,
                profiles=profiles,
                confidence_threshold=confidence_threshold,
            )

            return {
                "success": True,
                "run_id": save_result["run_id"],
                "n_clusters": len(cluster_definitions),
                "confidence_threshold": confidence_threshold,
                "metrics": metrics,
                "profiles": profiles,
                "assignments_path": save_result["assignments_path"],
                "message": (
                    f"Guided clustering complete. "
                    f"{metrics['total_assigned']:,} assigned, "
                    f"{metrics['total_unassigned']:,} unassigned "
                    f"across {len(cluster_definitions)} clusters."
                ),
            }

        except Exception as e:
            self.logger.error(f"Guided clustering failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"Error: {str(e)}",
            }

    def list_guided_runs(self) -> Dict[str, Any]:
        """List all saved guided clustering runs."""
        self.logger.info("API: Listing guided clustering runs...")

        try:
            guided = GuidedClustering(self.config, self.logger)
            runs = guided.list_runs()

            return {
                "success": True,
                "runs": runs,
                "total": len(runs),
                "message": f"Found {len(runs)} guided clustering run(s)",
            }

        except Exception as e:
            self.logger.error(f"List guided runs failed: {str(e)}")
            return {
                "success": False,
                "runs": [],
                "total": 0,
                "message": f"Error: {str(e)}",
            }

    def get_guided_run(self, run_id: str) -> Dict[str, Any]:
        """Load a specific guided clustering run."""
        self.logger.info(f"API: Loading guided run '{run_id}'...")

        try:
            guided = GuidedClustering(self.config, self.logger)
            run_data = guided.load_run(run_id)

            return {
                "success": True,
                **run_data,
                "message": f"Loaded guided run '{run_id}'",
            }

        except FileNotFoundError:
            return {
                "success": False,
                "message": f"Guided run '{run_id}' not found.",
            }
        except Exception as e:
            self.logger.error(f"Load guided run failed: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
            }

    def delete_guided_run(self, run_id: str) -> Dict[str, Any]:
        """Delete a guided clustering run."""
        self.logger.info(f"API: Deleting guided run '{run_id}'...")

        try:
            guided = GuidedClustering(self.config, self.logger)
            deleted = guided.delete_run(run_id)

            if deleted:
                return {
                    "success": True,
                    "run_id": run_id,
                    "message": f"Deleted guided run '{run_id}'",
                }
            else:
                return {
                    "success": False,
                    "message": f"Guided run '{run_id}' not found.",
                }

        except Exception as e:
            self.logger.error(f"Delete guided run failed: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
            }



















# ============================================================
# GUIDED CLUSTERING ROUTES
# ============================================================
#
# Paste this into your existing routes file (e.g., pipeline_routes.py).
#
# Add these Pydantic models at the top of the file alongside your
# other models, and add the route functions wherever your other
# route functions are.
# ============================================================


# ----- PYDANTIC MODELS (add alongside your other models) -----

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class ClusterDefinition(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Feature name -> weight mapping. e.g. {'genre_Comedy': 0.8, 'genre_Romance': 0.5}",
    )


class GuidedClusteringRequest(BaseModel):
    clusters: List[ClusterDefinition] = Field(
        ...,
        min_length=1,
        description="List of cluster definitions, each with features and weights",
    )
    confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Min cosine similarity to assign a user. Below this -> unassigned (-1).",
    )


# ----- ROUTE FUNCTIONS (add alongside your other routes) -----


@router.get("/guided/features")
async def get_guided_features():
    """
    Get all available feature names for the guided clustering UI.
    The frontend uses this to populate the feature selection dropdown/search.
    """
    service = get_pipeline_service()
    return service.get_guided_features()


@router.post("/guided/run")
async def run_guided_clustering(request: GuidedClusteringRequest):
    """
    Run guided clustering with manager-defined cluster definitions.

    Each cluster is defined by a set of features + weights.
    Users are assigned to the nearest cluster via cosine similarity.
    Users below the confidence_threshold are left unassigned (cluster = -1).

    Example request body:
    {
        "clusters": [
            {"features": {"genre_Comedy": 0.8, "genre_Romance": 0.5}},
            {"features": {"genre_Action": 0.9, "genre_Thriller": 0.7}},
            {"features": {"genre_Drama": 0.6, "genre_Korean": 0.8, "genre_Hindi": 0.4}}
        ],
        "confidence_threshold": 0.1
    }
    """
    service = get_pipeline_service()

    # Convert Pydantic models to plain dicts for the service layer
    cluster_definitions = [cd.model_dump() for cd in request.clusters]

    return service.run_guided_clustering(
        cluster_definitions=cluster_definitions,
        confidence_threshold=request.confidence_threshold,
    )


@router.get("/guided/runs")
async def list_guided_runs():
    """List all saved guided clustering runs."""
    service = get_pipeline_service()
    return service.list_guided_runs()


@router.get("/guided/runs/{run_id}")
async def get_guided_run(run_id: str):
    """Load a specific guided clustering run by its run_id."""
    service = get_pipeline_service()
    return service.get_guided_run(run_id)


@router.delete("/guided/runs/{run_id}")
async def delete_guided_run(run_id: str):
    """Delete a guided clustering run."""
    service = get_pipeline_service()
    return service.delete_guided_run(run_id)




















// ============================================================
// GUIDED CLUSTERING
// ============================================================
// Add these functions to your existing frontend/src/services/api.js

export const getGuidedFeatures = async () => {
  const response = await api.get('/guided/features');
  return response.data;
};

export const runGuidedClustering = async (clusters, confidenceThreshold = 0.0) => {
  const response = await api.post('/guided/run', {
    clusters: clusters,
    confidence_threshold: confidenceThreshold,
  });
  return response.data;
};

export const listGuidedRuns = async () => {
  const response = await api.get('/guided/runs');
  return response.data;
};

export const getGuidedRun = async (runId) => {
  const response = await api.get(`/guided/runs/${runId}`);
  return response.data;
};

export const deleteGuidedRun = async (runId) => {
  const response = await api.delete(`/guided/runs/${runId}`);
  return response.data;
};