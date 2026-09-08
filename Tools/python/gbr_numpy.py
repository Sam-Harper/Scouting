"""
Numpy evaluator for CMSSW GBRForestD regressions dumped to .npz by
Scouting/Tools/scripts/dumpForests.py.

Reproduces GBRForestD::GetResponse exactly (CondFormats/GBRForest):
features and cut values are float32 and the descent uses a strict >
comparison; a node index <= 0 is a terminal node whose response is
fResponses[-index]. The root node (index 0) is always evaluated first.
"""
import numpy as np


class GBRForestNP:
    def __init__(self, npz, prefix):
        def get(name):
            return npz[f"{prefix}_{name}"]

        self.initial_response = float(get("initial_response"))

        cut_indices = get("cut_indices").astype(np.intp)
        # cut values are stored as float in GBRTreeD so float32 is lossless
        cut_vals = get("cut_vals").astype(np.float32)
        left = get("left").astype(np.int32)
        right = get("right").astype(np.int32)
        responses = get("responses").astype(np.float64)

        node_bounds = np.append(get("node_offset"), len(cut_indices))
        resp_bounds = np.append(get("resp_offset"), len(responses))
        self.trees = [
            (
                cut_indices[node_bounds[i] : node_bounds[i + 1]],
                cut_vals[node_bounds[i] : node_bounds[i + 1]],
                left[node_bounds[i] : node_bounds[i + 1]],
                right[node_bounds[i] : node_bounds[i + 1]],
                responses[resp_bounds[i] : resp_bounds[i + 1]],
            )
            for i in range(len(node_bounds) - 1)
        ]

    def get_response(self, features):
        """Evaluate the forest for a (n_candidates, n_features) matrix.

        Returns a (n_candidates,) float64 array of raw responses.
        """
        feats = np.ascontiguousarray(features, dtype=np.float32)
        n = feats.shape[0]
        response = np.full(n, self.initial_response, dtype=np.float64)
        if n == 0:
            return response

        for cut_indices, cut_vals, left, right, responses in self.trees:
            node = np.zeros(n, dtype=np.int32)
            alive = np.arange(n, dtype=np.intp)
            while alive.size:
                cur = node[alive]
                go_right = feats[alive, cut_indices[cur]] > cut_vals[cur]
                nxt = np.where(go_right, right[cur], left[cur])
                node[alive] = nxt
                alive = alive[nxt > 0]
            response += responses[-node]
        return response


def load_regression(base_file):
    """Load {region}->(mean, sigma) GBRForestNP pairs from npz files.

    base_file is a template with a {region} placeholder, matching the
    root file naming but with .npz extension.
    """
    forests = {}
    for region in ("EB", "EE"):
        npz = np.load(base_file.format(region=region))
        forests[region] = (GBRForestNP(npz, "mean"), GBRForestNP(npz, "sigma"))
    return forests
