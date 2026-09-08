"""
One-time conversion of the GBRForestD regression files to .npz so the
regression can be evaluated with numpy without ROOT/CMSSW
(see Scouting/Tools/python/gbr_numpy.py for the evaluator).

Needs a CMSSW environment (PyROOT + GBRForestD dictionaries). Run:
    python3 Scouting/Tools/scripts/dumpForests.py
"""
import argparse
import numpy as np
import ROOT


def dump_forest(forest):
    """Flatten a GBRForestD into numpy arrays.

    Nodes of all trees are concatenated; node_offset/resp_offset give the
    start of each tree's nodes/responses in the flat arrays.
    """
    trees = forest.Trees()
    cut_indices, cut_vals, left, right, responses = [], [], [], [], []
    node_offset, resp_offset, n_nodes, n_resps = [], [], [], []
    for tree in trees:
        node_offset.append(len(cut_indices))
        resp_offset.append(len(responses))
        n_nodes.append(tree.CutIndices().size())
        n_resps.append(tree.Responses().size())
        cut_indices.extend(tree.CutIndices())
        cut_vals.extend(tree.CutVals())
        left.extend(tree.LeftIndices())
        right.extend(tree.RightIndices())
        responses.extend(tree.Responses())

    if min(n_nodes) < 1:
        raise RuntimeError("tree with no cut nodes found, evaluator cannot handle this")

    return {
        "initial_response": np.float64(forest.InitialResponse()),
        "cut_indices": np.asarray(cut_indices, dtype=np.int16),
        "cut_vals": np.asarray(cut_vals, dtype=np.float64),
        "left": np.asarray(left, dtype=np.int32),
        "right": np.asarray(right, dtype=np.int32),
        "responses": np.asarray(responses, dtype=np.float64),
        "node_offset": np.asarray(node_offset, dtype=np.int64),
        "resp_offset": np.asarray(resp_offset, dtype=np.int64),
    }


def dump_file(input_name, output_name, region):
    root_file = ROOT.TFile.Open(input_name)
    arrays = {}
    for kind, obj_name in (("mean", f"{region}Correction"), ("sigma", f"{region}Uncertainty")):
        forest = root_file[obj_name]
        for key, val in dump_forest(forest).items():
            arrays[f"{kind}_{key}"] = val
    np.savez_compressed(output_name, **arrays)
    print(f"wrote {output_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump GBRForestD regressions to npz")
    parser.add_argument("-d", "--datadir", default="Scouting/Tools/data", help="Directory with the regression root files")
    args = parser.parse_args()

    for base in ("regEleEcalScout2024_stdVar_stdCuts_{region}_ntrees1500_results",
                 "regEleEcalTrkTrainScout2024_stdVar_stdCuts_{region}_ntrees1500_results"):
        for region in ("EB", "EE"):
            name = base.format(region=region)
            dump_file(f"{args.datadir}/{name}.root", f"{args.datadir}/{name}.npz", region)
