"""
Validate the columnar regression (applyRegFast.py / egregression_np.py)
against the original PyROOT implementation (applyReg.py / egregression.py).

Checks, on the first N events of the input file:
  1. seed DetId decoding vs ROOT EBDetId/EEDetId
  2. the numpy GBRForest evaluator vs GBRForestD::GetResponse on
     identical feature rows
  3. end-to-end: ID selection and per-electron corrected energies vs the
     original per-event code (with the applyReg.py calo-energy indexing
     bug fixed on the ROOT side for the comparison)

Needs a CMSSW environment. Run from the src directory:
    python3 Scouting/Tools/scripts/validateReg.py -i <file> [-n 2000]
"""
import argparse

import numpy as np
import awkward as ak
import uproot
import ROOT

import Scouting.Tools.egregression as egreg_root
import Scouting.Tools.egregression_np as egreg_np
from applyReg import get_eles_passing_id
from applyRegFast import BRANCHES, get_flat_electrons, get_id_mask, apply_regressions


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}: {name} {detail}")
    return ok


def max_reldiff(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    scale = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.abs(a - b) / np.where(scale == 0, 1.0, scale), initial=0.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate columnar regression vs PyROOT")
    parser.add_argument("-i", "--inputfile", required=True)
    parser.add_argument("-n", "--nevents", type=int, default=2000)
    parser.add_argument("--datadir", default="Scouting/Tools/data")
    args = parser.parse_args()

    ecal_base = "regEleEcalScout2024_stdVar_stdCuts_{region}_ntrees1500_results"
    comb_base = "regEleEcalTrkTrainScout2024_stdVar_stdCuts_{region}_ntrees1500_results"

    # ---------- columnar side ----------
    events = uproot.open(f"{args.inputfile}:Events").arrays(BRANCHES, entry_stop=args.nevents)
    vals, counts = get_flat_electrons(events)
    ecal_reg_np = egreg_np.RegressionContainer(f"{args.datadir}/{ecal_base}.npz", 0.2, 2, 0.0002, 0.5)
    comb_reg_np = egreg_np.RegressionContainer(f"{args.datadir}/{comb_base}.npz", 0.2, 3, 0.0002, 0.5)
    corr = apply_regressions(vals, ecal_reg_np, comb_reg_np)
    id_mask = get_id_mask(vals)
    ele_event = np.repeat(np.arange(len(counts)), counts)
    ele_index = np.concatenate([np.arange(n) for n in counts]) if len(counts) else np.array([], dtype=int)

    all_ok = True

    # ---------- 1. DetId decode ----------
    i1_np, i2_np, is_eb_np = egreg_np.decode_seed_id(vals["seedId"])
    i1_root, i2_root, is_eb_root = [], [], []
    for seed_id in vals["seedId"]:
        i1, i2 = egreg_root.get_ietaiphi(int(seed_id))
        i1_root.append(i1)
        i2_root.append(i2)
        is_eb_root.append(egreg_root.is_eb(int(seed_id)))
    all_ok &= check(
        "DetId decode",
        np.array_equal(i1_np, i1_root) and np.array_equal(i2_np, i2_root) and np.array_equal(is_eb_np, is_eb_root),
        f"({len(i1_np)} seed ids)",
    )

    # ---------- 2. forest evaluator vs GetResponse ----------
    ecal_reg_root = egreg_root.RegressionContainer(f"{args.datadir}/{ecal_base}.root", 0.2, 2, 0.0002, 0.5)
    calo_features, is_eb = egreg_np.get_features_calo(vals)
    mean_np, sigma_np = ecal_reg_np.get_meansigma(calo_features, is_eb)
    mean_root, sigma_root = [], []
    for row, row_is_eb in zip(calo_features, is_eb):
        feat_vec = ROOT.std.vector("float")(len(row))
        for i, val in enumerate(row):
            feat_vec[i] = float(val)
        mean, sigma = ecal_reg_root.get_meansigma(feat_vec, bool(row_is_eb))
        mean_root.append(mean)
        sigma_root.append(sigma)
    diff_mean, diff_sigma = max_reldiff(mean_np, mean_root), max_reldiff(sigma_np, sigma_root)
    all_ok &= check(
        "forest evaluator",
        diff_mean < 1e-12 and diff_sigma < 1e-12,
        f"(n={len(is_eb)}, max reldiff mean={diff_mean:.2e} sigma={diff_sigma:.2e})",
    )

    # ---------- 3. end-to-end vs original per-event code ----------
    comb_reg_root = egreg_root.RegressionContainer(f"{args.datadir}/{comb_base}.root", 0.2, 3, 0.0002, 0.5)
    root_file = ROOT.TFile.Open(args.inputfile)
    event_tree = root_file.Events

    ref = {}  # (event, ele_index) -> (calo_corr_energy, comb_energy)
    for event_indx, event in enumerate(event_tree):
        if event_indx >= args.nevents:
            break
        passing = list(get_eles_passing_id(event_tree))
        calo_features_r = egreg_root.get_features_calo(event_tree, passing)
        calo_ms = [ecal_reg_root.get_meansigma(f["features"], f["isEB"]) for f in calo_features_r]
        comb_features_r = egreg_root.get_features_comb(event_tree, calo_ms, passing)
        comb_ms = [comb_reg_root.get_meansigma(f["features"], f["isEB"]) for f in comb_features_r]
        raw_comb = egreg_root.get_raw_comb(event_tree, calo_ms, passing)
        for pos, tree_idx in enumerate(passing):
            calo_e = egreg_root.pt_to_p(
                event_tree.ScoutingElectron_pt[tree_idx], event_tree.ScoutingElectron_eta[tree_idx]
            )
            ref[(event_indx, tree_idx)] = (calo_e * calo_ms[pos][0], raw_comb[pos] * comb_ms[pos][0])

    new_sel = {
        (int(evt), int(idx)): (corr["corrEcalEnergy"][i], corr["corrEnergy"][i])
        for i, (evt, idx, passing) in enumerate(zip(ele_event, ele_index, id_mask))
        if passing
    }
    all_ok &= check(
        "ID selection", set(new_sel) == set(ref), f"(ROOT {len(ref)} vs columnar {len(new_sel)} electrons)"
    )
    if set(new_sel) == set(ref):
        keys = sorted(ref)
        diff_calo = max_reldiff([new_sel[k][0] for k in keys], [ref[k][0] for k in keys])
        diff_comb = max_reldiff([new_sel[k][1] for k in keys], [ref[k][1] for k in keys])
        all_ok &= check(
            "corrected energies",
            diff_calo < 1e-9 and diff_comb < 1e-9,
            f"(n={len(keys)}, max reldiff calo={diff_calo:.2e} comb={diff_comb:.2e})",
        )

    print("\nALL PASSED" if all_ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if all_ok else 1)
