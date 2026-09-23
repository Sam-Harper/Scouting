"""
Columnar (uproot + awkward + numpy) version of applyReg.py.

Applies the ECAL-only and E-p combination energy regressions to scouting
nanoAOD electrons and writes:
  - an Events tree with the corrected energies as jagged per-electron
    branches (all electrons; electrons without a valid best track fall
    back to the ECAL-only corrected energy for the combined branches)
  - the raw / combination-corrected / ECAL-corrected dielectron mass
    histograms of applyReg.py, filled with electrons passing the ID

Needs the regression npz files produced by
Scouting/Tools/scripts/dumpForests.py, but no ROOT/CMSSW.

Differences to applyReg.py:
  - the calo-corrected energy indexing bug (applyReg.py used the position
    in the ID-passing list instead of the tree index) is fixed, so
    caloCorrMassHist differs where an event has ID-failing electrons
  - histograms lose their under/overflow content
"""
import argparse
import concurrent.futures
import time

import numpy as np
import awkward as ak
import uproot
import vector

vector.register_awkward()

try:
    import Scouting.Tools.egregression_np as egreg
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
    import egregression_np as egreg

ELE_BRANCHES = [
    "pt", "eta", "phi", "sigmaIetaIeta", "r9", "sMin", "sMaj","hOverE",
    "rechitZeroSuppression", "seedId", "trackIso", "dEtaIn", "dPhiIn","trackfbrem",
    "bestTrack_etaMode", "bestTrack_phiMode", "bestTrack_pMode","bestTrack_pt",
    "bestTrack_qoverpModeError", "bestTrack_charge","transparencyCorr"
]
BRANCHES = (
    ["run", "luminosityBlock", "event", "ScoutingRho_fixedGridRhoFastjetAll"]
    + [f"ScoutingElectron_{name}" for name in ELE_BRANCHES]
)

MASS_BINS = np.linspace(0.0, 120.0, 12001)


def get_flat_electrons(events):
    """Flatten the per-electron branches into a dict of numpy arrays."""
    counts = ak.to_numpy(ak.num(events["ScoutingElectron_pt"]))
    vals = {
        name: ak.to_numpy(ak.flatten(events[f"ScoutingElectron_{name}"]))
        for name in ELE_BRANCHES
    }
    vals["rho"] = np.repeat(ak.to_numpy(events["ScoutingRho_fixedGridRhoFastjetAll"]), counts)
    return vals, counts


def get_id_mask(vals):
    """Electron ID of applyReg.py (barrel only unless include_ee)."""
    valid_trk = vals["bestTrack_etaMode"] <= 1000
    trk_pt = vals["bestTrack_pMode"] / np.cosh(np.where(valid_trk, vals["bestTrack_etaMode"], 0.0))
    is_eb = np.abs(vals["eta"]) < 1.479
    common = valid_trk & (vals["trackIso"] <= 0)  & (trk_pt > 3)  
    passing = common & is_eb & (vals["sigmaIetaIeta"] < 0.0105) & (np.abs(vals["dPhiIn"]) <= 0.03) & (np.abs(vals["dEtaIn"]) <= 0.003)
    
    passing |= common & ~is_eb & (vals["sigmaIetaIeta"] < 0.034) & (np.abs(vals["dPhiIn"]) <= 0.03) & (np.abs(vals["dEtaIn"]) <= 0.005)
    return passing


def get_jpsi_id_mask(vals, corr):
    """
    gets the ID mask for low pt boosted electrons
    specifically cant have isolation on it

    Returns a boolean mask indicating which electrons pass the J/psi ID criteria.

    """
    valid_trk = vals["bestTrack_etaMode"] <= 1000
    trk_pt = vals["bestTrack_pMode"] / np.cosh(np.where(valid_trk, vals["bestTrack_etaMode"], 0.0))
    is_eb = np.abs(vals["eta"]) < 1.479
    rel_err = corr["corrEnergyErr"] / np.maximum(corr["corrEnergy"], 1e-9)
    rel_err = 0
    pass_mask = (
        valid_trk & is_eb & (vals["sigmaIetaIeta"] < 0.0105)
        & (np.abs(vals["dEtaIn"]) <= 0.03) & (trk_pt > 3) & (rel_err < 0.04)
        & (np.abs(vals["dPhiIn"]) <= 0.1)
    )
    
    pass_mask |= (
        valid_trk & ~is_eb & (vals["sigmaIetaIeta"] < 0.031)
        & (np.abs(vals["dEtaIn"]) <= 0.03) & (trk_pt > 5)
        #& (np.abs(vals["dPhiIn"]) <= 0.06)
        & (vals["hOverE"] < 0.1)
    )
    return pass_mask


def apply_regressions(vals, ecal_reg, comb_reg, ecal_scale=1.0, tag=""):
    """Run both regression stages; returns a dict of flat result arrays.

    Electrons without a valid best track get the ECAL-only corrected
    energy (and its error) copied into the combined columns.
    """
    calo_features, is_eb = egreg.get_features_calo(vals, ecal_scale)
    ecal_mean, ecal_sigma = ecal_reg.get_meansigma(calo_features, is_eb)

    ecal_energy = egreg.pt_to_p(vals["pt"], vals["eta"]) * ecal_scale
    corr_ecal_energy = ecal_energy * ecal_mean
    corr_ecal_energy_err = ecal_energy * ecal_sigma

    valid_trk = vals["bestTrack_etaMode"] <= 1000
    corr_energy = corr_ecal_energy.copy()
    corr_energy_err = corr_ecal_energy_err.copy()
    if np.any(valid_trk):
        trk_vals = {name: np.asarray(arr)[valid_trk] for name, arr in vals.items()}
        scale = ecal_scale[valid_trk] if type(ecal_scale) is np.ndarray else ecal_scale
        comb_features = egreg.get_features_comb(
            trk_vals, ecal_mean[valid_trk], ecal_sigma[valid_trk], ecal_scale=scale
        )
        comb_mean, comb_sigma = comb_reg.get_meansigma(comb_features, is_eb[valid_trk])
        raw_comb = egreg.get_raw_comb(trk_vals, ecal_mean[valid_trk], ecal_sigma[valid_trk], ecal_scale=scale)
        corr_energy[valid_trk] = raw_comb * comb_mean
        corr_energy_err[valid_trk] = raw_comb * comb_sigma

    return {
        f"corrEcalEnergy{tag}": corr_ecal_energy,
        f"corrEcalEnergyErr{tag}": corr_ecal_energy_err,
        f"corrEnergy{tag}": corr_energy,
        f"corrEnergyErr{tag}": corr_energy_err,
    }


def fill_mass_hists(vals, counts, corr, id_mask, hists, max_dr=None):
    """Fill the dielectron mass histograms from pairs of ID-passing electrons.

    Each histogram is filled from opposite-charge pairs; the "SS"-suffixed
    variant is filled from same-charge pairs.  If max_dr is given, only
    pairs with deltaR (from the track mode direction) below it are used.
    With split_regions, additional histograms split by the pair's
    detector-region category (EBEB/EBEE/EEEE) are filled.
    """
    sel = ak.unflatten(id_mask, counts)

    def selected(flat_arr):
        return ak.unflatten(flat_arr, counts)[sel]

    eta = selected(vals["bestTrack_etaMode"])
    phi = selected(vals["bestTrack_phiMode"])
    charge1, charge2 = ak.unzip(ak.combinations(selected(vals["bestTrack_charge"]), 2))
    opp_charge = charge1 * charge2 < 0
    same_charge = ~opp_charge
    if max_dr is not None:
        eta1, eta2 = ak.unzip(ak.combinations(eta, 2))
        phi1, phi2 = ak.unzip(ak.combinations(phi, 2))
        dphi = np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi
        pass_dr = (eta1 - eta2) ** 2 + dphi**2 < max_dr**2
        opp_charge = opp_charge & pass_dr
        same_charge = same_charge & pass_dr
    region_cats = {}
    eb1, eb2 = ak.unzip(ak.combinations(selected(np.abs(vals["eta"]) < 1.479), 2))
    region_cats = {"EBEB": eb1 & eb2, "EBEE": eb1 != eb2, "EEEE": ~eb1 & ~eb2}
    cosh_eta = np.cosh(eta)
    pts = {
        "hltMassHist": selected(vals["pt"]),
        "trkMassHist": selected(vals["bestTrack_pt"]),
        "trkModeMassHist": selected(vals["bestTrack_pMode"]) / cosh_eta,
        "caloCorrMassHist": selected(corr["corrEcalEnergy"]) / cosh_eta,
        "caloCorrMassHistLaser": selected(corr["corrEcalEnergyLaser"]) / cosh_eta, 
        "caloTrkMassHist": selected(corr["corrEnergy"]) / cosh_eta,
        "caloTrkMassHistLaser": selected(corr["corrEnergyLaser"]) / cosh_eta,
    }
    for name, pt in pts.items():
        pt_pass1, pt_pass2 = ak.unzip(ak.combinations(pt > 0, 2))
        pt_mask = pt_pass1 & pt_pass2
        p4 = ak.zip(
            {"pt": pt, "eta": eta, "phi": phi, "mass": ak.zeros_like(pt)},
            with_name="Momentum4D",
        )

        ele1, ele2 = ak.unzip(ak.combinations(p4, 2))
        masses = (ele1 + ele2).mass
        for suffix, charge_mask in (("OS", opp_charge), ("SS", same_charge)):
            fills = [(name + suffix, charge_mask & pt_mask)]
            fills += [(name + reg + suffix, charge_mask & region_mask & pt_mask) for reg, region_mask in region_cats.items()]
            for hist_name, hist_mask in fills:
                hists[hist_name] += np.histogram(
                    ak.to_numpy(ak.flatten(masses[hist_mask])), bins=MASS_BINS
                )[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add the regressed energy to the tree (columnar version)")
    parser.add_argument("-i", "--inputfiles", nargs="+", help="Input file(s)", required=True)
    parser.add_argument("-o", "--outputfile", help="Output file", required=True)
    parser.add_argument("-n", "--max-events", type=int, default=None, help="Process at most this many events (total across all files)")
    parser.add_argument("--step-size", default="200 MB", help="uproot iteration step size")
    parser.add_argument("--decomp-threads", type=int, default=8, help="Threads for uproot basket decompression")
    parser.add_argument("--njobs", type=int, default=1, help="Split the input files over this many jobs")
    parser.add_argument("--jobnr", type=int, default=0, help="Which job this is (0..njobs-1); processes inputfiles[jobnr::njobs]")
    parser.add_argument("--jpsi-sel", action="store_true", help="Use the low-mass (J/psi) electron ID and a deltaR<0.7 pair cut for the mass histograms")    
    parser.add_argument("--datadir", default="Scouting/Tools/data", help="Directory with the regression npz files")
    
    args = parser.parse_args()

    write_tree = True

    if len(args.inputfiles) == 1 and not args.inputfiles[0].endswith(".root"):
        with open(args.inputfiles[0]) as f:
            inputfiles = [line.strip() for line in f if line.strip()]
    else:
        inputfiles = args.inputfiles
    
    if not 0 <= args.jobnr < args.njobs:
        parser.error(f"--jobnr must be in [0, {args.njobs})")

    print("inputfiles before job split:", len(inputfiles))
    print("inputfiles after job split:", len(inputfiles[args.jobnr :: args.njobs]))
    inputfiles = inputfiles[args.jobnr :: args.njobs]
    if not inputfiles:
        parser.error(f"job {args.jobnr}/{args.njobs} has no input files ({len(args.inputfiles)} files given)")

    ecal_reg = egreg.RegressionContainer(
        f"{args.datadir}/regEleEcalScout2024_stdVar_stdCuts_{{region}}_ntrees1500_results.npz",
        0.2, 2, 0.0002, 0.5,
    )
    comb_reg = egreg.RegressionContainer(
        f"{args.datadir}/regEleEcalTrkTrainScout2024_stdVar_stdCuts_{{region}}_ntrees1500_results.npz",
        0.2, 3, 0.0002, 0.5,
    )

    hists = {
        name + region + suffix: np.zeros(len(MASS_BINS) - 1)
        for name in ("hltMassHist", "trkMassHist", "trkModeMassHist", 
                     "caloCorrMassHist", "caloTrkMassHist", 
                     "caloCorrMassHistLaser", "caloTrkMassHistLaser")
        for region in ("", "EBEB", "EBEE", "EEEE") 
        for suffix in ("OS", "SS")
    }

    output_file = uproot.recreate(args.outputfile)
    events_written = 0
    start_time = time.time()

    input_trees = [f"{name}:Events" for name in inputfiles]
    total_entries = sum(entries for _, _, entries in uproot.num_entries(input_trees))
    if args.max_events is not None:
        total_entries = min(total_entries, args.max_events)

    decomp_executor = concurrent.futures.ThreadPoolExecutor(args.decomp_threads)
    for events in uproot.iterate(input_trees, BRANCHES, step_size=args.step_size,
                                 decompression_executor=decomp_executor):
        if args.max_events is not None:
            events = events[: args.max_events - events_written]
        vals, counts = get_flat_electrons(events)
        corr = apply_regressions(vals, ecal_reg, comb_reg)
        corr.update(apply_regressions(vals, ecal_reg, comb_reg, ecal_scale=vals["transparencyCorr"], tag="Laser"))
        if args.jpsi_sel:
            id_mask = get_jpsi_id_mask(vals, corr)
        else:
            id_mask = get_id_mask(vals)
        fill_mass_hists(vals, counts, corr, id_mask, hists,
                        max_dr=1.2 if args.jpsi_sel else None,
        )
        if write_tree == 0:
            out_electrons = ak.zip(
                {name: ak.unflatten(arr.astype(np.float32), counts) for name, arr in corr.items()}
            )
            out_chunk = {
                "run": events["run"],
                "luminosityBlock": events["luminosityBlock"],
                "event": events["event"],
                "ScoutingElectron": out_electrons,
            }
            if events_written == 0:
                output_file["Events"] = out_chunk
            else:
                output_file["Events"].extend(out_chunk)

            events_written += len(events)
        rate = events_written / (time.time() - start_time)
        print(f"Processed {events_written}/{total_entries} events ({rate:.0f} ev/s)")
        if args.max_events is not None and events_written >= args.max_events:
            break

    for name, counts_hist in hists.items():
        output_file[name] = (counts_hist, MASS_BINS)
    output_file.close()
