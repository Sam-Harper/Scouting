"""
numpy version of Scouting.Tools.egregression.

All functions operate on flat per-electron numpy arrays 
which are one entry per electron candidate
"""
import numpy as np

try:
    from Scouting.Tools.gbrforest_np import load_regression
except ImportError:
    from gbrforest_np import load_regression


class BDTTransformer:
    def __init__(self, range_min, range_max):
        self.offset = range_min + 0.5 * (range_max - range_min)
        self.scale = 0.5 * (range_max - range_min)

    def transform(self, raw_value):
        return self.offset + self.scale * np.sin(raw_value)


class RegressionContainer:
    """Numpy equivalent of egregression.RegressionContainer.

    base_file is the npz template (with {region} placeholder) produced by
    Scouting/Tools/scripts/dumpForests.py.
    """

    def __init__(self, base_file, range_mean_min, range_mean_max, range_sigma_min, range_sigma_max):
        self.forests = load_regression(base_file)
        self.transformer_mean = BDTTransformer(range_mean_min, range_mean_max)
        self.transformer_sigma = BDTTransformer(range_sigma_min, range_sigma_max)

    def get_meansigma(self, features, is_eb):
        """features: (n, n_features) matrix, is_eb: (n,) bool array.

        Returns (mean, sigma) float64 arrays of transformed responses.
        """
        is_eb = np.asarray(is_eb, dtype=bool)
        raw_mean = np.empty(len(is_eb), dtype=np.float64)
        raw_sigma = np.empty(len(is_eb), dtype=np.float64)
        for region, mask in (("EB", is_eb), ("EE", ~is_eb)):
            forest_mean, forest_sigma = self.forests[region]
            raw_mean[mask] = forest_mean.get_response(features[mask])
            raw_sigma[mask] = forest_sigma.get_response(features[mask])
        return self.transformer_mean.transform(raw_mean), self.transformer_sigma.transform(raw_sigma)


def safe_divide(numer, denom, default_val=0.0):
    denom = np.asarray(denom, dtype=np.float64)
    is_zero = denom == 0
    result = np.asarray(numer, dtype=np.float64) / np.where(is_zero, 1.0, denom)
    return np.where(is_zero, default_val, result)


def pt_to_p(pt, eta):
    """p = pt / sin(theta), matching egregression.pt_to_p numerically."""
    sin_theta = np.sin(2.0 * np.arctan(np.exp(-np.asarray(eta, dtype=np.float64))))
    return safe_divide(pt, sin_theta, 0.0)


def decode_seed_id(seed_id):
    """Decode ECAL DetIds: returns (ietaOrIx, iphiOrIy, is_eb) arrays.

    Bit layout from DataFormats/EcalDetId EBDetId/EEDetId:
      EB: |ieta| bits 9-15, iphi bits 0-8, zside bit 16 (set = +z)
      EE: ix bits 7-13, iy bits 0-6
    """
    sid = np.asarray(seed_id).astype(np.int64)
    is_eb = ((sid >> 25) & 0x7) == 1

    ieta = np.where(sid & 0x10000, 1, -1) * ((sid >> 9) & 0x7F)
    iphi = sid & 0x1FF

    ix = (sid >> 7) & 0x7F
    iy = sid & 0x7F

    return np.where(is_eb, ieta, ix), np.where(is_eb, iphi, iy), is_eb


def get_features_calo(vals, ecal_scale=1.0):
    """Feature matrix for the ECAL-only regression.

    vals: dict of flat per-electron arrays with keys
      rho, pt, eta, phi, sigmaIetaIeta, r9, sMin, sMaj,
      rechitZeroSuppression, seedId
    Returns (features (n, 11) float32, is_eb (n,) bool).
    """
    ieta_or_ix, iphi_or_iy, is_eb = decode_seed_id(vals["seedId"])
    features = np.column_stack(
        [
            vals["rho"],
            pt_to_p(vals["pt"], vals["eta"]) * ecal_scale,
            vals["eta"],
            vals["phi"],
            vals["sigmaIetaIeta"],
            vals["r9"],
            np.where(np.isnan(vals["sMin"]), 0.0, vals["sMin"]),
            np.where(np.isnan(vals["sMaj"]), 0.0, vals["sMaj"]),
            vals["rechitZeroSuppression"],
            ieta_or_ix,
            iphi_or_iy,
        ]
    ).astype(np.float32)
    return features, is_eb


def get_raw_comb(vals, ecal_mean, ecal_sigma, ecal_scale=1.0):
    """Raw E-p combination (see egregression.get_raw_comb).

    vals needs keys pt, eta, bestTrack_pMode, bestTrack_qoverpModeError.
    """
    calo_e = pt_to_p(vals["pt"], vals["eta"]) * ecal_scale
    calo_e_corr = calo_e * ecal_mean
    calo_e_err = calo_e * ecal_sigma
    trk_p = np.asarray(vals["bestTrack_pMode"], dtype=np.float64)
    trk_p_err = np.abs(vals["bestTrack_qoverpModeError"]) * trk_p * trk_p
    numer = calo_e_corr * trk_p_err**2 + trk_p * calo_e_err**2
    denom = trk_p_err**2 + calo_e_err**2
    return np.where(denom == 0, calo_e_corr, numer / np.where(denom == 0, 1.0, denom))


def get_features_comb(vals, ecal_mean, ecal_sigma,ecal_scale=1.0):
    """Feature matrix for the E-p combination regression.

    vals needs keys pt, eta, r9, trackfbrem, bestTrack_pMode,
    bestTrack_qoverpModeError, bestTrack_etaMode, bestTrack_phiMode.
    Returns a (n, 8) float32 matrix.
    """
    calo_e_corr = pt_to_p(vals["pt"], vals["eta"]) * ecal_mean * ecal_scale
    trk_p_mode = vals["bestTrack_pMode"]
    return np.column_stack(
        [
            calo_e_corr,
            safe_divide(ecal_sigma, ecal_mean),
            safe_divide(vals["bestTrack_qoverpModeError"], trk_p_mode),
            safe_divide(calo_e_corr, trk_p_mode),
            vals["r9"],
            vals["trackfbrem"],
            vals["bestTrack_etaMode"],
            vals["bestTrack_phiMode"],
        ]
    ).astype(np.float32)
