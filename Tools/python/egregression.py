import ROOT
import math

class BDTTransformer:
    def __init__(self,range_min,range_max):
        self.range_min = range_min
        self.range_max = range_max
        self.offset = self.range_min + 0.5 * (self.range_max - self.range_min)
        self.scale = 0.5 * (self.range_max - self.range_min)
    
    def transform(self,raw_value):
        #features_ptr = features.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return self.offset + self.scale * math.sin(raw_value)

class RegressionContainer:
    def __init__(self,base_file,range_mean_min,range_mean_max,range_sigma_min,range_sigma_max):
        
        
        self.root_file_eb = ROOT.TFile(base_file.format(region="EB"),"READ")
        self.root_file_ee = ROOT.TFile(base_file.format(region="EE"),"READ")
        
        self.forest_mean_eb = getattr(self.root_file_eb,f"EBCorrection")
        self.forest_mean_ee = getattr(self.root_file_ee,f"EECorrection")
        
        self.forest_sigma_eb = getattr(self.root_file_eb,f"EBUncertainty")
        self.forest_sigma_ee = getattr(self.root_file_ee,f"EEUncertainty")

        self.transformer_mean = BDTTransformer(range_mean_min,range_mean_max)
        self.transformer_sigma = BDTTransformer(range_sigma_min,range_sigma_max)
       
        
    def get_meansigma(self,features,isEB):
        forest_mean = self.forest_mean_eb if isEB else self.forest_mean_ee
        forest_sigma = self.forest_sigma_eb if isEB else self.forest_sigma_ee
        
        raw_mean = forest_mean.GetResponse(features.data())
        raw_sigma = forest_sigma.GetResponse(features.data()) 
        return self.transformer_mean.transform(raw_mean),self.transformer_sigma.transform(raw_sigma)

def safe_divide(numer,denom,default_val =0):
    if denom == 0:
        return default_val
    return numer/denom

def eta_to_theta(eta: float) -> float:
    """Convert pseudorapidity eta to polar angle theta (radians)."""
    return 2.0 * math.atan(math.exp(-eta))

def pt_to_p(pt,eta):
    return safe_divide(pt,math.sin(eta_to_theta(eta)),0)

def get_best_trk_indx(calo_pt,calo_eta,trk_pts,trk_etas):
    """
    Returns the index of the track that is closest to the calo pt
    """
    best_trk_indx = 0
    best_abs_epm1 = 9999
    calo_energy = pt_to_p(calo_pt,calo_eta)
    for trk_indx in range(0,trk_pts.size()):
        trk_p = pt_to_p(trk_pts[trk_indx],trk_etas[trk_indx])
        abs_epm1 = abs(trk_p/calo_energy -1)
        if abs_epm1 < best_abs_epm1:
            best_abs_epm1 = abs_epm1
            best_trk_indx = trk_indx

    return best_trk_indx

def get_ietaiphi(seedid):
    detid = ROOT.DetId(seedid)
    if detid.subdetId() == 1:
        ebdetid = ROOT.EBDetId(detid)
        return ebdetid.ieta(),ebdetid.iphi()
    else:
        eedetid = ROOT.EEDetId(detid)
        return eedetid.ix(),eedetid.iy()

def is_eb(seedid):
    detid = ROOT.DetId(seedid)
    return detid.subdetId() == 1

def get_features_calo(tree,ele_mask=None):
    """
    Features for the ECAL-only regression read from the scouting nanoAOD.

    Mirrors RegArgs.set_defaults() var_eb/var_ee in
    EgRegresTrainerLegacy/python/regtools_scouting.py where ele.energy is
    pt converted to p and iEtaOrIX/iPhiOrIY are derived from the seed id
    """
    features = [None]*tree.nScoutingElectron

    if ele_mask is None:
        ele_mask = [True]*tree.nScoutingElectron
    
    for ele_nr in range(tree.nScoutingElectron):
        if not ele_mask[ele_nr]:
            continue

        ele_features = ROOT.std.vector('float')(11)
        seed_id = tree.ScoutingElectron_seedId[ele_nr]
        s_min = tree.ScoutingElectron_sMin[ele_nr]
        s_maj = tree.ScoutingElectron_sMaj[ele_nr]

        ele_features[0] = tree.ScoutingRho_fixedGridRhoFastjetAll
        ele_features[1] = pt_to_p(tree.ScoutingElectron_pt[ele_nr],tree.ScoutingElectron_eta[ele_nr])
        ele_features[2] = tree.ScoutingElectron_eta[ele_nr]
        ele_features[3] = tree.ScoutingElectron_phi[ele_nr]
        ele_features[4] = tree.ScoutingElectron_sigmaIetaIeta[ele_nr]
        ele_features[5] = tree.ScoutingElectron_r9[ele_nr]
        ele_features[6] = 0. if math.isnan(s_min) else s_min
        ele_features[7] = 0. if math.isnan(s_maj) else s_maj
        ele_features[8] = tree.ScoutingElectron_rechitZeroSuppression[ele_nr]
        ele_features[9],ele_features[10] = get_ietaiphi(seed_id)
        features[ele_nr] = {"features" : ele_features,"isEB" : is_eb(seed_id)}
    return features


def get_raw_comb(tree,ecal_meansigmas,ele_mask=None):
    """
    Raw E-p combination, ie the inverse of the RegArgs.set_elecomb_default()
    target without the mc.energy:
       (corrEcalE*trkPErr^2 + trkP*ecalErr^2) / (trkPErr^2 + ecalErr^2)
    with the track quantities taken from the best track mode variables
    """
    raw_comb = [0]*tree.nScoutingElectron
    if ele_mask is None:
        ele_mask = [True]*tree.nScoutingElectron
    for ele_index in range(tree.nScoutingElectron):
        if not ele_mask[ele_index]:
            continue
        calo_e = pt_to_p(tree.ScoutingElectron_pt[ele_index],tree.ScoutingElectron_eta[ele_index])
        ecal_mean,ecal_sigma = ecal_meansigmas[ele_index]
        calo_e_corr = calo_e * ecal_mean
        calo_e_err = calo_e * ecal_sigma
        trk_p = tree.ScoutingElectron_bestTrack_pMode[ele_index]
        trk_p_err = abs(tree.ScoutingElectron_bestTrack_qoverpModeError[ele_index]) * trk_p * trk_p
        numer = calo_e_corr * trk_p_err**2 + trk_p * calo_e_err**2
        denom = trk_p_err**2 + calo_e_err**2
        raw_comb[ele_index] = safe_divide(numer,denom,calo_e_corr)
    return raw_comb

def get_features_comb(tree,ecal_meansigmas,ele_mask=None):
    """
    Features for the E-p combination regression read from the scouting nanoAOD.

    Mirrors RegArgs.set_elecomb_default() var_eb/var_ee in
    EgRegresTrainerLegacy/python/regtools_scouting.py with the track mode
    variables taken directly from the best track branches
    """
    features = [None]*tree.nScoutingElectron
    if ele_mask is None:
        ele_mask = [True]*tree.nScoutingElectron
    for ele_index in range(tree.nScoutingElectron):
        if not ele_mask[ele_index]:
            continue
        ele_features = ROOT.std.vector('float')(8)
        calo_e = pt_to_p(tree.ScoutingElectron_pt[ele_index],tree.ScoutingElectron_eta[ele_index])
        ecal_mean,ecal_sigma = ecal_meansigmas[ele_index]
        calo_e_corr = calo_e * ecal_mean
        trk_p_mode = tree.ScoutingElectron_bestTrack_pMode[ele_index]

        ele_features[0] = calo_e_corr
        ele_features[1] = safe_divide(ecal_sigma,ecal_mean)
        ele_features[2] = safe_divide(tree.ScoutingElectron_bestTrack_qoverpModeError[ele_index],trk_p_mode)
        ele_features[3] = safe_divide(calo_e_corr,trk_p_mode)
        ele_features[4] = tree.ScoutingElectron_r9[ele_index]
        ele_features[5] = tree.ScoutingElectron_trackfbrem[ele_index]
        ele_features[6] = tree.ScoutingElectron_bestTrack_etaMode[ele_index]
        ele_features[7] = tree.ScoutingElectron_bestTrack_phiMode[ele_index]
        features[ele_index] = {"features" : ele_features,"isEB" : is_eb(tree.ScoutingElectron_seedId[ele_index])}
    return features


def make_p4(energy, eta, phi):
    """Massless 4-vector from E, eta, phi. Returns ROOT.Math.PtEtaPhiMVector."""    
    pt = energy / math.cosh(eta)   # massless: |p| = E, pt = E / cosh(eta)
    return ROOT.Math.PtEtaPhiMVector(pt, eta, phi, 0.0)