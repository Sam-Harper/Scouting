import argparse
import ROOT
import itertools

from Scouting.Tools.egregression import RegressionContainer
import Scouting.Tools.egregression as egregression

def get_eles_id_mask(event_tree):
    mask = [False]*event_tree.nScoutingElectron
    for ele_nr in range(event_tree.nScoutingElectron):
        if event_tree.ScoutingElectron_bestTrack_etaMode[ele_nr] > 1000:
            continue
        
        is_eb = abs(event_tree.ScoutingElectron_eta[ele_nr]) < 1.479
        if is_eb:
            if event_tree.ScoutingElectron_sigmaIetaIeta[ele_nr] >= 0.014:
                continue
            if event_tree.ScoutingElectron_trackIso[ele_nr] > 5:
                continue
            if abs(event_tree.ScoutingElectron_dEtaIn[ele_nr]) > 0.05:
                continue
        else:            
            if event_tree.ScoutingElectron_sigmaIetaIeta[ele_nr] >= 0.034:
                continue
            if event_tree.ScoutingElectron_trackIso[ele_nr] > 5:
                continue
            if abs(event_tree.ScoutingElectron_dEtaIn[ele_nr]) > 0.05:
                continue

        mask[ele_nr] = True
    return mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add in the regressed energy to the tree')
    parser.add_argument('-i', '--inputfile', help='Input file', required=True)
    parser.add_argument('-o', '--outputfile', help='Output file', required=True)
    args = parser.parse_args()

    input_file = ROOT.TFile(args.inputfile, "READ")
    event_tree = input_file.Events
  
    ecal_reg = RegressionContainer("Scouting/Tools/data/regEleEcalScout2024_stdVar_stdCuts_{region}_ntrees1500_results.root",0.2,2,0.0002,0.5)
    comb_reg = RegressionContainer("Scouting/Tools/data/regEleEcalTrkTrainScout2024_stdVar_stdCuts_{region}_ntrees1500_results.root",0.2,3,0.0002,0.5)

    output_file = ROOT.TFile(args.outputfile,"RECREATE")
    mass_hist = ROOT.TH1D("massHist","",1200,0,120)
    corr_mass_hist = ROOT.TH1D("corrMassHist","",1200,0,120)
    calo_corr_mass_hist = ROOT.TH1D("caloCorrMassHist","",1200,0,120)

    nr_events = int(event_tree.GetEntries())
    for event_indx,event in enumerate(event_tree):
        if event_indx % 10000 == 0:
            print(f"Processing event {event_indx}/{nr_events}")
        ele_passing_id = get_eles_id_mask(event_tree)        
        calo_features = egregression.get_features_calo(event_tree,ele_passing_id)
        calo_meansigmas = [ecal_reg.get_meansigma(f["features"],f["isEB"]) if f is not None else [1.0, 0.0] for f in calo_features]
        comb_features = egregression.get_features_comb(event_tree,calo_meansigmas,ele_passing_id)
        comb_meansigmas = [comb_reg.get_meansigma(f["features"],f["isEB"]) if f is not None else [1.0, 0.0] for f in comb_features]
        raw_comb = egregression.get_raw_comb(event_tree,calo_meansigmas,ele_passing_id)
        

        calo_corr_energy = [egregression.pt_to_p(event_tree.ScoutingElectron_pt[index],event_tree.ScoutingElectron_eta[index])*meansigma[0] for index,meansigma in enumerate(calo_meansigmas)]
        comb_energy = [raw*meansigma[0] for raw,meansigma in zip(raw_comb,comb_meansigmas)]

        p4s = []
        corr_p4s = []
        calo_corr_p4s = []


        for ele_index in range(event_tree.nScoutingElectron):
            if not ele_passing_id[ele_index]:
                continue
            eta = event_tree.ScoutingElectron_bestTrack_etaMode[ele_index]
            phi = event_tree.ScoutingElectron_bestTrack_phiMode[ele_index]
            if event_tree.ScoutingElectron_bestTrack_etaMode[ele_index] > 1000:
                continue
            corr_p4 = egregression.make_p4(comb_energy[ele_index],eta,phi)
            calo_corr_p4 = egregression.make_p4(calo_corr_energy[ele_index],eta,phi)
            p4 = ROOT.Math.PtEtaPhiMVector(event_tree.ScoutingElectron_pt[ele_index], eta, phi, 0.0)
            p4s.append(p4)
            corr_p4s.append(corr_p4)
            calo_corr_p4s.append(calo_corr_p4)


        for ele1_nr, ele2_nr in itertools.combinations(range(len(p4s)),2):
            mass = (p4s[ele1_nr]+p4s[ele2_nr]).mag()
            corr_mass = (corr_p4s[ele1_nr]+corr_p4s[ele2_nr]).mag()
            calo_corr_mass = (calo_corr_p4s[ele1_nr]+calo_corr_p4s[ele2_nr]).mag()
            mass_hist.Fill(mass)
            corr_mass_hist.Fill(corr_mass)
            calo_corr_mass_hist.Fill(calo_corr_mass)

    output_file.Write()
    #output_file.Close()

