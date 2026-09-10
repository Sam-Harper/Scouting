import argparse
import math

import ROOT
from dataclasses import dataclass

def read_hist(root_file, hist_name,rebin=None):
    hist = root_file.Get(hist_name)
    if rebin is not None:
        hist.Rebin(rebin)
    return hist


@dataclass
class MassHist:
    name: str
    hist: ROOT.TH1
    energy: str
    charge: str
    region: str
    label: str

def load_mass_hists(root_file, energies,charges,regions):
    mass_hists = {}
    for energy in energies:
        for charge in charges:
            for region in regions:
                hist_name = f"{energy['tag']}MassHist{region['tag']}{charge['tag']}"
                mass_hists[hist_name] = MassHist(
                    name=hist_name, 
                    hist=read_hist(root_file, hist_name), 
                    energy=energy['tag'],
                    charge=charge['tag'],
                    region=region['tag'],
                    label=f"{energy['label']}, {region['label']}, {charge['label']}"
                    )
    return mass_hists

def set_hist_attributes(hist,index):
        colours = [ROOT.kBlack,ROOT.kRed, ROOT.kBlue]
        colour_index = index % len(colours)
        colour = colours[colour_index]

        marker_styles = [20, 21, 22]
        marker_style = marker_styles[colour_index % len(marker_styles)]

        line_styles = [1, 2]
        line_style_index = math.floor(index / len(colours))
        line_style = line_styles[line_style_index % len(line_styles)]
            
        hist.SetLineColor(colour)
        hist.SetLineStyle(line_style)
        hist.SetMarkerColor(colour)
        hist.SetMarkerStyle(marker_style)


def plot_hists(mass_hists: list[MassHist], canvas: ROOT.TCanvas, rebin, xrange=None):
    canvas.cd()
    if not mass_hists:
        return
    ROOT.leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    ROOT.leg.SetBorderSize(0)
    ROOT.leg.SetFillStyle(0)
    if rebin is not None:
        for mass_hist in mass_hists:
            mass_hist.hist.Rebin(rebin)
    y_max = max(hist.hist.GetMaximum() for hist in mass_hists) if mass_hists else 0
    for hist_nr, mass_hist in enumerate(mass_hists):
        set_hist_attributes(mass_hist.hist, hist_nr)
        if hist_nr == 0:
            if xrange is not None:
                mass_hist.hist.GetXaxis().SetRangeUser(*xrange)
            mass_hist.hist.GetYaxis().SetRangeUser(0, y_max*1.2)
            mass_hist.hist.SetTitle(";M_{ee} [GeV];Events/bin")
            mass_hist.hist.Draw("EP")
        else:
            mass_hist.hist.Draw("SAME EP")

        ROOT.leg.AddEntry(mass_hist.hist, mass_hist.label, "lep")
    ROOT.leg.Draw()
    canvas.Update()
    return mass_hists[0].hist if mass_hists else None

def get_hists(mass_hists, energies = None, charge = None, region=None):
    if energies is not None and not isinstance(energies, list):
        energies = [energies]
    filtered_hists = []
    for hist in mass_hists.values():
        if energies is not None and hist.energy not in energies:
            continue
        if charge is not None and hist.charge != charge:
            continue
        if region is not None and hist.region != region:
            continue
        filtered_hists.append(hist)
    return filtered_hists



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot mass histogram from ROOT files')
    parser.add_argument('in_file', help='Input ROOT file')
    args = parser.parse_args()

    root_file = ROOT.TFile.Open(args.in_file)
    ROOT.gStyle.SetOptStat(0)
    c1 = ROOT.TCanvas("c1", "Mass Histogram", 800, 600)
    c1.SetGridx()
    c1.SetGridy()
    def ls():
        for key in root_file.GetListOfKeys():
            print(key.GetName())

    energies = [
        {"tag" : "hlt", "label": "HLT"},
        {"tag" : "trk", "label": "Track"},
        {"tag" : "trkMode", "label": "Track Mode"},
        {"tag" : "caloCorr", "label": "Retrained Ecal"},
        {"tag" : "caloTrk", "label": "Ecal - Trk"}
    ]    

    charges = [
        {"tag" : "OS", "label": "OS"},
        {"tag" : "SS", "label": "SS"},
    ]
    regions = [
        {"tag" : "EBEB", "label": "EB-EB"},
        {"tag" : "EBEE", "label": "EB-EE"},
        {"tag" : "EEEE", "label": "EE-EE"},
    ]


    mass_hists = load_mass_hists(root_file, energies, charges, regions)

    filtered_hists = get_hists(mass_hists, energies=["hlt", "trkMode","caloTrk"], charge="OS", region="EBEB")
    hist = plot_hists(filtered_hists, c1, rebin=5, xrange=(0, 6))
    
