import argparse
import math

import ROOT
from dataclasses import dataclass

JPSI_MASS_PDG = 3.0969

#keep fit functions alive, ROOT/python gc will otherwise remove them from the canvas
_fit_objs = []

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


def fit_jpsi(hist, fit_range=(2.0, 4.6), exclude=(3.55, 3.85)):
    """fits a gaussian + quadratic bkg to the J/psi peak

    the psi(2S) region is excluded from the fit by default
    returns the fit function and fit result
    """
    def model(x, p):
        if exclude is not None and exclude[0] < x[0] < exclude[1]:
            ROOT.TF1.RejectPoint()
            return 0.
        gaus = p[0] * math.exp(-0.5 * ((x[0] - p[1]) / p[2]) ** 2)
        return gaus + p[3] + p[4] * x[0] + p[5] * x[0] * x[0]

    fit_func = ROOT.TF1(f"{hist.GetName()}JpsiFit", model, fit_range[0], fit_range[1], 6)
    fit_func.SetParNames("N", "mean", "sigma", "a0", "a1", "a2")
    fit_func.SetParameters(
        hist.GetBinContent(hist.FindBin(JPSI_MASS_PDG)),
        JPSI_MASS_PDG,
        0.05,
        hist.GetBinContent(hist.FindBin(fit_range[0])),
        0.,
        0.
    )
    fit_result = hist.Fit(fit_func, "RQN0S")
    _fit_objs.append(fit_func)
    return fit_func, fit_result


def jpsi_yield(fit_func, fit_result, bin_width):
    """signal yield (and error) from the gaussian normalisation and width"""
    n, sigma = fit_func.GetParameter(0), abs(fit_func.GetParameter(2))
    yield_ = n * sigma * math.sqrt(2 * math.pi) / bin_width
    yield_err = 0.
    if int(fit_result) == 0:
        var = (sigma ** 2 * fit_result.CovMatrix(0, 0)
               + n ** 2 * fit_result.CovMatrix(2, 2)
               + 2 * n * sigma * fit_result.CovMatrix(0, 2))
        yield_err = math.sqrt(max(var, 0.)) * math.sqrt(2 * math.pi) / bin_width
    return yield_, yield_err


def draw_jpsi_fit(mass_hist, colour, fit_range=(2.0, 4.6), exclude=(3.55, 3.85), legend=None):
    """fits the J/psi peak and draws the fit (and bkg only component) on the current pad"""
    fit_func, fit_result = fit_jpsi(mass_hist.hist, fit_range, exclude)

    mean, mean_err = fit_func.GetParameter(1), fit_func.GetParError(1)
    sigma, sigma_err = abs(fit_func.GetParameter(2)), fit_func.GetParError(2)
    yield_, yield_err = jpsi_yield(fit_func, fit_result, mass_hist.hist.GetBinWidth(1))
    chi2, ndf = fit_result.Chi2(), fit_result.Ndf()
    print(f"{mass_hist.label}: mu = {mean:.4f} +/- {mean_err:.4f} GeV, "
          f"sigma = {sigma*1000:.1f} +/- {sigma_err*1000:.1f} MeV, "
          f"yield = {yield_:.0f} +/- {yield_err:.0f}, chi2/ndf = {chi2:.1f}/{ndf}")

    #the fitted func rejects points in the excluded region so make smooth ones for drawing
    total_func = ROOT.TF1(f"{fit_func.GetName()}Draw",
                          "[0]*exp(-0.5*((x-[1])/[2])^2)+[3]+[4]*x+[5]*x*x",
                          fit_range[0], fit_range[1])
    bkg_func = ROOT.TF1(f"{fit_func.GetName()}BkgDraw", "[0]+[1]*x+[2]*x*x",
                        fit_range[0], fit_range[1])
    for par_nr in range(fit_func.GetNpar()):
        total_func.SetParameter(par_nr, fit_func.GetParameter(par_nr))
    for par_nr in range(bkg_func.GetNpar()):
        bkg_func.SetParameter(par_nr, fit_func.GetParameter(par_nr + 3))
    for func, line_style in ((total_func, 1), (bkg_func, 2)):
        func.SetLineColor(colour)
        func.SetLineStyle(line_style)
        func.SetLineWidth(2)
        func.SetNpx(400)
        func.Draw("SAME")
        _fit_objs.append(func)
    if legend is not None:
        legend.AddEntry(total_func, f"#mu = {mean:.3f} #pm {mean_err:.3f} GeV", "l")
        legend.AddEntry(ROOT.nullptr,
                        f"#sigma = {sigma*1000:.0f} #pm {sigma_err*1000:.0f} MeV", "")
    return fit_func, fit_result


def plot_hists(mass_hists: list[MassHist], canvas: ROOT.TCanvas, 
               rebin, xrange=None, fit=False, normalise=False):
    canvas.cd()
    if not mass_hists:
        return
    if normalise:
        norm_target = mass_hists[0].hist.Integral()
        for mass_hist in mass_hists[1:]:
            integral = mass_hist.hist.Integral()
            if integral > 0:
                mass_hist.hist.Scale(norm_target / integral)
    nr_leg_entries = len(mass_hists) * (3 if fit else 1)
    #ROOT.leg = ROOT.TLegend(0.6, max(0.5, 0.9 - 0.045 * nr_leg_entries), 0.9, 0.9)
    ROOT.leg = ROOT.TLegend(0.6, max(0.5, 0.9 - 0.1 * nr_leg_entries), 0.91, 0.87)

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
            mass_hist.hist.GetYaxis().SetTitleOffset(1.3)
            mass_hist.hist.SetTitle(";M_{ee} [GeV];Events/bin")
            mass_hist.hist.Draw("EP")
        else:
            mass_hist.hist.Draw("SAME EP")

        ROOT.leg.AddEntry(mass_hist.hist, mass_hist.label, "lep")
        if fit:
            draw_jpsi_fit(mass_hist, mass_hist.hist.GetLineColor(), legend=ROOT.leg)
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
    parser.add_argument('--fit', action='store_true', help='fit the J/psi peak of each plotted hist')
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

    filtered_hists = get_hists(mass_hists, energies=["hlt", "trkMode","caloTrk"], 
                               charge="OS", region="EBEB")
    filtered_hists = get_hists(mass_hists, energies=["caloTrk"], 
                               charge=None, region="EBEB")

    regions = ["EBEB", "EBEE", "EEEE"]
    regions = ["EBEB"]
    energies = ["hlt", "trkMode", "caloTrk"]
    def plot_all_hists():
        for energy in energies:
            for region in regions:
                filtered_hists = get_hists(mass_hists, energies=[energy], 
                                           charge=None, region=region)
                plot_hists(filtered_hists, c1, rebin=5, xrange=(0, 6), fit=args.fit,
                           normalise=True)
                yield
    gen  = plot_all_hists()
    #hist = plot_hists(filtered_hists, c1, rebin=5, xrange=(0, 6), fit=args.fit,
    #                   normalise=False)

