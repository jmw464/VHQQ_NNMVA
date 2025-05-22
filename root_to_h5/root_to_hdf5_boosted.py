import os,sys,glob
import argparse
import numpy as np
import awkward as ak
import h5py
import uproot


incl_mva_scores = True

GN2X_Hbb_WPs = {50: 4.335, 55: 4.087, 60: 3.818, 65: 3.518, 70: 3.166, 75: 2.735, 80: 2.211, 85: 1.560}
GN2X_Hcc_WPs = {10: 4.050, 20: 3.352, 30: 2.810, 40: 2.319, 50: 1.851, 60: 1.370}
GN2_b_WPs = {65: 2.669, 70: 1.892, 77: 0.844, 85: -0.378, 90: -1.34}
GN2_c_WPs = {10: 3.958, 30: 2.09, 50: 0.503}

hcc = ['ggZllHcc_PP8', 'ggZvvHcc_PP8', 'qqWlvHccJ_PP8', 'qqZllHccJ_PP8', 'qqZvvHccJ_PP8']
hbb = ['ggZllHbb_PP8', 'ggZvvHbb_PP8', 'qqWlvHbbJ_PP8', 'qqZllHbbJ_PP8', 'qqZvvHbbJ_PP8']
diboson = ['ggWqqWlv_Sh222', 'ggZqqZll_Sh222', 'ggZqqZvv_Sh222', 'WlvWqq_Sh2211', 'WlvZbb_Sh2211', 'WlvZqq_Sh2211', 'WqqZll_Sh2211', 'WqqZvv_Sh2211', 'ZbbZll_Sh2211', 'ZbbZvv_Sh2211', 'ZqqZll_Sh2211', 'ZqqZvv_Sh2211']
wjets = ['WenuB_Sh2211', 'WenuC_Sh2211', 'WenuL_Sh2211', 'WmunuB_Sh2211', 'WmunuC_Sh2211', 'WmunuL_Sh2211', 'WtaunuB_Sh2211', 'WtaunuC_Sh2211', 'WtaunuL_Sh2211']
zjets = ['ZeeB_Sh2211', 'ZeeC_Sh2211', 'ZeeL_Sh2211', 'ZmumuB_Sh2211', 'ZmumuC_Sh2211', 'ZmumuL_Sh2211', 'ZnunuB_Sh2211', 'ZnunuC_Sh2211', 'ZnunuL_Sh2211', 'ZtautauB_Sh2211', 'ZtautauC_Sh2211', 'ZtautauL_Sh2211']
top = ['stopWt_DS_PwPy8', 'stopWt_DS_PwPy8_METfilt', 'stopWt_DS_PwPy8_dilep', 'stops_PwPy8', 'stopt_PwPy8', 'ttbar_nonallhad_PwPy8', 'ttbar_nonallhad_PwPy8_METfilt', 'ttbar_dilep_PwPy8', 'ttbar_nonallhad_PwPy8_100_200pTV', 'ttbar_nonallhad_PwPy8_200pTV']


def main(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-w", "--overwrite", action="store_true", default=False, dest="overwrite", help="overwrite existing files")
    parser.add_argument("-mc", "--mc_campaign", type=str, required=True, dest="mc_campaign", help="MC campaign (a, d or e)")
    parser.add_argument("-l", "--lepton_channel", type=int, required=True, dest="lepton_channel", help="lepton channel (0, 1 or 2)")
    parser.add_argument("-i", "--indir", type=str, required=True, dest="indir", help="input directory")
    parser.add_argument("-o", "--outdir", type=str, required=True, dest="outdir", help="output directory") 
    args = parser.parse_args()

    mc_campaign = args.mc_campaign
    lepton_channel = str(args.lepton_channel)+"L"
    overwrite = args.overwrite

    indir = args.indir+"/Reader_"+lepton_channel+"_33-24_"+mc_campaign+"_Merged_D_D/data-MVATree/"
    outdir = args.outdir+lepton_channel+"_"+mc_campaign+"/"

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    rootfiles = sorted(glob.glob(indir+'/*.root'))

    for infilename in rootfiles:
        outfilename = outdir+"/"+infilename.split("/")[-1].replace(".root", ".h5")
        sample = infilename.split("/")[-1].split(".")[0]

        valid_dtype = h5py.enum_dtype({"FALSE": 0, "TRUE": 1}, basetype="<i1")

        if os.path.exists(outfilename) and not overwrite:
            print("Skip {} - File already found".format(infilename.split("/")[-1]))
            continue
        else:
            print("Begin processing {}".format(infilename.split("/")[-1]))

        infile = uproot.open(infilename)
        outfile = h5py.File(outfilename+'.part',"w")

        if infile.keys() == []:
            print("Skip {} - No tree found".format(infilename.split("/")[-1]))
            os.remove(outfilename+'.part')
            continue

        tree = infile["Nominal"]
        nsmalljets = 15 # maximum number of small-R jets per event
        nlargejets = 5 # maximum number of large-R jets per event

        # implement large weight protection
        weights = tree["EventWeight"].array(library="np")
        if sample in ["ggZllHbb_PP8", "ggZvvHbb_PP8", "ggZllHcc_PP8", "ggZvvHcc_PP8"]:
            mask = weights < 1.0
        else:
            mask = np.ones_like(weights, dtype=bool)

        nevents = np.sum(mask)
        if nevents == 0:
            print("Skip {} - No events found".format(infilename.split("/")[-1]))
            os.remove(outfilename+'.part')
            continue

        event_base_dtypes = np.dtype([('EventNumber', '<i4'), ('EventNumberHash10', '<i2'), ('EventNumberHash8', '<i2'), ('EventNumberHash6', '<i2'), ('EventWeight', '<f4'), ('sampleID', '<i1'), ('FlavourLabel', '<i4'), ('DescriptionID', '<i4'), ('LeptonRegion', '<i2'), ('nTaus', '<i4'), ('MET', '<f4'), ('METSig', '<f4'), ('softMET', '<f4'), ('SUSYMET', '<f4'), ('MEffBoosted', '<f4'), ('absdeltaPhiVJ', '<f4'), ('deltaYVJ', '<f4'), ('lepPtBalance', '<f4'), ('HTBoosted', '<f4'), ('pTV', '<f4'), ('phiV', '<f4'), ('mJ', '<f4'), ('pTJ', '<f4'), ('D2', '<f4'), ('C2', '<f4'), ('NAdditionalCaloJets', '<i4'), ('NLargeRJets', '<i4'), ('pTsmallJ1', '<f4'), ('pTsmallJ2', '<f4'), ('dRsmallRJ1J2', '<f4'), ('dRsmallRJ1J3', '<f4'), ('dRsmallRJ2J3', '<f4'), ('dRlargeRJsmallRJ1', '<f4'), ('dRlargeRJsmallRJ2', '<f4'), ('dRlargeRJsmallRJ3', '<f4'), ('XbbScore', '<f4'), ('XccScore', '<f4'), ('XbbTag70', '<i4'), ('XbbWP', '<i4'), ('XccTag30', '<i4'), ('XccWP', '<i4')])
        event_01L_dtypes = np.dtype([('pTL', '<f4'), ('etaL', '<f4'), ('phiL', '<f4')])
        event_2L_dtypes = np.dtype([('pTL1', '<f4'), ('pTL2', '<f4'), ('etaL1', '<f4'), ('etaL2', '<f4'), ('phiL1', '<f4'), ('phiL2', '<f4'), ('cosThetaLep', '<f4')])
        event_mva_dtypes = np.dtype([('mva_transformer_phbb', '<f4'), ('mva_transformer_phcc', '<f4'), ('mva_transformer_pdiboson', '<f4'), ('mva_transformer_pwjets', '<f4'), ('mva_transformer_pzjets', '<f4'), ('mva_transformer_ptop', '<f4'), ('mva_transformer_dhbb', '<f4'), ('mva_transformer_dhcc', '<f4'), ('mva_dnn_phbb', '<f4'), ('mva_dnn_phcc', '<f4'), ('mva_dnn_pdiboson', '<f4'), ('mva_dnn_pwjets', '<f4'), ('mva_dnn_pzjets', '<f4'), ('mva_dnn_ptop', '<f4'), ('mva_dnn_dhbb', '<f4'), ('mva_dnn_dhcc', '<f4')])
        event_dtypes = np.dtype(event_base_dtypes.descr + event_2L_dtypes.descr)
        if incl_mva_scores: event_dtypes = np.dtype(event_dtypes.descr + event_mva_dtypes.descr)
        event_features = {}

        smallR_dtypes = np.dtype([('sj_pt', '<f4'), ('sj_eta', '<f4'), ('sj_phi', '<f4'), ('sj_m', '<f4'), ('sj_pb', '<f4'), ('sj_pc', '<f4'), ('sj_pu', '<f4'), ('sj_ptau', '<f4'), ('GN2btag70', '<i4'), ('GN2ctag30', '<i4'), ('GN2bWP', '<i4'), ('GN2cWP', '<i4'), ('valid', valid_dtype)])
        smallR_features = {}

        largeR_dtypes = np.dtype([('fj_pt', '<f4'), ('fj_eta', '<f4'), ('fj_phi', '<f4'), ('fj_m', '<f4'), ('fj_phbb', '<f4'), ('fj_phcc', '<f4'), ('fj_pqcd', '<f4'), ('fj_ptop', '<f4'), ('GN2XHbbtag70', '<i4'), ('GN2XHcctag30', '<i4'), ('GN2XHbbWP', '<i4'), ('GN2XHccWP', '<i4'), ('valid', valid_dtype)])
        largeR_features = {}

        event_dataset = outfile.create_dataset('events', dtype=event_dtypes, shape=(nevents,))#, compression="lzf")
        event_dataset.attrs["sampleID"] = ["hcc", "hbb", "diboson", "wjets", "zjets", "top"]
        outfile.create_dataset('smallRjets', dtype=smallR_dtypes, shape=(nevents,nsmalljets))#, compression="lzf")
        outfile.create_dataset('largeRjets', dtype=largeR_dtypes, shape=(nevents,nlargejets))#, compression="lzf")

        for name in event_base_dtypes.names:
            if name == "sampleID":
                if sample in hcc:
                    event_features[name] = np.full(nevents, 0)
                elif sample in hbb:
                    event_features[name] = np.full(nevents, 1)
                elif sample in diboson:
                    event_features[name] = np.full(nevents, 2)
                elif sample in wjets:
                    event_features[name] = np.full(nevents, 3)
                elif sample in zjets:
                    event_features[name] = np.full(nevents, 4)
                elif sample in top:
                    event_features[name] = np.full(nevents, 5)
            elif name == "DescriptionID":
                event_features[name] = np.full(nevents, 0)
                event_features[name][tree["Description"].array(library="np")[mask] == 'SR'] = 0
                # ADD CRs once defined
            elif name == "LeptonRegion":
                event_features[name] = np.full(nevents, args.lepton_channel)
            elif name == "NLargeRJets":
                event_features[name] = np.array([array.size for array in tree["fj_pt"].array(library="np")])[mask]
            elif name == "XbbTag70": # whether event passes 70% Hbb WP
                event_features[name] = tree["XbbScore"].array(library="np")[mask] > GN2X_Hbb_WPs[70]
            elif name == "XbbWP": # tightest WP that passes
                event_features[name] = np.full(nevents, 100)
                wp_keys = list(GN2X_Hbb_WPs.keys())
                wp_keys.sort(reverse=True)
                for WP in wp_keys:
                    event_features[name][tree["XbbScore"].array(library="np")[mask] > GN2X_Hbb_WPs[WP]] = WP
            elif name == "XccTag30": # whether event passes 30% Hcc WP
                event_features[name] = tree["XccScore"].array(library="np")[mask] > GN2X_Hcc_WPs[30]
            elif name == "XccWP": # tightest WP that passes
                event_features[name] = np.full(nevents, 100)
                wp_keys = list(GN2X_Hcc_WPs.keys())
                wp_keys.sort(reverse=True)
                for WP in wp_keys:
                    event_features[name][tree["XccScore"].array(library="np")[mask] > GN2X_Hcc_WPs[WP]] = WP
            elif name == "EventNumberHash10":
                event_features[name] = tree["EventNumber"].array(library="np")[mask] % 10
            elif name == "EventNumberHash8":
                event_features[name] = tree["EventNumber"].array(library="np")[mask] % 8
            elif name == "EventNumberHash6":
                event_features[name] = tree["EventNumber"].array(library="np")[mask] % 6
            else:
                event_features[name] = tree[name].array(library="np")[mask].astype(event_dtypes[name].type)

        for name in event_2L_dtypes.names:
            if lepton_channel == "2L":
                event_features[name] = tree[name].array(library="np")[mask].astype(event_2L_dtypes[name].type)
            else:
                event_features[name] = np.full((nevents,), -99).astype(event_2L_dtypes[name].type)

        for name in event_01L_dtypes.names:
            if lepton_channel == "0L" or lepton_channel == "1L":
                event_features[name+"1"] = tree[name].array(library="np")[mask]

        if incl_mva_scores:
            for name in event_mva_dtypes.names:
                event_features[name] = tree[name].array(library="np")[mask].astype(event_mva_dtypes[name].type)
        
        for name in smallR_dtypes.names:
            if name == "valid" or name == "GN2btag70" or name == "GN2ctag30" or name == "GN2bWP" or name == "GN2cWP":
                continue
            jagged_array = tree[name].array(library="ak")[mask]
            max_length = max(nsmalljets, np.max(ak.num(jagged_array)))
            rect_array = ak.fill_none(ak.pad_none(jagged_array, max_length), -99.)
            smallR_features[name] = ak.to_numpy(rect_array).astype(smallR_dtypes[name].type)[:,:nsmalljets]

        smallR_features["valid"] = (smallR_features["sj_pt"] > 0).astype(valid_dtype)

        smallR_Db = np.log(smallR_features["sj_pb"]/(0.2*smallR_features["sj_pc"]+0.79*smallR_features["sj_pu"]+0.01*smallR_features["sj_ptau"]))
        smallR_features["GN2btag70"] = (smallR_Db > GN2_b_WPs[70]).astype(np.int32)
        smallR_features["GN2bWP"] = np.full((nevents, nsmalljets), 100)
        b_wp_keys = list(GN2_b_WPs.keys())
        b_wp_keys.sort(reverse=True)
        for WP in b_wp_keys:
            smallR_features["GN2bWP"][smallR_Db > GN2_b_WPs[WP]] = WP
        smallR_features["GN2btag70"][np.logical_not(smallR_features["valid"])] = -99
        smallR_features["GN2bWP"][np.logical_not(smallR_features["valid"])] = -99
 
        smallR_Dc = np.log(smallR_features["sj_pc"]/(0.3*smallR_features["sj_pb"]+0.65*smallR_features["sj_pu"]+0.05*smallR_features["sj_ptau"]))
        smallR_features["GN2ctag30"] = (smallR_Dc > GN2_c_WPs[30]).astype(np.int32)
        smallR_features["GN2cWP"] = np.full((nevents, nsmalljets), 100)
        c_wp_keys = list(GN2_c_WPs.keys())
        c_wp_keys.sort(reverse=True)
        for WP in c_wp_keys:
            smallR_features["GN2cWP"][smallR_Dc > GN2_c_WPs[WP]] = WP
        smallR_features["GN2ctag30"][np.logical_not(smallR_features["valid"])] = -99
        smallR_features["GN2cWP"][np.logical_not(smallR_features["valid"])] = -99

        for name in largeR_dtypes.names:
            if name == "valid" or name == "GN2XHbbtag70" or name == "GN2XHcctag30" or name == "GN2XHbbWP" or name == "GN2XHccWP":
                continue
            jagged_array = tree[name].array(library="ak")[mask]
            max_length = max(nlargejets, np.max(ak.num(jagged_array)))
            rect_array = ak.fill_none(ak.pad_none(jagged_array, max_length), -99.)
            largeR_features[name] = ak.to_numpy(rect_array).astype(largeR_dtypes[name].type)[:,:nlargejets]

        largeR_features["valid"] = (largeR_features["fj_pt"] >= 0.).astype(valid_dtype)

        largeR_DHbb = np.log(largeR_features["fj_phbb"]/(0.02*largeR_features["fj_phcc"]+0.73*largeR_features["fj_pqcd"]+0.25*largeR_features["fj_ptop"]))
        largeR_DHbb[np.logical_not(largeR_features["valid"])] = -99.
        largeR_features["GN2XHbbtag70"] = (largeR_DHbb > GN2X_Hbb_WPs[70]).astype(np.int32)
        largeR_features["GN2XHbbWP"] = np.full((nevents, nlargejets), 100)
        hbb_wp_keys = list(GN2X_Hbb_WPs.keys())
        hbb_wp_keys.sort(reverse=True)
        for WP in hbb_wp_keys:
            largeR_features["GN2XHbbWP"][largeR_DHbb > GN2X_Hbb_WPs[WP]] = WP
        largeR_features["GN2XHbbtag70"][np.logical_not(largeR_features["valid"])] = -99
        largeR_features["GN2XHbbWP"][np.logical_not(largeR_features["valid"])] = -99

        largeR_DHcc = np.log(largeR_features["fj_phcc"]/(0.3*largeR_features["fj_phbb"]+0.45*largeR_features["fj_pqcd"]+0.25*largeR_features["fj_ptop"]))
        largeR_features["GN2XHcctag30"] = (largeR_DHcc > GN2X_Hcc_WPs[30]).astype(np.int32)
        largeR_features["GN2XHccWP"] = np.full((nevents, nlargejets), 100)
        hcc_wp_keys = list(GN2X_Hcc_WPs.keys())
        hcc_wp_keys.sort(reverse=True)
        for WP in hcc_wp_keys:
            largeR_features["GN2XHccWP"][largeR_DHcc > GN2X_Hcc_WPs[WP]] = WP
        largeR_features["GN2XHcctag30"][np.logical_not(largeR_features["valid"])] = -99
        largeR_features["GN2XHccWP"][np.logical_not(largeR_features["valid"])] = -99

        for variable in outfile['events'].dtype.fields.keys():
            outfile['events'][variable] = event_features[variable]
        for variable in outfile['smallRjets'].dtype.fields.keys():
            outfile['smallRjets'][variable] = smallR_features[variable]
        for variable in outfile['largeRjets'].dtype.fields.keys():
            outfile['largeRjets'][variable] = largeR_features[variable]

        outfile.close()
        infile.close()

        os.rename(outfilename+'.part', outfilename)


if __name__ == '__main__':
    main(sys.argv) 