import pandas as pd

# 2M Precursor 40M PSM, 1M Precursor 20M PSM,  0.5M Precursor 10M PSM, 0.25 Precursor 5M PSM
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from pandarallel import pandarallel
from peptdeep.pretrained_models import ModelManager
import torch
import itertools
from peptdeep.model.ms2 import ModelMS2pDeep, pDeepModel, calc_ms2_similarity, ModelMS2Transformer, ModelMS2Bert
from alphabase.peptide.fragment import (
    init_fragment_by_precursor_dataframe,
    update_sliced_fragment_dataframe,
    get_sliced_fragment_dataframe,
    get_charged_frag_types,
)
from msnet_trainer import MSNetMS2Model as CustomModelMannger
from peptdeep.settings import global_settings as settings, model_const
import glob


def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


loss_fuc = torch.nn.L1Loss()


def cal_loss(target, pred):
    target = target / target.max().max()
    target = torch.tensor(target, dtype=torch.float32).view(-1, pred.shape[0], 4)
    pred = torch.tensor(pred, dtype=torch.float32).view(-1, pred.shape[0], 4)
    test_loss = loss_fuc(pred, target)
    return test_loss


M40 = ModelManager(mask_modloss=True)
M40.load_external_models(ms2_model_file="./DDP/final_model_7.pt")
print("M40 parameters: {0}".format(sum(p.numel() for p in M40.ms2_model.model.parameters())))

M20 = ModelManager(mask_modloss=True)
M20.load_external_models(ms2_model_file="./DDP/scaling_law_model_P1.pt")
print("M20 parameters: {0}".format(sum(p.numel() for p in M20.ms2_model.model.parameters())))

M10 = ModelManager(mask_modloss=True)
M10.load_external_models(ms2_model_file="./DDP/scaling_law_model_P2.pt")
print("M10 parameters: {0}".format(sum(p.numel() for p in M10.ms2_model.model.parameters())))

M5 = ModelManager(mask_modloss=True)
M5.load_external_models(ms2_model_file="./DDP/scaling_law_model_P3.pt")
print("M5 parameters: {0}".format(sum(p.numel() for p in M5.ms2_model.model.parameters())))

M2 = ModelManager(mask_modloss=True)
M2.load_external_models(ms2_model_file="./DDP/scaling_law_model_P4.pt")
print("M2 parameters: {0}".format(sum(p.numel() for p in M2.ms2_model.model.parameters())))


def make_prediction_2(inputs):
    l4_512 = CustomModelMannger(mask_modloss=True)
    l4_512.ms2_model.build(ModelMS2Bert,
                           num_frag_types=4,
                           dropout=0.1,
                           nlayers=4,
                           hidden=512)

    l4_512.load_external_models(ms2_model_file="MSNet_Epoch100_ratio_full_layer4_512.pt")
    print("p1 parameters: {0}".format(sum(p.numel() for p in l4_512.ms2_model.model.parameters())))

    l4_512 = l4_512.predict_all(precursor_df=inputs, predict_items=["ms2"],
                                frag_types=['b_z1', 'y_z1', 'b_z2', 'y_z2'])

    l1 = CustomModelMannger(mask_modloss=True)
    l1.ms2_model.build(ModelMS2Bert,
                       num_frag_types=4,
                       dropout=0.1,
                       nlayers=1)

    l1.load_external_models(ms2_model_file="MSNet_Epoch100_ratio_full_layer1.pt")
    print("p1 parameters: {0}".format(sum(p.numel() for p in l1.ms2_model.model.parameters())))

    l1 = l1.predict_all(precursor_df=inputs, predict_items=["ms2"],
                        frag_types=['b_z1', 'y_z1', 'b_z2', 'y_z2'])

    l2 = CustomModelMannger(mask_modloss=True)
    l2.ms2_model.build(ModelMS2Bert,
                       num_frag_types=4,
                       dropout=0.1,
                       nlayers=2)

    l2.load_external_models(ms2_model_file="MSNet_Epoch100_ratio_full_layer2.pt")
    print("p2 parameters: {0}".format(sum(p.numel() for p in l2.ms2_model.model.parameters())))
    #
    l3 = CustomModelMannger(mask_modloss=True)
    l3.ms2_model.build(ModelMS2Bert,
                       num_frag_types=4,
                       dropout=0.1,
                       nlayers=3)

    l3.load_external_models(ms2_model_file="MSNet_Epoch100_ratio_full_layer3.pt")
    print("p3 parameters: {0}".format(sum(p.numel() for p in l3.ms2_model.model.parameters())))

    l3_128 = CustomModelMannger(mask_modloss=True)
    l3_128.ms2_model.build(ModelMS2Bert,
                           num_frag_types=4,
                           dropout=0.1,
                           nlayers=3,
                           hidden=128)
    
    l3_128.load_external_models(ms2_model_file="MSNet_Epoch100_ratio_full_layer3_128.pt")
    print("p3_128 parameters: {0}".format(sum(p.numel() for p in l3_128.ms2_model.model.parameters())))
    #
    l4 = CustomModelMannger(mask_modloss=True)
    l4.ms2_model.build(ModelMS2Bert,
                       num_frag_types=4,
                       dropout=0.1,
                       nlayers=4)

    l4.load_external_models(ms2_model_file="MSNet_Epoch100_ratio_full_layer4_dp01.pt")
    print("l4 parameters: {0}".format(sum(p.numel() for p in l4.ms2_model.model.parameters())))
    #
    # l5 = CustomModelMannger(mask_modloss=True)
    # l5.ms2_model.build(ModelMS2Bert,
    #                    num_frag_types=4,
    #                    dropout=0.1,
    #                    nlayers=5)
    #
    # l5.load_external_models(ms2_model_file="MSNet_Epoch100_ratio_full_layer5.pt")
    # print("l5 parameters: {0}".format(sum(p.numel() for p in l5.ms2_model.model.parameters())))
    #
    # l3 = l3.predict_all(precursor_df=inputs, predict_items=["ms2"],
    #                     frag_types=['b_z1', 'y_z1', 'b_z2', 'y_z2'])
    #
    l4 = l4.predict_all(precursor_df=inputs, predict_items=["ms2"],
                        frag_types=['b_z1', 'y_z1', 'b_z2', 'y_z2'])

    l2_pred = l2.predict_all(precursor_df=inputs, predict_items=["ms2"],
                             frag_types=['b_z1', 'y_z1', 'b_z2', 'y_z2'])
    #
    # l3_128 = l3_128.predict_all(precursor_df=inputs, predict_items=["ms2"],
    #                             frag_types=['b_z1', 'y_z1', 'b_z2', 'y_z2'])
    #
    # l5 = l5.predict_all(precursor_df=inputs, predict_items=["ms2"],
    #                     frag_types=['b_z1', 'y_z1', 'b_z2', 'y_z2'])

    l3_pred = l3.predict_all(precursor_df=inputs, predict_items=["ms2"],
                             frag_types=['b_z1', 'y_z1', 'b_z2', 'y_z2'])

    # l_full_pred = l_full.predict_all(precursor_df=inputs, predict_items=["ms2"],
    #                                  frag_types=['b_z1', 'y_z1', 'b_z2', 'y_z2'])

    # l1_pred = l1.predict_all(precursor_df=inputs, predict_items=["ms2"],
    #                          frag_types=['b_z1', 'y_z1', 'b_z2', 'y_z2'])
    # #
    # l8_pred = l8.predict_all(precursor_df=inputs, predict_items=["ms2"],
    #                          frag_types=['b_z1', 'y_z1', 'b_z2', 'y_z2'])

    return l1["fragment_intensity_df"], l2_pred["fragment_intensity_df"], l3_pred["fragment_intensity_df"], l4[
        "fragment_intensity_df"], l4_512["fragment_intensity_df"]


def make_prediction_1(inputs):
    default = M40.predict_all(precursor_df=inputs, predict_items=["ms2"],
                              frag_types=['b_z1', 'b_z2', 'y_z1', 'y_z2'])["fragment_intensity_df"]
    return inputs, default


def make_prediction_2(inputs):
    default = M20.predict_all(precursor_df=inputs, predict_items=["ms2"],
                              frag_types=['b_z1', 'b_z2', 'y_z1', 'y_z2'])["fragment_intensity_df"]
    return inputs, default


def make_prediction_3(inputs):
    default = M10.predict_all(precursor_df=inputs, predict_items=["ms2"],
                              frag_types=['b_z1', 'b_z2', 'y_z1', 'y_z2'])["fragment_intensity_df"]
    return inputs, default


def make_prediction_4(inputs):
    default = M5.predict_all(precursor_df=inputs, predict_items=["ms2"],
                             frag_types=['b_z1', 'b_z2', 'y_z1', 'y_z2'])["fragment_intensity_df"]
    return inputs, default


def make_prediction_5(inputs):
    default = M2.predict_all(precursor_df=inputs, predict_items=["ms2"],
                             frag_types=['b_z1', 'b_z2', 'y_z1', 'y_z2'])["fragment_intensity_df"]
    return inputs, default


def run_on(inputs, fragment_intensity_df):
    train_data = pd.read_csv("tests/train_data_precursor_v2.csv")
    inputs["mods"] = inputs["mods"].fillna("")
    inputs["mod_sites"] = inputs["mod_sites"].fillna("")

    print(inputs["instrument"].unique())
    print(inputs["nce"].unique())
    inputs.rename(columns={"frag_start_idx": "truth_frag_start_idx", "frag_stop_idx": "truth_frag_stop_idx"},
                  inplace=True)

    m40_inputs, default = make_prediction_1(inputs)
    m40_inputs = m40_inputs[-(m40_inputs["sequence"].isin(train_data["sequence"].tolist())) & (
        m40_inputs["charge"].isin(train_data["charge"].tolist()))]
    pandarallel.initialize(nb_workers=4)
    m40_inputs["loss"] = m40_inputs.parallel_apply(
        lambda row: cal_loss(fragment_intensity_df[row["truth_frag_start_idx"]:row["truth_frag_stop_idx"]][
                                 ['b_z1', 'b_z2', 'y_z1', 'y_z2']].values,
                             default[row["frag_start_idx"]:row["frag_stop_idx"]][
                                 ['b_z1', 'b_z2', 'y_z1', 'y_z2']].values), axis=1)

    m40 = m40_inputs["loss"].sum()

    m20_inputs, default = make_prediction_2(inputs)
    m20_inputs = m20_inputs[-(m20_inputs["sequence"].isin(train_data["sequence"].tolist())) & (
        m20_inputs["charge"].isin(train_data["charge"].tolist()))]
    pandarallel.initialize(nb_workers=4)
    m20_inputs["loss"] = m20_inputs.parallel_apply(
        lambda row: cal_loss(fragment_intensity_df[row["truth_frag_start_idx"]:row["truth_frag_stop_idx"]][
                                 ['b_z1', 'b_z2', 'y_z1', 'y_z2']].values,
                             default[row["frag_start_idx"]:row["frag_stop_idx"]][
                                 ['b_z1', 'b_z2', 'y_z1', 'y_z2']].values), axis=1)
    m20 = m20_inputs["loss"].sum()

    m10_inputs, default = make_prediction_3(inputs)
    m10_inputs = m10_inputs[-(m10_inputs["sequence"].isin(train_data["sequence"].tolist())) & (
        m10_inputs["charge"].isin(train_data["charge"].tolist()))]
    pandarallel.initialize(nb_workers=4)
    m10_inputs["loss"] = m10_inputs.parallel_apply(
        lambda row: cal_loss(fragment_intensity_df[row["truth_frag_start_idx"]:row["truth_frag_stop_idx"]][
                                 ['b_z1', 'b_z2', 'y_z1', 'y_z2']].values,
                             default[row["frag_start_idx"]:row["frag_stop_idx"]][
                                 ['b_z1', 'b_z2', 'y_z1', 'y_z2']].values), axis=1)
    m10 = m10_inputs["loss"].sum()

    m5_inputs, default = make_prediction_4(inputs)
    m5_inputs = m5_inputs[-(m5_inputs["sequence"].isin(train_data["sequence"].tolist())) & (
        m5_inputs["charge"].isin(train_data["charge"].tolist()))]
    pandarallel.initialize(nb_workers=4)
    m5_inputs["loss"] = m5_inputs.parallel_apply(
        lambda row: cal_loss(fragment_intensity_df[row["truth_frag_start_idx"]:row["truth_frag_stop_idx"]][
                                 ['b_z1', 'b_z2', 'y_z1', 'y_z2']].values,
                             default[row["frag_start_idx"]:row["frag_stop_idx"]][
                                 ['b_z1', 'b_z2', 'y_z1', 'y_z2']].values), axis=1)
    m5 = m5_inputs["loss"].sum()

    m2_inputs, default = make_prediction_5(inputs)
    m2_inputs = m2_inputs[-(m2_inputs["sequence"].isin(train_data["sequence"].tolist())) & (
        m2_inputs["charge"].isin(train_data["charge"].tolist()))]
    pandarallel.initialize(nb_workers=4)
    m2_inputs["loss"] = m2_inputs.parallel_apply(
        lambda row: cal_loss(fragment_intensity_df[row["truth_frag_start_idx"]:row["truth_frag_stop_idx"]][
                                 ['b_z1', 'b_z2', 'y_z1', 'y_z2']].values,
                             default[row["frag_start_idx"]:row["frag_stop_idx"]][
                                 ['b_z1', 'b_z2', 'y_z1', 'y_z2']].values), axis=1)

    m2 = m2_inputs["loss"].sum()
    return [m40, m20, m10, m5, m2], [m40_inputs.shape[0], m20_inputs.shape[0], m10_inputs.shape[0], m5_inputs.shape[0],
                                     m2_inputs.shape[0]]


def plot_scaling_laws():

    plt.figure(figsize=(10, 6), dpi=400)
    dataset_sizes = [19836096 * 2, 19836096, 9970378, 4974816, 2479829]
    test_losses = [8784.8994 / 322922, 9390.9629 / 322922, 9913.9004 / 322922, 12515.1523 / 322922, 13954.5957 / 322922]

    plt.plot(dataset_sizes, test_losses, linestyle='-', color='#EEC186', label='Test Loss')
    marker_sizes = [y * 2000 for y in test_losses]

    plt.scatter(dataset_sizes, test_losses, s=marker_sizes, color='#EEC186', edgecolors='k', alpha=0.8)

    log_x = np.log(dataset_sizes)
    log_y = np.log(test_losses)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    a, b = slope, np.exp(intercept)

    x_fit = np.logspace(np.log10(min(dataset_sizes)), np.log10(max(dataset_sizes)), 100)
    y_fit = b * x_fit ** a
    plt.plot(x_fit, y_fit, color="#808080", label=fr'$L = {b:.2f} \cdot D^{{{a:.2f}}}$')


    plt.title("Test Loss vs. Dataset Size", fontsize=16)
    plt.xlabel("Dataset Size", fontsize=16)
    plt.ylabel("Test L1 Loss ", fontsize=16)

    plt.xscale("log")
    plt.yscale("log")

    plt.xticks([5000000, 10000000, 40000000], [r"$5*10^6$", r"$10^7$", r"$4*10^7$"], fontsize=16)
    plt.text(x=2479829 + 200, y=13954.5957 / 322922, s="2M PSMs/0.15M Precursors", fontsize=13)
    plt.text(x=4974816 + 200, y=12515.1523 / 322922, s="5M PSMs/0.25M Precursors", fontsize=13)
    plt.text(x=9970378 + 200, y=9913.9004 / 322922, s="10M PSMs/0.5M Precursors", fontsize=13)
    plt.text(x=19836096 - 200, y=9390.9629 / 322922, s="20M PSMs/1M Precursors", fontsize=13)
    plt.text(x=19000000, y=8784.8994 / 322922, s="40M PSMs/2M Precursors", fontsize=13)

    plt.yticks([0.03, 0.04], [r"$3*10^{-2}$", r"$4*10^{-2}$"], fontsize=16)

    plt.legend(fontsize=13)

    plt.savefig("scaling_law_test_loss_v3.svg", dpi=500, bbox_inches='tight')

    test_losses = [0.37725247, 0.28512237, 0.26547013, 0.25183565, 0.20145988]
    parameter_size = [210089, 813225, 1602985, 2392745, 3182505]
    plt.figure(figsize=(10, 6), dpi=400)

    plt.plot(parameter_size, test_losses, linestyle='-', color='#EEC186', label='Test Loss')
    marker_sizes = [y * 500 for y in test_losses]

    plt.scatter(parameter_size, test_losses, s=marker_sizes, color='#EEC186', edgecolors='k', alpha=0.8)

    log_x = np.log(parameter_size)
    log_y = np.log(test_losses)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    a, b = slope, np.exp(intercept)
    x_fit = np.logspace(np.log10(min(parameter_size)), np.log10(max(parameter_size)), 100)
    y_fit = b * x_fit ** a
    plt.plot(x_fit, y_fit, color="#808080", label=fr'$L = {b:.2f} \cdot N^{{{a:.2f}}}$')


    plt.title("Test Loss vs. Parameter Size", fontsize=16)
    plt.xlabel("Parameter Size", fontsize=16)
    plt.ylabel("Test L1 Loss ", fontsize=16)

    plt.xscale("log") 
    plt.yscale("log") 

    plt.xticks([200000, 1000000, 3000000], [r"$2*10^5$", r"$10^6$", r"$3*10^6$"], fontsize=16)

    plt.yticks([0.2, 0.3], [r"$2*10^{-1}$", r"$3*10^{-1}$"], fontsize=16)

    plt.legend(fontsize=13)

    plt.savefig("scaling_law_test_loss_parameters_v3.svg", dpi=500, bbox_inches='tight')


# plot_scaling_laws()

if __name__ == "__main__":
    plot_scaling_laws()
