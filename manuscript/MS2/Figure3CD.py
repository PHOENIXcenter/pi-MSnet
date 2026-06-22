import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from pandarallel import pandarallel  # 导入pandaralle
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
from peptdeep.settings import global_settings as settings, model_const
import glob

frag_types = settings["model"]["frag_types"]
max_frag_charge = settings["model"]["max_frag_charge"]
num_ion_types = len(frag_types) * max_frag_charge


class CustomMS2Model(pDeepModel):
    def __init__(self, dropout=0.1,
                 mask_modloss=True,
                 modloss_type="modloss",
                 model_class: torch.nn.Module = ModelMS2Bert,
                 device: str = "gpu", **kwargs):
        super().__init__(device=device)
        self.charged_frag_types = ['b_z1', 'b_z2', 'y_z1', 'y_z2']
        self._get_modloss_frags(modloss_type)

        self.charge_factor = 0.1
        self.NCE_factor = 0.01
        self.model: ModelMS2Bert = None
        self.build(
            model_class,
            num_frag_types=len(self.charged_frag_types),
            num_modloss_types=len(self._modloss_frag_types),
            mask_modloss=mask_modloss,
            dropout=dropout,
            **kwargs,  # other model params
        )

        self.loss_func = torch.nn.L1Loss()
        self.min_inten = 1e-4

    def _set_batch_predict_data(self, batch_df: pd.DataFrame,
                                predicts: np.ndarray,
                                **kwargs):
        apex_intens = predicts.reshape((len(batch_df), -1)).max(axis=1)
        apex_intens[apex_intens <= 0] = 1
        predicts /= apex_intens.reshape((-1, 1, 1))
        predicts[predicts < self.min_inten] = 0.0
        if self._predict_in_order:
            self.predict_df.values[
            batch_df.frag_start_idx.values[0]: batch_df.frag_stop_idx.values[-1], :
            ] = predicts.reshape((-1, len(self.charged_frag_types)))
        else:
            update_sliced_fragment_dataframe(
                self.predict_df,
                self.predict_df.to_numpy(copy=True),
                predicts.reshape((-1, len(self.charged_frag_types))),
                batch_df[["frag_start_idx", "frag_stop_idx"]].values,
            )


class CustomModelMannger(ModelManager):
    def __init__(self, mask_modloss: bool = False,
                 device: str = "gpu"):
        super().__init__()
        self.ms2_model = CustomMS2Model(
            mask_modloss=mask_modloss, device=device
        )


def l2_normalize(vec):
    # 计算 L2 范数
    norm = np.linalg.norm(vec)
    # 避免除以 0 的情况
    if norm == 0:
        return vec
    return vec / norm


def plot_pcc():
    pic_data = pd.DataFrame()

    PXD000561_AlphaPeptDeep = np.load("./PXD000561_AlphaPeptDeep.npy")
    PXD000561_AlphaPeptDeep = np.array(PXD000561_AlphaPeptDeep)
    PXD000561_AlphaPeptDeep = PXD000561_AlphaPeptDeep[~np.isnan(PXD000561_AlphaPeptDeep)]
    tmp = pd.DataFrame({"PCC": PXD000561_AlphaPeptDeep})
    tmp["Model"] = "AlphaPeptDeep"
    tmp["datasets"] = "Orbitrap Elite/HCD@32"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(PXD000561_AlphaPeptDeep[PXD000561_AlphaPeptDeep > 0.60]) / len(PXD000561_AlphaPeptDeep))
    print(np.nanmedian(PXD000561_AlphaPeptDeep))  # 0.87
    PXD000561_Prosit = np.load("./PXD000561_Prosit.npy")
    PXD000561_Prosit = np.array(PXD000561_Prosit)
    PXD000561_Prosit = PXD000561_Prosit[~np.isnan(PXD000561_Prosit)]
    tmp = pd.DataFrame({"PCC": PXD000561_Prosit})
    tmp["Model"] = "Prosit"
    tmp["datasets"] = "Orbitrap Elite/HCD@32"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(PXD000561_Prosit[PXD000561_Prosit > 0.60]) / len(PXD000561_Prosit))
    print(np.nanmedian(PXD000561_Prosit))  # 0.83
    PXD000561_Unispec = np.load("./PXD000561_Unispec.npy")
    PXD000561_Unispec = np.array(PXD000561_Unispec)
    PXD000561_Unispec = PXD000561_Unispec[~np.isnan(PXD000561_Unispec)]
    tmp = pd.DataFrame({"PCC": PXD000561_Unispec})
    tmp["Model"] = "Unispec"
    tmp["datasets"] = "Orbitrap Elite/HCD@32"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(PXD000561_Unispec[PXD000561_Unispec > 0.60]) / len(PXD000561_Unispec))
    print(np.nanmedian(PXD000561_Unispec))  # 0.88

    PXD000561_experiment_pccs = np.load("PXD000561_experiment_pccs.npy")
    tmp = pd.DataFrame({"PCC": PXD000561_experiment_pccs})
    tmp["Model"] = "Experiment"
    tmp["datasets"] = "Orbitrap Elite/HCD@32"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(np.nanmedian(np.array(PXD000561_experiment_pccs)))  # 0.96

    PXD002395_AlphaPeptDeep = np.load("./PXD002395_AlphaPeptDeep.npy")
    PXD002395_AlphaPeptDeep = np.array(PXD002395_AlphaPeptDeep)
    PXD002395_AlphaPeptDeep = PXD002395_AlphaPeptDeep[~np.isnan(PXD002395_AlphaPeptDeep)]
    tmp = pd.DataFrame({"PCC": PXD002395_AlphaPeptDeep})
    tmp["Model"] = "AlphaPeptDeep"
    tmp["datasets"] = "Orbitrap Velos/HCD@40"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(PXD002395_AlphaPeptDeep[PXD002395_AlphaPeptDeep > 0.60]) / len(PXD002395_AlphaPeptDeep))
    print(np.nanmedian(PXD002395_AlphaPeptDeep))  # 0.89
    PXD002395_Prosit = np.load("./PXD002395_Prosit.npy")
    PXD002395_Prosit = np.array(PXD002395_Prosit)
    PXD002395_Prosit = PXD002395_Prosit[~np.isnan(PXD002395_Prosit)]
    tmp = pd.DataFrame({"PCC": PXD002395_Prosit})
    tmp["Model"] = "Prosit"
    tmp["datasets"] = "Orbitrap Velos/HCD@40"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(PXD002395_Prosit[PXD002395_Prosit > 0.60]) / len(PXD002395_Prosit))
    print(np.nanmedian(PXD002395_Prosit))  # 0.76
    PXD002395_Unispec = np.load("./PXD002395_Unispec.npy")
    PXD002395_Unispec = np.array(PXD002395_Unispec)
    PXD002395_Unispec = PXD002395_Unispec[~np.isnan(PXD002395_Unispec)]
    tmp = pd.DataFrame({"PCC": PXD002395_Unispec})
    tmp["Model"] = "Unispec"
    tmp["datasets"] = "Orbitrap Velos/HCD@40"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(PXD002395_Unispec[PXD002395_Unispec > 0.60]) / len(PXD002395_Unispec))
    print(np.nanmedian(PXD002395_Unispec))  # 0.92

    experiment_pccs = np.load("PXD002395_ground_truth_pcc.npy")
    tmp = pd.DataFrame({"PCC": experiment_pccs})
    tmp["Model"] = "Experiment"
    tmp["datasets"] = "Orbitrap Velos/HCD@40"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)
    print(np.nanmedian(np.array(experiment_pccs)))  # 0.90

    huiyan_AlphaPeptDeep = np.load("./huiyan_AlphaPeptDeep.npy")
    huiyan_AlphaPeptDeep = np.array(huiyan_AlphaPeptDeep)
    huiyan_AlphaPeptDeep = huiyan_AlphaPeptDeep[~np.isnan(huiyan_AlphaPeptDeep)]
    tmp = pd.DataFrame({"PCC": huiyan_AlphaPeptDeep})
    tmp["Model"] = "AlphaPeptDeep"
    tmp["datasets"] = "Exploris480/HCD@28"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(huiyan_AlphaPeptDeep[huiyan_AlphaPeptDeep > 0.60]) / len(huiyan_AlphaPeptDeep))
    print(np.nanmedian(huiyan_AlphaPeptDeep))  # 0.94
    huiyan_Prosit = np.load("./huiyan_Prosit.npy")
    huiyan_Prosit = np.array(huiyan_Prosit)
    huiyan_Prosit = huiyan_Prosit[~np.isnan(huiyan_Prosit)]
    tmp = pd.DataFrame({"PCC": huiyan_Prosit})
    tmp["Model"] = "Prosit"
    tmp["datasets"] = "Exploris480/HCD@28"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(huiyan_Prosit[huiyan_Prosit > 0.60]) / len(huiyan_Prosit))
    print(np.nanmedian(huiyan_Prosit))  # 0.94
    huiyan_Unispec = np.load("./huiyan_Unispec.npy")
    huiyan_Unispec = np.array(huiyan_Unispec)
    huiyan_Unispec = huiyan_Unispec[~np.isnan(huiyan_Unispec)]
    tmp = pd.DataFrame({"PCC": huiyan_Unispec})
    tmp["Model"] = "Unispec"
    tmp["datasets"] = "Exploris480/HCD@28"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)
    print(len(huiyan_Unispec[huiyan_Unispec > 0.60]) / len(huiyan_Unispec))
    print(np.nanmedian(huiyan_Unispec))  # 0.92

    experiment_pccs = np.load("huiyan_ground_truth_pcc.npy")
    print(np.nanmedian(experiment_pccs))

    tmp = pd.DataFrame({"PCC": experiment_pccs})
    tmp["Model"] = "Experiment"
    tmp["datasets"] = "Exploris480/HCD@28"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)
    print(np.nanmedian(np.array(experiment_pccs)))  # 0.57

    # PXD012636
    PXD012636_AlphaPeptDeep = np.load("./PXD012636_AlphaPeptDeep.npy")
    PXD012636_AlphaPeptDeep = np.array(PXD012636_AlphaPeptDeep)
    PXD012636_AlphaPeptDeep = PXD012636_AlphaPeptDeep[~np.isnan(PXD012636_AlphaPeptDeep)]
    tmp = pd.DataFrame({"PCC": PXD012636_AlphaPeptDeep})
    tmp["Model"] = "AlphaPeptDeep"
    tmp["datasets"] = "Q Exactive/HCD@28"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(PXD012636_AlphaPeptDeep[PXD012636_AlphaPeptDeep > 0.60]) / len(PXD012636_AlphaPeptDeep))
    print(np.nanmedian(PXD012636_AlphaPeptDeep))  # 0.97
    PXD012636_Prosit = np.load("./PXD012636_Prosit.npy")
    PXD012636_Prosit = np.array(PXD012636_Prosit)
    PXD012636_Prosit = PXD012636_Prosit[~np.isnan(PXD012636_Prosit)]
    tmp = pd.DataFrame({"PCC": PXD012636_Prosit})
    tmp["Model"] = "Prosit"
    tmp["datasets"] = "Q Exactive/HCD@28"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(PXD012636_Prosit[PXD012636_Prosit > 0.60]) / len(PXD012636_Prosit))
    print(np.nanmedian(PXD012636_Prosit))  # 0.88
    PXD012636_Unispec = np.load("./PXD012636_Unispec.npy")
    PXD012636_Unispec = np.array(PXD012636_Unispec)
    PXD012636_Unispec = PXD012636_Unispec[~np.isnan(PXD012636_Unispec)]
    tmp = pd.DataFrame({"PCC": PXD012636_Unispec})
    tmp["Model"] = "Unispec"
    tmp["datasets"] = "Q Exactive/HCD@28"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(PXD012636_Unispec[PXD012636_Unispec > 0.60]) / len(PXD012636_Unispec))
    print(np.nanmedian(PXD012636_Unispec))  # 0.90

    experiment_pccs = np.load("PXD012636_ground_truth_pcc.npy")
    print(len(experiment_pccs))
    tmp = pd.DataFrame({"PCC": experiment_pccs})
    tmp["Model"] = "Experiment"
    tmp["datasets"] = "Q Exactive/HCD@28"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(np.nanmedian(np.array(experiment_pccs)))  # 0.99

    # PXD009737  Gluc 也有重复的
    PXD009737gluc_AlphaPeptDeep = np.load("./PXD009737gluc_AlphaPeptDeep.npy")
    PXD009737gluc_AlphaPeptDeep = np.array(PXD009737gluc_AlphaPeptDeep)
    PXD009737gluc_AlphaPeptDeep = PXD009737gluc_AlphaPeptDeep[~np.isnan(PXD009737gluc_AlphaPeptDeep)]
    tmp = pd.DataFrame({"PCC": PXD009737gluc_AlphaPeptDeep})
    tmp["Model"] = "AlphaPeptDeep"
    tmp["datasets"] = "Gluc/Q Exactive HF-X/HCD@27"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(PXD009737gluc_AlphaPeptDeep[PXD009737gluc_AlphaPeptDeep > 0.60]) / len(PXD009737gluc_AlphaPeptDeep))
    print(np.nanmedian(PXD009737gluc_AlphaPeptDeep))  # 0.85
    PXD009737gluc_Prosit = np.load("./PXD009737gluc_Prosit.npy")
    PXD009737gluc_Prosit = np.array(PXD009737gluc_Prosit)
    PXD009737gluc_Prosit = PXD009737gluc_Prosit[~np.isnan(PXD009737gluc_Prosit)]
    tmp = pd.DataFrame({"PCC": PXD009737gluc_Prosit})
    tmp["Model"] = "Prosit"
    tmp["datasets"] = "Gluc/Q Exactive HF-X/HCD@27"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(PXD009737gluc_Prosit[PXD009737gluc_Prosit > 0.60]) / len(PXD009737gluc_Prosit))
    print(np.nanmedian(PXD009737gluc_Prosit))  # 0.78
    PXD009737gluc_Unispec = np.load("./PXD009737gluc_Unispec.npy")
    PXD009737gluc_Unispec = np.array(PXD009737gluc_Unispec)
    PXD009737gluc_Unispec = PXD009737gluc_Unispec[~np.isnan(PXD009737gluc_Unispec)]
    tmp = pd.DataFrame({"PCC": PXD009737gluc_Unispec})
    tmp["Model"] = "Unispec"
    tmp["datasets"] = "Gluc/Q Exactive HF-X/HCD@27"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(PXD009737gluc_Unispec[PXD009737gluc_Unispec > 0.60]) / len(PXD009737gluc_Unispec))
    print(np.nanmedian(PXD009737gluc_Unispec))  # 0.79

    experiment_pccs = np.load("PXD009737gluc_ground_truth_pcc.npy")
    tmp = pd.DataFrame({"PCC": experiment_pccs})
    tmp["Model"] = "Experiment"
    tmp["datasets"] = "Gluc/Q Exactive HF-X/HCD@27"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)
    print(np.nanmedian(np.array(experiment_pccs)))  # 0.98

    # PXD009737  Lysc  也有重复
    PXD009737lysc_AlphaPeptDeep = np.load("./PXD009737lysc_AlphaPeptDeep.npy")
    PXD009737lysc_AlphaPeptDeep = np.array(PXD009737lysc_AlphaPeptDeep)
    PXD009737lysc_AlphaPeptDeep = PXD009737lysc_AlphaPeptDeep[~np.isnan(PXD009737lysc_AlphaPeptDeep)]
    tmp = pd.DataFrame({"PCC": PXD009737lysc_AlphaPeptDeep})
    tmp["Model"] = "AlphaPeptDeep"
    tmp["datasets"] = "Lycs/Q Exactive HF-X/HCD@27"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(PXD009737lysc_AlphaPeptDeep[PXD009737lysc_AlphaPeptDeep > 0.60]) / len(PXD009737lysc_AlphaPeptDeep))
    print(np.nanmedian(PXD009737lysc_AlphaPeptDeep))  # 0.86
    PXD009737lysc_Prosit = np.load("./PXD009737lysc_Prosit.npy")
    PXD009737lysc_Prosit = np.array(PXD009737lysc_Prosit)
    PXD009737lysc_Prosit = PXD009737lysc_Prosit[~np.isnan(PXD009737lysc_Prosit)]
    tmp = pd.DataFrame({"PCC": PXD009737lysc_Prosit})
    tmp["Model"] = "Prosit"
    tmp["datasets"] = "Lycs/Q Exactive HF-X/HCD@27"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(PXD009737lysc_Prosit[PXD009737lysc_Prosit > 0.60]) / len(PXD009737lysc_Prosit))
    print(np.nanmedian(PXD009737lysc_Prosit))  # 0.78
    PXD009737lysc_Unispec = np.load("./PXD009737lysc_Unispec.npy")
    PXD009737lysc_Unispec = np.array(PXD009737lysc_Unispec)
    PXD009737lysc_Unispec = PXD009737lysc_Unispec[~np.isnan(PXD009737lysc_Unispec)]
    tmp = pd.DataFrame({"PCC": PXD009737lysc_Unispec})
    tmp["Model"] = "Unispec"
    tmp["datasets"] = "Lycs/Q Exactive HF-X/HCD@27"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)

    print(len(PXD009737lysc_Unispec[PXD009737lysc_Unispec > 0.60]) / len(PXD009737lysc_Unispec))
    print(np.nanmedian(PXD009737lysc_Unispec))  # 0.84

    experiment_pccs = np.load("PXD009737lysc_ground_truth_pcc.npy")
    tmp = pd.DataFrame({"PCC": experiment_pccs})
    tmp["Model"] = "Experiment"
    tmp["datasets"] = "Lycs/Q Exactive HF-X/HCD@27"
    pic_data = pd.concat([pic_data, tmp], ignore_index=True)
    print(np.nanmedian(np.array(experiment_pccs)))  # 0.90
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 6), sharey=True)

    i = 0
    fs = 6
    colors = ['#0E606B', '#1597A5', '#FFF4F2', '#FEB3AE']
    pic_data = pic_data[-pic_data["PCC"] <= 0]
    for k, group in pic_data.groupby("datasets"):
        if i <= 2:
            d1 = group[group["Model"] == "Experiment"]["PCC"].tolist()
            d2 = group[group["Model"] == "AlphaPeptDeep"]["PCC"].tolist()
            d3 = group[group["Model"] == "Prosit"]["PCC"].tolist()
            d4 = group[group["Model"] == "Unispec"]["PCC"].tolist()
            min_len = min(len(d1), len(d2), len(d3), len(d4))
            d = np.array([d1[:min_len],
                          d2[:min_len],
                          d3[:min_len],
                          d4[:min_len]]).T
            print(d.shape)
            bplot = axs[0, i].boxplot(d, tick_labels=["Experiment", "AlphaPeptDeep", "Prosit", "Unispec"],
                                      showfliers=False,
                                      patch_artist=True)
            axs[0, i].set_title(k, fontsize=fs)
            axs[0, i].tick_params(labelsize=6)
            # fill with colors
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            i += 1
        else:
            d1 = group[group["Model"] == "Experiment"]["PCC"].tolist()
            d2 = group[group["Model"] == "AlphaPeptDeep"]["PCC"].tolist()
            d3 = group[group["Model"] == "Prosit"]["PCC"].tolist()
            d4 = group[group["Model"] == "Unispec"]["PCC"].tolist()
            min_len = min(len(d1), len(d2), len(d3), len(d4))
            d = np.array([d1[:min_len],
                          d2[:min_len],
                          d3[:min_len],
                          d4[:min_len]]).T
            bplot = axs[1, i - 3].boxplot(d, tick_labels=["Experiment", "AlphaPeptDeep", "Prosit", "Unispec"],
                                          showfliers=False,
                                          patch_artist=True)
            # fill with colors
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

            axs[1, i - 3].set_title(k, fontsize=fs)
            axs[1, i - 3].tick_params(labelsize=6)
            i += 1
    fig.subplots_adjust(hspace=0.4)
    plt.ylim(0, 1)
    plt.xticks(fontsize=6)
    plt.show()
    #     sns.boxplot(data=group, x="Condition", hue="exp", y="CV[%]", ax=ax[i], fliersize=1, palette=custom_palette(),
    #                 hue_order=["percolator_comet", "percolator_msgf_comet",
    #                            "ms2rescore_msgf_comet", "ms2rescore_msgf_comet_sage",
    #                            "MaxQuant"])
    #     # ax[i].get_legend().remove()
    #     # ax[i].set_ylim(-4,6)
    #     ax[i].set_title(k)
    #     ax[i].set_ylabel("CV[%]")
    #     ax[i].spines['top'].set_visible(False)
    #     ax[i].spines['right'].set_visible(False)
    #     # handles, labels = ax[i].get_legend_handles_labels()
    #     # ax[i].get_legend().remove()
    #     i += 1
    #
    # # handles, labels = ax.get_legend_handles_labels()
    # # plt.legend(handles=handles[0:], labels=labels[0:], title="", frameon=False)
    # ax[1].set_xlabel("Concentration")
    # ax[0].set_xlabel(None)
    # ax[1].get_legend().remove()
    # ax[0].legend(loc="upper right", frameon=False)
    # # plt.legend()
    plt.savefig("PCCplot.png", bbox_inches='tight', dpi=500)


def rechanged_annotation(annotation):
    return annotation[0] + "_z" + annotation[-1], int(annotation.replace("b", "").replace("y", "").split("+")[0])


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

    # l3_128 = CustomModelMannger(mask_modloss=True)
    # l3_128.ms2_model.build(ModelMS2Bert,
    #                        num_frag_types=4,
    #                        dropout=0.1,
    #                        nlayers=3,
    #                        hidden=128)
    #
    # l3_128.load_external_models(ms2_model_file="MSNet_Epoch100_ratio_full_layer3_128.pt")
    # print("p3_128 parameters: {0}".format(sum(p.numel() for p in l3_128.ms2_model.model.parameters())))
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


loss_fuc = torch.nn.L1Loss()


def cal_loss(target, pred):
    target = target / target.max().max()
    target = target.values
    target = torch.tensor(target, dtype=torch.float32).view(-1, pred.shape[0], 4)
    pred = torch.tensor(pred, dtype=torch.float32).view(-1, pred.shape[0], 4)
    test_loss = loss_fuc(pred, target)
    return test_loss


def test_ground_truth():
    inputs = pd.read_csv("PXD009737gluc_inputs.csv")
    inputs.rename(columns={"peptide_sequences": "sequence", "precursor_charges": "charge",
                           "collision_energies": "nce", "instrument_types": "instrument"}, inplace=True)
    inputs["mods"] = ""
    inputs["mod_sites"] = ""
    inputs["nAA"] = inputs["sequence"].apply(len)
    t = inputs["nAA"].apply(lambda x: x - 1)
    end = list(itertools.accumulate(t.values.tolist()))
    inputs['frag_stop_idx'] = end
    inputs['frag_start_idx'] = [0] + end[:-1]
    predict_1, predict_2, predict_3, predict_4, predict_5 = make_prediction_2(inputs)

    ground_truth = pd.read_csv("PXD009737gluc_ground_truth_with_zero.csv", usecols=["PSM_ID", "peptide_sequences",
                                                                                    "precursor_charges",
                                                                                    "exp_intensities",
                                                                                    "annotation"])
    print(ground_truth.shape)
    ground_truth = ground_truth[
        -(ground_truth["annotation"].str.contains("-NH3") | ground_truth["annotation"].str.contains("-H2O"))]
    print(ground_truth.shape)
    ground_truth = ground_truth.drop_duplicates(subset=["PSM_ID", "annotation"], keep='first')
    ground_truth = ground_truth.sort_values("PSM_ID")
    ground_truth = ground_truth.drop_duplicates(subset=["peptide_sequences", "precursor_charges", "annotation"],
                                                keep="last")
    inputs = inputs[["sequence", "charge", "frag_start_idx", "frag_stop_idx"]]
    print(ground_truth.shape)

    pandarallel.initialize(nb_workers=10)
    ground_truth[["type", "slices"]] = ground_truth.parallel_apply(lambda x: rechanged_annotation(x["annotation"]),
                                                                   axis=1,
                                                                   result_type="expand")
    ground_truth = ground_truth.pivot(index=["peptide_sequences", "precursor_charges", "slices"], columns="type",
                                      values="exp_intensities").reset_index()
    ground_truth.sort_values(by=["peptide_sequences", "precursor_charges", "slices"])
    t = pd.merge(ground_truth, inputs, left_on=["peptide_sequences", "precursor_charges"],
                 right_on=["sequence", "charge"], how="left")
    print(ground_truth)
    pandarallel.initialize(nb_workers=10)
    # loss = t.groupby(["peptide_sequences", "precursor_charges"]).parallel_apply(
    #     lambda row: cal_loss(row[["b_z1", "b_z2", "y_z1", "y_z2"]],
    #                          l3_128.iloc[row["frag_start_idx"].iloc[0]:row["frag_stop_idx"].iloc[0], ].values))
    # print(sum(loss))

    loss = t.groupby(["peptide_sequences", "precursor_charges"]).parallel_apply(
        lambda row: cal_loss(row[["b_z1", "b_z2", "y_z1", "y_z2"]],
                             predict_1.iloc[row["frag_start_idx"].iloc[0]:row["frag_stop_idx"].iloc[0], ].values))
    print(sum(loss))

    loss = t.groupby(["peptide_sequences", "precursor_charges"]).parallel_apply(
        lambda row: cal_loss(row[["b_z1", "b_z2", "y_z1", "y_z2"]],
                             predict_2.iloc[row["frag_start_idx"].iloc[0]:row["frag_stop_idx"].iloc[0], ].values))
    print(sum(loss))

    loss = t.groupby(["peptide_sequences", "precursor_charges"]).parallel_apply(
        lambda row: cal_loss(row[["b_z1", "b_z2", "y_z1", "y_z2"]],
                             predict_3.iloc[row["frag_start_idx"].iloc[0]:row["frag_stop_idx"].iloc[0], ].values))
    print(sum(loss))

    loss = t.groupby(["peptide_sequences", "precursor_charges"]).parallel_apply(
        lambda row: cal_loss(row[["b_z1", "b_z2", "y_z1", "y_z2"]],
                             predict_4.iloc[row["frag_start_idx"].iloc[0]:row["frag_stop_idx"].iloc[0], ].values))
    print(sum(loss))

    loss = t.groupby(["peptide_sequences", "precursor_charges"]).parallel_apply(
        lambda row: cal_loss(row[["b_z1", "b_z2", "y_z1", "y_z2"]],
                             predict_5.iloc[row["frag_start_idx"].iloc[0]:row["frag_stop_idx"].iloc[0], ].values))
    print(sum(loss))

    # loss = t.groupby(["peptide_sequences", "precursor_charges"]).parallel_apply(
    #     lambda row: cal_loss(row[["b_z1", "b_z2", "y_z1", "y_z2"]],
    #                          predict_2.iloc[row["frag_start_idx"].iloc[0]:row["frag_stop_idx"].iloc[0], ].values))
    # print(sum(loss))

    # loss = t.groupby(["peptide_sequences", "precursor_charges"]).parallel_apply(
    #     lambda row: cal_loss(row[["b_z1", "b_z2", "y_z1", "y_z2"]],
    #                          predict_3.iloc[row["frag_start_idx"].iloc[0]:row["frag_stop_idx"].iloc[0], ].values))
    # print(sum(loss))

    # loss = t.groupby(["peptide_sequences", "precursor_charges"]).parallel_apply(
    #     lambda row: cal_loss(row[["b_z1", "b_z2", "y_z1", "y_z2"]],
    #                          predict_4.iloc[row["frag_start_idx"].iloc[0]:row["frag_stop_idx"].iloc[0], ].values))
    # print(sum(loss))

    # loss = t.groupby(["peptide_sequences", "precursor_charges"]).parallel_apply(
    #     lambda row: cal_loss(row[["b_z1", "b_z2", "y_z1", "y_z2"]],
    #                          predict_1.iloc[row["frag_start_idx"].iloc[0]:row["frag_stop_idx"].iloc[0], ].values))
    # print(sum(loss))

    # loss = t.groupby(["peptide_sequences", "precursor_charges"]).parallel_apply(
    #     lambda row: cal_loss(row[["b_z1", "b_z2", "y_z1", "y_z2"]],
    #                          predict_6.iloc[row["frag_start_idx"].iloc[0]:row["frag_stop_idx"].iloc[0], ].values))
    # print(sum(loss))
    #
    # loss = t.groupby(["peptide_sequences", "precursor_charges"]).parallel_apply(
    #     lambda row: cal_loss(row[["b_z1", "b_z2", "y_z1", "y_z2"]],
    #                          predict_8.iloc[row["frag_start_idx"].iloc[0]:row["frag_stop_idx"].iloc[0], ].values))
    # print(sum(loss))
    #
    # print(len(loss))


def plot_latest_pcc():
    all_psm = 0
    all_precursors = pd.DataFrame()
    all_unseen_psm = 0
    all_unseen_precursors = pd.DataFrame()
    pic_data = pd.DataFrame()
    unseen_pic_data = pd.DataFrame()
    train_data = pd.read_csv("tests/train_data_precursor_v2.csv")
    msnet = pd.read_csv("./tests/PXD012636_Danio_rerio_MSNet_PCC_latest.csv")
    apeptdeep = pd.read_csv("./tests/PXD012636_Danio_rerio_Alphapeptdeep.csv")
    prosit = pd.read_csv("./tests/PXD012636_Danio_rerio_Prosit.csv")
    unispec = pd.read_csv("./tests/PXD012636_Danio_rerio_unispec.csv")
    all_psm += msnet.shape[0]
    all_precursors = pd.concat([all_precursors, msnet[["sequence", "charge"]]])
    print("----------PXD012636---------")
    print(msnet.drop_duplicates(subset=["sequence", "charge"]).shape[0])
    desc = pd.Series(msnet["pcc"].describe())
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    desc = pd.Series(unispec["PCC"].describe())
    print(desc)
    print(unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0])
    pic_data = pic_data._append({"Model": "MSNet+Alphapeptdeep", "datasets": "Q Exactive HF/HCD@28",
                                 "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Alphapeptdeep", "datasets": "Q Exactive HF/HCD@28",
                                 "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0]},
                                ignore_index=True)
    pic_data = pic_data._append({"Model": "Prosit", "datasets": "Q Exactive HF/HCD@28",
                                 "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Unispec", "datasets": "Q Exactive HF/HCD@28",
                                 "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                ignore_index=True)
    msnet = msnet[-(msnet["sequence"].isin(train_data["sequence"].tolist())) & (
        msnet["charge"].isin(train_data["charge"].tolist()))]
    apeptdeep = apeptdeep[-(apeptdeep["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        apeptdeep["precursor_charges"].isin(train_data["charge"].tolist()))]
    prosit = prosit[-(prosit["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        prosit["precursor_charges"].isin(train_data["charge"].tolist()))]
    unispec = unispec[-(unispec["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        unispec["precursor_charges"].isin(train_data["charge"].tolist()))]
    all_unseen_psm += msnet.shape[0]
    all_unseen_precursors = pd.concat([all_unseen_precursors, msnet[["sequence", "charge"]]])
    desc = pd.Series(msnet["pcc"].describe())
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    desc = pd.Series(unispec["PCC"].describe())
    print(desc)
    print(unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0])
    unseen_pic_data = unseen_pic_data._append({"Model": "MSNet+Alphapeptdeep", "datasets": "Q Exactive HF/HCD@28",
                                               "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Alphapeptdeep", "datasets": "Q Exactive HF/HCD@28",
                                               "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[
                                                   0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Prosit", "datasets": "Q Exactive HF/HCD@28",
                                               "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Unispec", "datasets": "Q Exactive HF/HCD@28",
                                               "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                              ignore_index=True)

    msnet = pd.read_csv("./tests/IPX0004073001_MSNet_PCC_latest.csv")
    apeptdeep = pd.read_csv("./tests/IPX0004073001_Alphapeptdeep.csv")
    prosit = pd.read_csv("./tests/IPX0004073001_Prosit.csv")
    unispec = pd.read_csv("./tests/IPX0004073001_unispec.csv")
    all_psm += msnet.shape[0]
    all_precursors = pd.concat([all_precursors, msnet[["sequence", "charge"]]])
    desc = pd.Series(msnet["pcc"].describe())
    print("----------IPX0004073001---------")
    print(msnet.drop_duplicates(subset=["sequence", "charge"]).shape[0])
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    desc = pd.Series(unispec["PCC"].describe())
    print(desc)
    print(unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0])
    pic_data = pic_data._append({"Model": "MSNet+Alphapeptdeep", "datasets": "Orbitrap Fusion Lumos/HCD@30",
                                 "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Alphapeptdeep", "datasets": "Orbitrap Fusion Lumos/HCD@30",
                                 "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0]},
                                ignore_index=True)
    pic_data = pic_data._append({"Model": "Prosit", "datasets": "Orbitrap Fusion Lumos/HCD@30",
                                 "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Unispec", "datasets": "Orbitrap Fusion Lumos/HCD@30",
                                 "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                ignore_index=True)
    msnet = msnet[-(msnet["sequence"].isin(train_data["sequence"].tolist())) & (
        msnet["charge"].isin(train_data["charge"].tolist()))]
    apeptdeep = apeptdeep[-(apeptdeep["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        apeptdeep["precursor_charges"].isin(train_data["charge"].tolist()))]
    prosit = prosit[-(prosit["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        prosit["precursor_charges"].isin(train_data["charge"].tolist()))]
    unispec = unispec[-(unispec["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        unispec["precursor_charges"].isin(train_data["charge"].tolist()))]
    all_unseen_psm += msnet.shape[0]
    all_unseen_precursors = pd.concat([all_unseen_precursors, msnet[["sequence", "charge"]]])
    desc = pd.Series(msnet["pcc"].describe())
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    desc = pd.Series(unispec["PCC"].describe())
    print(desc)
    print(unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0])
    unseen_pic_data = unseen_pic_data._append(
        {"Model": "MSNet+Alphapeptdeep", "datasets": "Orbitrap Fusion Lumos/HCD@30",
         "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]},
        ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Alphapeptdeep", "datasets": "Orbitrap Fusion Lumos/HCD@30",
                                               "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[
                                                   0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Prosit", "datasets": "Orbitrap Fusion Lumos/HCD@30",
                                               "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Unispec", "datasets": "Orbitrap Fusion Lumos/HCD@30",
                                               "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                              ignore_index=True)

    msnet = pd.read_csv("./tests/PXD009737_lysc_msnet_MSNet_PCC_latest.csv")
    apeptdeep = pd.read_csv("./tests/PXD009737_lysc_msnet_Alphapeptdeep.csv")
    prosit = pd.read_csv("./tests/PXD009737_lysc_msnet_Prosit.csv")
    unispec = pd.read_csv("./tests/PXD009737_lysc_msnet_unispec.csv")
    all_psm += msnet.shape[0]
    all_precursors = pd.concat([all_precursors, msnet[["sequence", "charge"]]])
    desc = pd.Series(msnet["pcc"].describe())
    T = msnet
    print("----------PXD009737---------")
    print(T.drop_duplicates(subset=["sequence", "charge"]).shape[0])
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    pic_data = pic_data._append({"Model": "MSNet+Alphapeptdeep", "datasets": "Q Exactive HF-X/Lysc/HCD@30",
                                 "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Alphapeptdeep", "datasets": "Q Exactive HF-X/Lysc/HCD@30",
                                 "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0]},
                                ignore_index=True)
    pic_data = pic_data._append({"Model": "Prosit", "datasets": "Q Exactive HF-X/Lysc/HCD@30",
                                 "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Unispec", "datasets": "Q Exactive HF-X/Lysc/HCD@30",
                                 "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                ignore_index=True)
    msnet = msnet[-(msnet["sequence"].isin(train_data["sequence"].tolist())) & (
        msnet["charge"].isin(train_data["charge"].tolist()))]
    apeptdeep = apeptdeep[-(apeptdeep["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        apeptdeep["precursor_charges"].isin(train_data["charge"].tolist()))]
    prosit = prosit[-(prosit["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        prosit["precursor_charges"].isin(train_data["charge"].tolist()))]
    unispec = unispec[-(unispec["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        unispec["precursor_charges"].isin(train_data["charge"].tolist()))]
    all_unseen_psm += msnet.shape[0]
    all_unseen_precursors = pd.concat([all_unseen_precursors, msnet[["sequence", "charge"]]])
    desc = pd.Series(msnet["pcc"].describe())
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    desc = pd.Series(unispec["PCC"].describe())
    print(desc)
    print(unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0])
    unseen_pic_data = unseen_pic_data._append(
        {"Model": "MSNet+Alphapeptdeep", "datasets": "Q Exactive HF-X/Lysc/HCD@30",
         "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]},
        ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Alphapeptdeep", "datasets": "Q Exactive HF-X/Lysc/HCD@30",
                                               "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[
                                                   0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Prosit", "datasets": "Q Exactive HF-X/Lysc/HCD@30",
                                               "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Unispec", "datasets": "Q Exactive HF-X/Lysc/HCD@30",
                                               "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                              ignore_index=True)

    msnet = pd.read_csv("./tests/PXD009737_gluc_msnet_MSNet_PCC_latest.csv")
    apeptdeep = pd.read_csv("./tests/PXD009737_gluc_msnet_Alphapeptdeep.csv")
    prosit = pd.read_csv("./tests/PXD009737_gluc_msnet_Prosit.csv")
    unispec = pd.read_csv("./tests/PXD009737_gluc_msnet_unispec.csv")
    all_psm += msnet.shape[0]
    all_precursors = pd.concat([all_precursors, msnet[["sequence", "charge"]]])
    desc = pd.Series(msnet["pcc"].describe())
    print(pd.concat([T, msnet[["sequence", "charge"]]]).drop_duplicates(subset=["sequence", "charge"]).shape[0])
    print(msnet.drop_duplicates(subset=["sequence", "charge"]).shape[0])
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    desc = pd.Series(unispec["PCC"].describe())
    print(desc)
    print(unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0])
    pic_data = pic_data._append({"Model": "MSNet+Alphapeptdeep", "datasets": "Q Exactive HF-X/Gluc/HCD@30",
                                 "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Alphapeptdeep", "datasets": "Q Exactive HF-X/Gluc/HCD@30",
                                 "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0]},
                                ignore_index=True)
    pic_data = pic_data._append({"Model": "Prosit", "datasets": "Q Exactive HF-X/Gluc/HCD@30",
                                 "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Unispec", "datasets": "Q Exactive HF-X/Gluc/HCD@30",
                                 "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                ignore_index=True)
    msnet = msnet[-(msnet["sequence"].isin(train_data["sequence"].tolist())) & (
        msnet["charge"].isin(train_data["charge"].tolist()))]
    apeptdeep = apeptdeep[-(apeptdeep["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        apeptdeep["precursor_charges"].isin(train_data["charge"].tolist()))]
    prosit = prosit[-(prosit["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        prosit["precursor_charges"].isin(train_data["charge"].tolist()))]
    unispec = unispec[-(unispec["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        unispec["precursor_charges"].isin(train_data["charge"].tolist()))]
    all_unseen_psm += msnet.shape[0]
    all_unseen_precursors = pd.concat([all_unseen_precursors, msnet[["sequence", "charge"]]])
    desc = pd.Series(msnet["pcc"].describe())
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    desc = pd.Series(unispec["PCC"].describe())
    print(desc)
    print(unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0])
    unseen_pic_data = unseen_pic_data._append(
        {"Model": "MSNet+Alphapeptdeep", "datasets": "Q Exactive HF-X/Gluc/HCD@30",
         "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]},
        ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Alphapeptdeep", "datasets": "Q Exactive HF-X/Gluc/HCD@30",
                                               "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[
                                                   0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Prosit", "datasets": "Q Exactive HF-X/Gluc/HCD@30",
                                               "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Unispec", "datasets": "Q Exactive HF-X/Gluc/HCD@30",
                                               "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                              ignore_index=True)

    msnet = pd.read_csv("./tests/PXD019483_msnet_MSNet_PCC_latest.csv")
    apeptdeep = pd.read_csv("./tests/PXD019483_msnet_Alphapeptdeep.csv")
    prosit = pd.read_csv("./tests/PXD019483_msnet_Prosit.csv")
    unispec = pd.read_csv("./tests/PXD019483_msnet_unispec.csv")
    all_psm += msnet.shape[0]
    all_precursors = pd.concat([all_precursors, msnet[["sequence", "charge"]]])
    desc = pd.Series(msnet["pcc"].describe())
    print("------------PXD019483----------")
    print(msnet.drop_duplicates(subset=["sequence", "charge"]).shape[0])
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    desc = pd.Series(unispec["PCC"].describe())
    print(desc)
    print(unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0])
    pic_data = pic_data._append({"Model": "MSNet+Alphapeptdeep", "datasets": "Q Exactive HF-X/HCD@27",
                                 "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Alphapeptdeep", "datasets": "Q Exactive HF-X/HCD@27",
                                 "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0]},
                                ignore_index=True)
    pic_data = pic_data._append({"Model": "Prosit", "datasets": "Q Exactive HF-X/HCD@27",
                                 "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Unispec", "datasets": "Q Exactive HF-X/HCD@27",
                                 "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                ignore_index=True)
    msnet = msnet[-(msnet["sequence"].isin(train_data["sequence"].tolist())) & (
        msnet["charge"].isin(train_data["charge"].tolist()))]
    apeptdeep = apeptdeep[-(apeptdeep["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        apeptdeep["precursor_charges"].isin(train_data["charge"].tolist()))]
    prosit = prosit[-(prosit["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        prosit["precursor_charges"].isin(train_data["charge"].tolist()))]
    unispec = unispec[-(unispec["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        unispec["precursor_charges"].isin(train_data["charge"].tolist()))]
    all_unseen_psm += msnet.shape[0]
    all_unseen_precursors = pd.concat([all_unseen_precursors, msnet[["sequence", "charge"]]])
    desc = pd.Series(msnet["pcc"].describe())
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    desc = pd.Series(unispec["PCC"].describe())
    print(desc)
    print(unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0])
    unseen_pic_data = unseen_pic_data._append({"Model": "MSNet+Alphapeptdeep", "datasets": "Q Exactive HF-X/HCD@27",
                                               "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Alphapeptdeep", "datasets": "Q Exactive HF-X/HCD@27",
                                               "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[
                                                   0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Prosit", "datasets": "Q Exactive HF-X/HCD@27",
                                               "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Unispec", "datasets": "Q Exactive HF-X/HCD@27",
                                               "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                              ignore_index=True)

    msnet = pd.read_csv("./tests/PXD014877_Mus_musculus_MSNet_PCC_latest.csv")
    apeptdeep = pd.read_csv("./tests/PXD014877_Mus_musculus_Alphapeptdeep.csv")
    prosit = pd.read_csv("./tests/PXD014877_Mus_musculus_Prosit.csv")
    unispec = pd.read_csv("./tests/PXD014877_Mus_musculus_unispec.csv")
    print("------------PXD014877----------")
    msnet = pd.concat([msnet, pd.read_csv("./tests/PXD014877_Neurospora_MSNet_PCC_latest.csv")], ignore_index=True)
    apeptdeep = pd.concat([apeptdeep, pd.read_csv("./tests/PXD014877_Neurospora_Alphapeptdeep.csv")], ignore_index=True)
    prosit = pd.concat([prosit, pd.read_csv("./tests/PXD014877_Neurospora_Prosit.csv")], ignore_index=True)
    unispec = pd.concat([unispec, pd.read_csv("./tests/PXD014877_Neurospora_unispec.csv")], ignore_index=True)
    msnet = pd.concat([msnet, pd.read_csv("./tests/PXD014877_Bacteroides_Fragilis_MSNet_PCC_latest.csv")],
                      ignore_index=True)
    apeptdeep = pd.concat([apeptdeep, pd.read_csv("./tests/PXD014877_Bacteroides_Fragilis_Alphapeptdeep.csv")],
                          ignore_index=True)
    prosit = pd.concat([prosit, pd.read_csv("./tests/PXD014877_Bacteroides_Fragilis_Prosit.csv")], ignore_index=True)
    unispec = pd.concat([unispec, pd.read_csv("./tests/PXD014877_Bacteroides_Fragilis_unispec.csv")], ignore_index=True)
    all_psm += msnet.shape[0]
    all_precursors = pd.concat([all_precursors, msnet[["sequence", "charge"]]])
    desc = pd.Series(msnet["pcc"].describe())
    print(msnet.drop_duplicates(subset=["sequence", "charge"]).shape[0])
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    desc = pd.Series(unispec["PCC"].describe())
    print(desc)
    print(unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0])
    pic_data = pic_data._append({"Model": "MSNet+Alphapeptdeep", "datasets": "Q Exactive HF/HCD@27",
                                 "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Alphapeptdeep", "datasets": "Q Exactive HF/HCD@27",
                                 "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0]},
                                ignore_index=True)
    pic_data = pic_data._append({"Model": "Prosit", "datasets": "Q Exactive HF/HCD@27",
                                 "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Unispec", "datasets": "Q Exactive HF/HCD@27",
                                 "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                ignore_index=True)
    msnet = msnet[-(msnet["sequence"].isin(train_data["sequence"].tolist())) & (
        msnet["charge"].isin(train_data["charge"].tolist()))]
    apeptdeep = apeptdeep[-(apeptdeep["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        apeptdeep["precursor_charges"].isin(train_data["charge"].tolist()))]
    prosit = prosit[-(prosit["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        prosit["precursor_charges"].isin(train_data["charge"].tolist()))]
    unispec = unispec[-(unispec["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        unispec["precursor_charges"].isin(train_data["charge"].tolist()))]
    all_unseen_psm += msnet.shape[0]
    all_unseen_precursors = pd.concat([all_unseen_precursors, msnet[["sequence", "charge"]]])
    desc = pd.Series(msnet["pcc"].describe())
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    desc = pd.Series(unispec["PCC"].describe())
    print(desc)
    print(unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0])
    unseen_pic_data = unseen_pic_data._append({"Model": "MSNet+Alphapeptdeep", "datasets": "Q Exactive HF/HCD@27",
                                               "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Alphapeptdeep", "datasets": "Q Exactive HF/HCD@27",
                                               "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[
                                                   0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Prosit", "datasets": "Q Exactive HF/HCD@27",
                                               "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Unispec", "datasets": "Q Exactive HF/HCD@27",
                                               "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                              ignore_index=True)

    msnet = pd.read_csv("./tests/huiyan_MSNet_PCC_latest.csv")
    msnet = msnet.groupby(['sequence', 'charge'], group_keys=False).head(10)
    apeptdeep = pd.read_csv("./tests/huiyan_Alphapeptdeep.csv")
    apeptdeep = apeptdeep.groupby(['sequence', 'charge'], group_keys=False).head(10)
    prosit = pd.read_csv("./tests/huiyan_Prosit.csv")
    prosit = prosit.groupby(['sequence', 'charge'], group_keys=False).head(10)
    unispec = pd.read_csv("./tests/huiyan_unispec.csv")
    unispec = unispec.groupby(['sequence', 'charge'], group_keys=False).head(10)
    all_psm += msnet.shape[0]
    all_precursors = pd.concat([all_precursors, msnet[["sequence", "charge"]]])
    desc = pd.Series(msnet["pcc"].describe())
    print("------------huiyan----------")
    print(msnet.drop_duplicates(subset=["sequence", "charge"]).shape[0])
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    desc = pd.Series(unispec["PCC"].describe())
    print(desc)
    print(unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0])
    pic_data = pic_data._append({"Model": "MSNet+Alphapeptdeep", "datasets": "Exploris480/HCD@28&30",
                                 "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Alphapeptdeep", "datasets": "Exploris480/HCD@28&30",
                                 "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0]},
                                ignore_index=True)
    pic_data = pic_data._append({"Model": "Prosit", "datasets": "Exploris480/HCD@28&30",
                                 "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Unispec", "datasets": "Exploris480/HCD@28&30",
                                 "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                ignore_index=True)
    msnet = msnet[-(msnet["sequence"].isin(train_data["sequence"].tolist())) & (
        msnet["charge"].isin(train_data["charge"].tolist()))]
    apeptdeep = apeptdeep[-(apeptdeep["sequence"].isin(train_data["sequence"].tolist())) & (
        apeptdeep["charge"].isin(train_data["charge"].tolist()))]
    prosit = prosit[-(prosit["sequence"].isin(train_data["sequence"].tolist())) & (
        prosit["charge"].isin(train_data["charge"].tolist()))]
    unispec = unispec[-(unispec["sequence"].isin(train_data["sequence"].tolist())) & (
        unispec["charge"].isin(train_data["charge"].tolist()))]
    all_unseen_psm += msnet.shape[0]
    all_unseen_precursors = pd.concat([all_unseen_precursors, msnet[["sequence", "charge"]]])
    desc = pd.Series(msnet["pcc"].describe())
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    desc = pd.Series(unispec["PCC"].describe())
    print(desc)
    print(unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0])
    unseen_pic_data = unseen_pic_data._append({"Model": "MSNet+Alphapeptdeep", "datasets": "Exploris480/HCD@28&30",
                                               "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Alphapeptdeep", "datasets": "Exploris480/HCD@28&30",
                                               "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[
                                                   0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Prosit", "datasets": "Exploris480/HCD@28&30",
                                               "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Unispec", "datasets": "Exploris480/HCD@28&30",
                                               "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                              ignore_index=True)

    msnet = pd.read_csv("./tests/PXD000561_MSNet_PCC_32_latest.csv")
    apeptdeep = pd.read_csv("./tests/PXD000561_Alphapeptdeep_32_PCC.csv")
    prosit = pd.read_csv("./tests/PXD000561_Prosit_32_PCC.csv")
    unispec = pd.read_csv("./tests/PXD000561_unispec_32.csv")
    all_psm += msnet.shape[0]
    all_precursors = pd.concat([all_precursors, msnet[["sequence", "charge"]]])
    desc = pd.Series(msnet["pcc"].describe())
    print("---------PXD000561----------")
    print(msnet.drop_duplicates(subset=["sequence", "charge"]).shape[0])
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    pic_data = pic_data._append({"Model": "MSNet+Alphapeptdeep", "datasets": "Orbitrap Elite/HCD@32",
                                 "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Alphapeptdeep", "datasets": "Orbitrap Elite/HCD@32",
                                 "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0]},
                                ignore_index=True)
    pic_data = pic_data._append({"Model": "Prosit", "datasets": "Orbitrap Elite/HCD@32",
                                 "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]}, ignore_index=True)
    pic_data = pic_data._append({"Model": "Unispec", "datasets": "Orbitrap Elite/HCD@32",
                                 "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                ignore_index=True)
    msnet = msnet[-(msnet["sequence"].isin(train_data["sequence"].tolist())) & (
        msnet["charge"].isin(train_data["charge"].tolist()))]
    apeptdeep = apeptdeep[-(apeptdeep["sequence"].isin(train_data["sequence"].tolist())) & (
        apeptdeep["charge"].isin(train_data["charge"].tolist()))]
    prosit = prosit[-(prosit["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        prosit["precursor_charges"].isin(train_data["charge"].tolist()))]
    unispec = unispec[-(unispec["peptide_sequences"].isin(train_data["sequence"].tolist())) & (
        unispec["precursor_charges"].isin(train_data["charge"].tolist()))]
    all_unseen_psm += msnet.shape[0]
    all_unseen_precursors = pd.concat([all_unseen_precursors, msnet[["sequence", "charge"]]])
    desc = pd.Series(msnet["pcc"].describe())
    print(desc)
    print(msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0])
    desc = pd.Series(apeptdeep["PCC"].describe())
    print(desc)
    print(apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[0])
    desc = pd.Series(prosit["PCC"].describe())
    print(desc)
    print(prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0])
    desc = pd.Series(unispec["PCC"].describe())
    print(desc)
    print(unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0])
    unseen_pic_data = unseen_pic_data._append({"Model": "MSNet+Alphapeptdeep", "datasets": "Orbitrap Elite/HCD@32",
                                               "PCC90": msnet[msnet["pcc"] > 0.90].shape[0] / msnet.shape[0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Alphapeptdeep", "datasets": "Orbitrap Elite/HCD@32",
                                               "PCC90": apeptdeep[apeptdeep["PCC"] > 0.90].shape[0] / apeptdeep.shape[
                                                   0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Prosit", "datasets": "Orbitrap Elite/HCD@32",
                                               "PCC90": prosit[prosit["PCC"] > 0.90].shape[0] / prosit.shape[0]},
                                              ignore_index=True)
    unseen_pic_data = unseen_pic_data._append({"Model": "Unispec", "datasets": "Orbitrap Elite/HCD@32",
                                               "PCC90": unispec[unispec["PCC"] > 0.90].shape[0] / unispec.shape[0]},
                                              ignore_index=True)

    print(pic_data)
    plt.figure(dpi=400, figsize=(10, 6))
    plattle = {"MSNet+Alphapeptdeep": '#76CBB4', "Alphapeptdeep": '#FDCA93',
               "Prosit": '#F88455', "Unispec": '#3C9BC9'}
    sns.barplot(data=pic_data, x='datasets', y='PCC90', hue='Model', palette=plattle, errorbar=None,
                legend=False)
    plt.xticks(rotation=45, fontsize=16, ha='right')  # 可以改为 30、60、90 等
    plt.yticks(fontsize=16)  # 可以改为 30、60、90 等
    plt.ylabel("PCC90",fontsize=16)
    plt.xlabel('')
    plt.hlines(y=0.90, xmin=-0.5, xmax=7.5, linestyles='dotted')
    plt.hlines(y=0.80, xmin=-0.5, xmax=7.5, linestyles='dotted')
    # plt.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1, 1.2)).set_title('')  # , loc='upper right', bbox_to_anchor=(1, 1.2)
    plt.title("Overall PCC90 values", fontsize=16)
    plt.savefig("PCC90_latest_V4.svg", bbox_inches='tight')

    print(pic_data.groupby("Model")["PCC90"].mean())

    print(unseen_pic_data)
    plt.figure(dpi=400, figsize=(10, 6))
    plattle = {"MSNet+Alphapeptdeep": '#76CBB4', "Alphapeptdeep": '#FDCA93',
               "Prosit": '#F88455', "Unispec": '#3C9BC9'}
    sns.barplot(data=unseen_pic_data, x='datasets', y='PCC90', hue='Model', palette=plattle, errorbar=None,
                legend=False)
    plt.xticks(rotation=45, fontsize=16, ha='right')  # 可以改为 30、60、90 等
    plt.yticks(fontsize=16)  # 可以改为 30、60、90 等
    plt.xlabel('')
    plt.ylabel("PCC90",fontsize=16)
    plt.hlines(y=0.90, xmin=-0.5, xmax=7.5, linestyles='dotted')
    plt.hlines(y=0.80, xmin=-0.5, xmax=7.5, linestyles='dotted')
    # plt.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1.5, 1.2)).set_title(
    #     '')  # , loc='upper right', bbox_to_anchor=(1, 1.2)
    plt.title("Unseen precursors for MSNet", fontsize=16)
    plt.savefig("PCC90_unseen_latest_V4.svg", bbox_inches='tight')

    print(unseen_pic_data.groupby("Model")["PCC90"].mean())

    print(all_psm)
    print(all_unseen_psm)
    print(all_precursors.drop_duplicates(subset=["sequence", "charge"]).shape[0])
    print(all_unseen_precursors.drop_duplicates(subset=["sequence", "charge"]).shape[0])


if __name__ == '__main__':
    plot_latest_pcc()