from pycirclize import Circos
from pycirclize.utils import ColorCycler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# np.random.seed(0)
#

data = pd.read_csv("explore_summary.csv")

tmp = data.groupby("Species")["#PSM"].sum().reindex()
# print(data)
ref = data.drop_duplicates(subset=["Species", "Superkingdom"], keep="first")
data = pd.merge(tmp, ref[["Species", "Superkingdom"]], on="Species", how="left")
print(data)
# data.to_csv("F:/MSNet/web/explore_figure1.csv", index=False)
for g, d in data.groupby("Superkingdom"):
    print(g)
    print(d["Species"].nunique())

sectors = {"Archea": 2, "Prokaryote": 25, "Eukaryote": 22, "Virus": 6}
circos = Circos(sectors, space=10)
for sector in circos.sectors:
    print(sector)
    vmin, vmax = 1, 700000000
    if sector.name == "Prokaryote":
        y = data[data["Superkingdom"] == sector.name]
        y.loc[y["Species"] == "Riftia pachyptila", "#PSM"] = 1973461 / 10
        y = y["#PSM"].tolist()
        print(y)
    elif sector.name == "Eukaryote":
        y = data[data["Superkingdom"] == sector.name]
        y.loc[y["Species"] == "Homo sapiens", "#PSM"] = 385748340 / 50
        y.loc[y["Species"] == "Mus musculus", "#PSM"] = 76144110 / 10
        y.loc[y["Species"] == "Arabidopsis", "#PSM"] = 17606574 / 10

        y = y["#PSM"].tolist()
        print(y)
    else:
        y = data[data["Superkingdom"] == sector.name]["#PSM"].tolist()

    x = np.linspace(sector.start + 0.5, sector.end - 0.5, int(sector.size))
    # y = np.random.randint(vmin, vmax, len(x))
    sector.text(f"{sector.name}", size=11, r=160)

    # Set Track01 (Radius: 75 - 100)
    # Plot bar (default)
    track1 = sector.add_track((50, 70), r_pad_ratio=0.1)
    track1.axis()
    # track1.xticks_by_interval(1)
    track1.bar(x, y, tick_label=None)
    # track1.xticks_by_interval(0.1, tick_length=1, show_label=False, label_size=5)
    pos_list = list(range(0, int(track1.size)))
    labels = [f"{i:02d}" for i in pos_list]
    track1.xticks(pos_list, data[data["Superkingdom"] == sector.name]["Species"].tolist(), label_orientation="vertical",
                  label_size=9)
    if sector.name == "Archea":
        y = [0, 10000, 20000, 30000]
        y_labels = ["0", "$10^4$", "$20^4$", "$30^4$"]
        track1.yticks(y, y_labels, side="left", line_kws=dict(color="black", lw=1), text_kws=dict(color="black"),
                      label_size=6)
    elif sector.name == "Prokaryote":
        y = [0, 10000, 50000, 200000]
        y_labels = ["0", "$20^4$", "$80^4$", "$20^5$"]
        track1.yticks(y, y_labels, side="left", line_kws=dict(color="black", lw=1), text_kws=dict(color="black"),
                      label_size=6)
    elif sector.name == "Virus":
        y = [0, 1000, 2000]
        y_labels = ["0", "$10^3$", "$20^3$"]
        track1.yticks(y, y_labels, side="left", line_kws=dict(color="black", lw=1), text_kws=dict(color="black"),
                      label_size=6)
    elif sector.name == "Eukaryote":
        y = [10**5, 55**6, 60**6]
        y_labels = ["$10^5$", "$20^7$", "$30^8$"]
        track1.yticks(y, y_labels, side="left", line_kws=dict(color="black", lw=1), text_kws=dict(color="black"),
                      label_size=6)

    # Plot stacked bar with user-specified params
    # track2 = sector.add_track((50, 70))
    # track2.axis()
    # track2.xticks_by_interval(1, outer=False)
    # track2.xticks_by_interval(0.1, tick_length=1, show_label=False, label_size=5)
    # track2.bar(x, y, tick_label=data[data["Superkingdom"] == sector.name]["Species"].tolist())
    #
    # ColorCycler.set_cmap("tab10")
    # tab10_colors = [ColorCycler() for _ in range(len(x))]
    # track2.bar(x, y, width=1.0, color=tab10_colors, ec="grey", lw=0.5, vmax=vmax * 2)
    #
    # ColorCycler.set_cmap("Pastel1")
    # pastel_colors = [ColorCycler() for _ in range(len(x))]
    # y2 = np.random.randint(vmin, vmax, len(x))
    # track2.bar(x, y2, width=1.0, bottom=y, color=pastel_colors, ec="grey", lw=0.5, hatch="//", vmax=vmax * 2)


fig = circos.plotfig()
fig.savefig("MSNet_data_overview_v3.svg", dpi=500)
#
colors = sns.color_palette("GnBu", n_colors=10)

# #
instrument = ["LTQ Orbitrap Elite", "LTQ Orbitrap Velos", "Q Exactive",
              "Orbitrap Fusion", "Q Exactive Plus", "Q Exactive HF", "Orbitrap Fusion Lumos",
              "Q Exactive HF-X", "Orbitrap Exploris 480", "timsTOF"]


tmp = pd.read_excel("instrument.xlsx", sheet_name="Sheet1")
tmp['instrument'] = pd.Categorical(tmp['instrument'], categories=instrument, ordered=True)
print(tmp)
print(tmp["instrument"])
plt.figure(figsize=(6,6), dpi=500)
plt.pie(tmp["psm"].tolist(), labels=tmp["instrument"], autopct='%1.1f%%', startangle=140, colors=colors,
        pctdistance=0.9, labeldistance=1.1, #explode=(0, 0.2, 0.1, 0.4, 0.3),
        textprops={'fontsize': 8})
# plt.show()
plt.savefig("./figures/MSNet_pie_instrument_v2.svg", dpi=500, bbox_inches='tight')
# # #
# #
# #enzyme
tmp = data.groupby("Enzyme")["#PSM"].sum().reset_index(name="#PSM")

s = tmp["psm"].sum()
Trypsin = tmp[tmp["Degestion"] == "Trypsin"]["psm"].values[0] / s
non_Trypsin = 1 - Trypsin
print(tmp)
tmp.to_csv("E:/MSNet/web/enzyme_pie.csv", index=False)

print(Trypsin)
print(non_Trypsin)
fig, ax = plt.subplots(figsize=(2, 4), dpi=500)

ax.bar(["Trypsin", "Non-Trypsin"], [Trypsin, non_Trypsin],
       color=["#F46F44", "#54686F"])
for x, y in zip([0, 1], [0.98, 0.02]):
    plt.text(x + 0.01, y + 0.01, str(y * 100) + "%", ha='center', va='bottom', fontsize=8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.show()
plt.savefig("./figures/MSNet_pie_enzyme.png", dpi=500, bbox_inches='tight')
#
colors = sns.color_palette("coolwarm")
# colors = ["#264653", "#2A9D8C", "#E9C46B", "#E66F51"]
plt.figure(dpi=500)
tmp = tmp[tmp["Enzyme"] != "Trypsin"]
tmp.loc[tmp["Enzyme"] != "unspecific cleavage", "#PSM"] = (
    tmp.loc[tmp["Enzyme"] != "unspecific cleavage", "#PSM"] + 1000000
)
plt.pie(tmp["#PSM"].tolist(), labels=tmp["Enzyme"], autopct='%1.1f%%', startangle=140, colors=colors,
        textprops={'fontsize': 8})
# plt.show()
plt.savefig("./figures/MSNet_pie_Degestion.svg", dpi=500, bbox_inches='tight')