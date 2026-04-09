# π-MSNet: A billion-scale, AI-Ready living proteomics data Portal

![π-MSNet Logo](assets/Figure1_V9.png)  

π-MSNet is a high-quality, large-scale, living data portal for computational proteomics. It provides standardized, AI-ready datasets for training, benchmarking, and developing machine learning models in proteomics. The portal integrates diverse mass spectrometry (MS) datasets from public repositories and in-house projects, offering unprecedented scale, diversity, and reproducibility.

## Overview

Proteomics increasingly relies on data-driven methods, particularly deep learning, to interpret complex mass spectrometry data. Existing datasets are often fragmented, incompletely annotated, or limited in scale, impeding reproducibility and fair benchmarking. π-MSNet addresses these limitations by providing a continuously updated, standardized, and scalable dataset resource that supports AI model development across diverse experimental conditions.

Key highlights of π-MSNet:

- **501 million peptide-spectrum matches (PSMs) and 9 million precursors** from 55 species, including eukaryotes, prokaryotes, viruses, and archaea.
- **1.66 billion MS² spectra** from 36,356 LC–MS/MS runs across 114 projects (~30 TB of raw data).
- Data acquired on **ten different mass spectrometer types** and processed with diverse fragmentation strategies.
- PSMs cover both **typical tryptic peptides** and peptides from **non-specific, Lys-C, Glu-C, and chymotrypsin cleavage**.
- Uniformly annotated in **SDRF format**, following the HUPO-PSI metadata standard.
- Stored in the **QPX Parquet format** for scalable, fast access and reduced storage requirements.

## Data Processing Workflow

1. **Data Curation:** Collected 114 public datasets from ProteomeXchange and π-HuB projects, covering diverse species, instruments, and experimental strategies.
2. **Uniform Annotation:** All datasets were standardized using SDRF format.
3. **Reanalysis:** MS² spectra were processed with the open-source [quantms](https://github.com/bigbio/quantms) workflow, integrating results from multiple search engines (e.g., MS-GF+ and Comet) to improve PSM robustness.
4. **Data Export:** PSMs and metadata exported to QPX Parquet format, optimized for rapid access, reduced storage (96% smaller than CSV), and efficient downstream AI workflows.
5. **Model Benchmarking:** Existing deep learning models were retrained and benchmarked on the π-MSNet dataset to demonstrate performance improvements.

![π-MSNet Workflow](assets/Figure2_Latest.png)  
*Figure 1: π-MSNet processing workflow.*

## MSNetLoader: Efficient Data Loading for π-MSNet

MSNetLoader is a Python utility designed to streamline access to π-MSNet datasets in QPX Parquet format. It enables efficient loading of PSMs and metadata, supports batch processing, and integrates seamlessly with machine learning workflows.

![MSNetLoader](assets/msnetloader.jpg)  

**Key Features:**

- Load PSMs and associated metadata from QPX Parquet files with minimal memory overhead.
- Supports batched and shuffled data access for model training and evaluation.
- Provides integration-ready PyTorch and TensorFlow dataset objects.

**Example Usage:**

```python
from msnetloader.ms2_loader import (
    AlphaPeptDeepConverter,
    MS2TorchDataset,
    LengthExactSampler,
)
from torch.utils.data import DataLoader

file_path = 'test_data/PXD014877-Akkermansia_muciniphilia-MSNet.parquet'
converter = AlphaPeptDeepConverter(file_path)
precursor_df, fragment_df = converter.convert_parquet_to_training_format()
dataset = MS2TorchDataset(precursor_df, fragment_df)
lengths = dataset.dataset["nAA"]
sampler = LengthExactSampler(
    lengths=lengths,
    batch_size=8,   
    shuffle=False 
)
dataloader = DataLoader(
    dataset.dataset,
    batch_sampler=sampler,
    num_workers=0,
    pin_memory=False
)

# Iterate through batches for model training
for batch_psms, batch_meta in dataloader:
    # process batch
    pass
```

## Data Access

- Interactive portal: [π-MSNet Portal](https://msnet.ncpsb.org.cn)  
- Dataset downloads and documentation: [π-MSNet Portal](https://msnet.ncpsb.org.cn) and [quantms Datasets](https://quantms.org/datasets)

## Citation

If you use π-MSNet in your research, please cite:
