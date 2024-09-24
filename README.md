# AsTFSONN 

**AsTFSONN: A Unified Framework Based on Time-Frequency Domain Self-Operational Neural Network for Asthmatic Lung Sound Classification** 

**Authors: Arka Roy, Udit Satija** 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rsarka34/AsTFSONN/blob/main/model/AsTFSONN.ipynb)
[![Paper Link](https://img.shields.io/badge/Paper%20Link-IEEE%20Xplore-blue)](https://ieeexplore.ieee.org/abstract/document/10171911)


# Abstract:
<p align="justify">
Asthma is one of the most severe chronic respiratory diseases which can be diagnosed using several modalities, such as lung function test or spirometric measures, peak flow meter-based measures, sputum eosinophils, pathological speech, and wheezing events of the lung auscultation sound, etc. Lung sound examinations are more accurate for diagnosing respiratory problems since these are associated with respiratory abnormalities occurred due to pulmonary disorders. In this paper, we propose a time-frequency domain self-operational neural network (SONN) based framework, namely, AsTFSONN, to efficiently categorize asthmatic lung sound signals, which uses the SONN-based heterogeneous neural model by incorporating an additional non-linearity into the neural network architecture, unlike the vanilla convolutional neural model that uses homogeneous perceptions which resemble the fundamental linear neuron model. The proposed framework comprises three major stages: pre-processing of the input lung sounds, mel-spectrogram time-frequency representation (TFR) extraction, and finally, classification using AsTFSONN based on the mel-spectrogram images. The proposed framework supersedes the notable prior works of asthma classification based on lung sounds and other diagnostic modalities by achieving the highest accuracy, specificity, sensitivity, and ICBHI-score of 98.50%, 98.80%, 98.11%, and 98.46%, respectively, using lung sounds as the input diagnostic modality, as evaluated on publicly available chest wall lung sound dataset.
</p>

# Methodology
![block_diag_astfonn (1)](https://github.com/rsarka34/AsTFSONN/assets/89518952/4e7f719d-0f45-4a48-a8b5-dcd0e0bf0ffc)

# Results
<p align="center">
<img src="https://github.com/user-attachments/assets/a4f6027c-736e-4b98-843f-d807cf5dff0b" alt="Description of image" width="1000"/>
</p>

# Performance
<p align="center">
  <img src="https://github.com/user-attachments/assets/37ccee0d-e5dd-42ed-8f91-2ebd9337b59e" alt="Description of image" width="800"/>
</p>

# Cite as
A. Roy and U. Satija, "AsTFSONN: A Unified Framework Based on Time-Frequency Domain Self-Operational Neural Network for Asthmatic Lung Sound Classification," *2023 IEEE International Symposium on Medical Measurements and Applications (MeMeA)*, Jeju, Korea, Republic of, 2023, pp. 1-6, doi: 10.1109/MeMeA57477.2023.10171911.

```bibtex
@INPROCEEDINGS{10171911,\
  author={Roy, Arka and Satija, Udit},\
  booktitle={2023 IEEE International Symposium on Medical Measurements and Applications (MeMeA)},\ 
  title={AsTFSONN: A Unified Framework Based on Time-Frequency Domain Self-Operational Neural Network for Asthmatic Lung Sound Classification},\ 
  year={2023},\
  volume={},\
  number={},\
  pages={1-6},\
  doi={10.1109/MeMeA57477.2023.10171911}}
