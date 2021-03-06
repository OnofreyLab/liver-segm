# Improved performance and consistency of deep learning 3D liver segmentation with heterogeneous cancer stages in magnetic resonance imaging

Moritz Gross<sup>1, 2</sup>, M.S.; Michael Spektor<sup>1</sup>, M.D.; Ariel Jaffe<sup>3</sup>, M.D.; Ahmet S. Kucukkaya<sup>1,2</sup>, M.S.; Simon Iseke<sup>1,4</sup>, M.S.; Stefan P. Haider<sup>1,5</sup>, M.S.; Mario Strazzabosco<sup>3</sup>, M.D., Ph.D.; Julius Chapiro<sup>1</sup>, M.D., Ph.D.; John A. Onofrey<sup>1,6</sup>, Ph.D.

[Cite this article](#cite-this-article)

## Abstract

**Purpose:** Accurate liver segmentation is key for volumetry assessment to guide treatment decisions. Moreover, it is an important pre-processing step for cancer detection algorithms. Liver segmentation can be especially challenging in patients with cancer-related tissue changes and shape deformation. The aim of this study was to assess the ability of state-of-the-art deep learning 3D liver segmentation algorithms to generalize across all different Barcelona Clinic Liver Cancer (BCLC) liver cancer stages.

**Methods:** This retrospective study, included patients from an institutional database that had arterial-phase T1-weighted magnetic resonance images with corresponding manual liver segmentations. The data was split into 70/15/15% for training/validation/testing each proportionally equal across BCLC stages. Two 3D convolutional neural networks were trained using identical U-net-derived architectures with equal sized training datasets: one spanning all BCLC stages (“All-Stage-Net": AS-Net), and one limited to early and intermediate BCLC stages (“Early-Intermediate-Stage-Net": EIS-Net). Segmentation accuracy was evaluated by the Dice Similarity Coefficient (DSC) on a dataset spanning all BCLC stages and a Wilcoxon signed-rank test was used for pairwise comparisons.

**Results:** 219 subjects met the inclusion criteria (170 males, 49 females, 62.8±9.1 years) from all BCLC stages. Both networks were trained using 129 subjects: AS-Net training comprised 19, 74, 18, 8, and 10 BCLC 0, A, B, C, and D patients, respectively; EIS-Net training comprised 21, 86, and 22 BCLC 0, A, and B patients, respectively. DSCs (mean±SD) were 0.954±0.018 and 0.946±0.032 for AS-Net and EIS-Net (p<0.001), respectively. The AS-Net 0.956±0.014 significantly outperformed the EIS-Net 0.941±0.038 on advanced BCLC stages (p<0.001) and yielded similarly good segmentation performance on early and intermediate stages (AS-Net: 0.952±0.021; EIS-Net: 0.949±0.027; p=0.107).

**Conclusion:** To ensure robust segmentation performance across cancer stages that is independent of liver shape deformation and tumor burden, it is critical to train deep learning models on heterogeneous imaging data spanning all BCLC stages.


## Author information

**Affiliations**

<sup>1</sup>	Department of Radiology and Biomedical Imaging, Yale University School of Medicine, New Haven, Connecticut, United States of America 

<sup>2</sup>	Charité Center for Diagnostic and Interventional Radiology, Charité - Universitätsmedizin Berlin, Berlin, Germany

<sup>3</sup>	Department of Internal Medicine, Yale University School of Medicine, New Haven, Connecticut, United States of America

<sup>4</sup>	Department of Diagnostic and Interventional Radiology, Pediatric Radiology and Neuroradiology, Rostock University Medical Center, Rostock, Germany

<sup>5</sup> 	Department of Otorhinolaryngology, University Hospital of Ludwig Maximilians Universität München, Munich, Germany

<sup>6</sup>  Department of Urology, Yale University School of Medicine, New Haven, Connecticut, United States of America

## Cite this article
Gross, M., Spektor, M., Jaffe, A., Kucukkaya, A. S., Iseke, S., Haider, S. P., Strazzabosco, M., Chapiro, J., & Onofrey, J. A. (2021). Improved performance and consistency of deep learning 3D liver segmentation with heterogeneous cancer stages in magnetic resonance imaging. PLOS ONE, 16(12), e0260630.

### DOI
https://doi.org/10.1371/journal.pone.0260630


