# Improved performance and consistency of deep learning 3D liver segmentation with heterogenous cancer stages in magnetic resonance imaging

Moritz Gross1, 2, M.S.; Michael Spektor1, M.D.; Ariel Jaffe3, M.D.; Ahmet S. Kucukkaya1,2, M.S.; Simon Iseke1,4, M.S.; Stefan P. Haider1,5, M.S.; Mario Strazzabosco3, M.D., Ph.D.; Julius Chapiro1, M.D., Ph.D.; John A. Onofrey1,6, Ph.D.

## Abstract

**Purpose:** Accurate liver segmentation is key for volumetry assessment to guide treatment decisions. Moreover, it is an important pre-processing step for cancer detection algorithms. Liver segmentation can be especially challenging in patients with cancer-related tissue changes and shape deformation. The aim of this study was to assess the ability of state-of-the-art deep learning 3D liver segmentation algorithms to generalize across all different Barcelona Clinic Liver Cancer (BCLC) liver cancer stages.

**Methods:** This retrospective study, included patients from an institutional database that had arterial-phase T1-weighted magnetic resonance images with corresponding manual liver segmentations. The data was split into 70/15/15% for training/validation/testing each proportionally equal across BCLC stages. Two 3D convolutional neural networks were trained using identical U-net-derived architectures with equal sized training datasets: one spanning all BCLC stages (“All-Stage-Net": AS-Net), and one limited to early and intermediate BCLC stages (“Early-Intermediate-Stage-Net": EIS-Net). Segmentation accuracy was evaluated by the Dice Similarity Coefficient (DSC) on a dataset spanning all BCLC stages and a Wilcoxon signed-rank test was used for pairwise comparisons.

**Results:** 219 subjects met the inclusion criteria (170 males, 49 females, 62.8±9.1 years) from all BCLC stages. Both networks were trained using 129 subjects: AS-Net training comprised 19, 74, 18, 8, and 10 BCLC 0, A, B, C, and D patients, respectively; EIS-Net training comprised 21, 86, and 22 BCLC 0, A, and B patients, respectively. DSCs (mean±SD) were 0.954±0.018 and 0.946±0.032 for AS-Net and EIS-Net (p<0.001), respectively. The AS-Net 0.956±0.014 significantly outperformed the EIS-Net 0.941±0.038 on advanced BCLC stages (p<0.001) and yielded similarly good segmentation performance on early and intermediate stages (AS-Net: 0.952±0.021; EIS-Net: 0.949±0.027; p=0.107).

**Conclusion:** To ensure robust segmentation performance across cancer stages that is independent of liver shape deformation and tumor burden, it is critical to train deep learning models on heterogenous imaging data spanning all BCLC stages.

