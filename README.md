# SeGA: Preference-Aware Self-Contrastive Learning with Prompts for Anomalous User Detection on Twitter (AAAI 2024)
Official code and data of the paper SeGA: Preference-Aware Self-Contrastive Learning with Prompts for Anomalous User Detection on Twitter.

## About
* We propose SeGA to address the challenging but emerging anomalous user detection task on Twitter.
* We introduce preference-aware self-contrastive learning to learn user behaviors via the corresponding posts.
* Extensive experiments on the proposed TwBNT benchmark demonstrate that SeGA significantly outperforms the state-of-the-art methods (+3.5% ∼ 27.6%).

## Reproducing SeGA
  ### To reproduce the SeGA model, follow these steps:
  * Encode node features with the following command
  ```
  python preprocess-sega.py
  ```
  * Run SeGA with the following command
  ```
  python main.py --lst --pretrain
  ```
