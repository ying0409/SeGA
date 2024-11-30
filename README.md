# SeGA (AAAI 2024)
Official code and data of the paper [SeGA: Preference-Aware Self-Contrastive Learning with Prompts for Anomalous User Detection on Twitter](https://arxiv.org/abs/2312.11553).

## Overview
* We propose SeGA to address the challenging but emerging anomalous user detection task on Twitter.
* We introduce preference-aware self-contrastive learning to learn user behaviors via the corresponding posts.
* Extensive experiments on the proposed TwBNT benchmark demonstrate that SeGA significantly outperforms the state-of-the-art methods (+3.5% ∼ 27.6%).

## Data
We provide the user IDs and list IDs sampled from [Twibot-22](https://github.com/LuoUndergradXJTU/TwiBot-22) and user labels in this repo.

Download the complete dataset: https://drive.google.com/drive/folders/18q4qIwOH4QfG5eoDlYMigdTi6Dze3QbA?usp=sharing

## Reproducing SeGA
  ### To reproduce the SeGA model, follow these steps:
  * Encode node features
  ```
  python preprocess-sega.py
  ```
  * Run SeGA with list nodes and pre-train strategy
  ```
  python main.py --lst --pretrain
  ```

## Reference
If you use our dataset or find our project is relevant to your research, please consider citing our work!
```bibtex
@inproceedings{SeGA_AAAI2024,
  author       = {Ying{-}Ying Chang and
                  Wei{-}Yao Wang and
                  Wen{-}Chih Peng},
  title        = {SeGA: Preference-Aware Self-Contrastive Learning with Prompts for
                  Anomalous User Detection on Twitter},
  publisher = {{AAAI} Press},
  booktitle = {{AAAI}},
  year={2024},
  pages={30-37} 
}
```
