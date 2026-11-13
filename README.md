# Bird Call Separation

This is a research project done for Johns Hopkins University `Deep Learning Using Transformers - 705.744` course. It implements a simplified [SR Corrnet](https://arxiv.org/html/2603.29097v1)-like transformer model with early splitting for bird vocalization splitting. It includes a custom RoPE self-attention implementation that gets used for the time and frequency based attention blocks. Separately, it uses the same self-attention implementation for speaker based attention blocks. The code in this repository is a prototype and could be further refined for better performance.

## Environment Setup

Run the following commands:

```bash
conda env create -f environment.yaml
conda activate birdcallseparation
```

## Prepare Dataset

The full dataset is available at https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/rluo15_jh_edu/IQChtaN0mwcDSroT4fg83E_xAbmy9XC2oKZpxUIbaJXUNC0?e=vJ5s3v. Create a folder called data at the base of this project (next to src). Download the file to data, then cd into data and do `tar -xf iNaturalistMixtures.tar`. It should create `iNaturalistAudioSubset`.

Otherwise, to prepare the dataset from scratch, download the iNaturalist Sounds Dataset (https://github.com/visipedia/inat_sounds/tree/main/2024). Then, from the scripts directory, run:
 - mix_birdcalls.py ../data/iNaturalistAudioSubset/non-mixed -o ../data/iNaturalistAudioSubset/\<split\> -n \<number of desired samples\>

where split is train, val, and test.

## Training

To train the model, update the constants at the top of each file and run:
 - python train_modified_sr_corrnet.py \<path to your configuration file\>

Pretrained weights can be found at https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/rluo15_jh_edu/IQBW2Oul0muISr1fR11xs0ohAXGFTqjxiY5z6UML-Acd-II?e=ssjCNV. Create a models folder at the base of this project (next to src) and put this model in that folder.

## Evaluation

To evaluate a model, run `evaluate_modified_sr_corrnet.py`, updating the CHECKPOINT_PATH and DATASET_PATH as appropriate.
