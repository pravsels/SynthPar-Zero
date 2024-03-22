# SynthPar: Synthetic Faces with Demographic Parity

SynthPar aims to facilitate the development and evaluation of face recognition models, with the goal of closing the gap in performance across all demographic groups.

It provides 2 key resources:

- A conditional StyleGAN2 generator that allows users to create synthetic face images for specified attributes like race and sex.
    
- A dataset of 80,000 synthetic face images evenly distributed across 8 categories (4 races × 2 sexes), built upon the VGGFace dataset and labels.


## Loading the dataset

The dataset can be loaded from the HuggingFace repository:

```
from datasets import load_dataset

dataset = load_dataset("pravsels/synthpar")
```


## Generating images

Create an anaconda environment:
```
chmod +x *.sh

./install_conda_env.sh
```

Activate the environment:
```
conda activate synthpar
```

Run the generation script with the desired configuration:
```
python generation_script.py -c configs/BlackFemale.yaml
```

Please find the configs for the other demographics in the `configs` folder. 
