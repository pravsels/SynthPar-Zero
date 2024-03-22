# SynthPar: Synthetic Faces with Demographic Parity

SynthPar aims to facilitate the development and evaluation of face recognition models, particularly in terms of fairness and demographic balance.

It provides 2 key resources:

- A conditional StyleGAN2 generator that allows users to create synthetic face images based on specified attributes like race and sex.
    
- A dataset of 80,000 synthetic face images evenly distributed across 8 categories (4 races × 2 sexes), built upon the VGGFace dataset and labels.

## Loading the dataset

Load the dataset from the Hugging Face dataset repository `pravsels/synthpar`:

```
from datasets import load_dataset

dataset = load_dataset("pravsels/synthpar")
```


## Generating images

To generate your own synthetic face images using the provided conditional StyleGAN2 generator, follow these steps:

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
python generation_script.py -c BlackFemale.yaml
```

The configuration YAML files contains the following parameters:
- class_index: The index of the target class (race and sex) for generation.
- no_of_identities_per_class: The number of unique identities to generate for the specified class.
- images_per_batch: The number of images to generate per batch.

