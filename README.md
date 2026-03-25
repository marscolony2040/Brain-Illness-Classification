# Brain Illness Classification :brain:

Classifying Brain Images to test for Tumors, Trauma, and Schizophrenia

## Description :notebook:
I used Google Gemini to generate a high-fidelity synthetic dataset covering four distinct brain states: Normal, Trauma, Tumor, and Schizophrenia. I then engineered a custom CNN in PyTorch to handle feature extraction and classify these neurological patterns. To keep the project modular and reproducible, I decoupled the logic into a standalone trainer.py for model training. Finally, I loaded the model into a notebook to run batch inference and visualize how the predictions mapped against the ground-truth scans.

## Installation :computer:
[pip or pip3] install -r requirements.txt

## Runing :key:
[python or python3] trainer.py

## Sample Run (in Notebook)
![alt](https://github.com/marscolony2040/Brain-Illness-Classification/blob/main/predictions.png)


