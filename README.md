This repository contains the code to reproduce experiments performed in VISUAL PROMPTING UPGRADES NEURAL NETWORK SPARSIFICATION: A DATA-MODEL PERSPECTIVE

1. Install requirements: 
```
conda create -n vpns python=3.8
conda activate vpns
pip install -r requirements.txt
```

2. Symlink datasets (Optional):

If you already have the datasets downloaded, create a symlink. If you skip this step, the datasets will be downloaded automatically.
```
mkdir ./dataset
ln -s <DATASET_PARENT_DIR> ./datasets
```

3. Run VPNs pruning

You can run the vpns.sh file to replicate our results of VPNs, change the network and dataset to replicate all our main results.
```
bash vpns.sh
```
