# Training

1. Download the `tsv databases for train/val/test` and `geodesic dist. matrix` from the [RICH website](https://rich.is.tue.mpg.de/download.php). Unzip the zip file and place them under `dataset` folder following the structure below.
    ```
    ${REPO_DIR}  
    |-- models  
    |-- metro 
    |-- datasets 
    |   |-- rich_for_bstro_tsv_db
    |   |   |-- train.img.tsv
    |   |   |-- train.hw.tsv
    |   |   |-- train.label.tsv
    |   |   |-- ...
    |   |   |-- test.img.tsv
    |   |   |-- test.hw.tsv
    |   |   |-- test.label.tsv
    |   |   |-- ...
    |   |   |-- val.img.tsv
    |   |   |-- val.hw.tsv
    |   |   |-- val.label.tsv
    |   |   |-- ...
    |   |-- smpl_neutral_geodesic_dist.npy
    |-- predictions 
    |-- README.md 
    |-- ... 
    ``` 

2. Run the following command
    ```
    bash scripts/training.sh 32       # 32 is the batch size
    ```