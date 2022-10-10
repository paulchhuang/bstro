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
    bash scripts/training.sh 32         # 32 is the batch size.
    ```
    We trained BSTRO with a GPU with 32GB memory. Please adjust this number according to your GPUs.

    One should see the following output:
    ```
    ...
    METRO INFO: Update config parameter num_hidden_layers: 12 -> 4
    METRO INFO: Update config parameter hidden_size: 768 -> 1024
    METRO INFO: Update config parameter num_attention_heads: 12 -> 4
    METRO INFO: Init model from scratch.
    METRO INFO: Update config parameter num_hidden_layers: 12 -> 4
    METRO INFO: Update config parameter hidden_size: 768 -> 256
    METRO INFO: Update config parameter num_attention_heads: 12 -> 4
    METRO INFO: Init model from scratch.
    METRO INFO: Update config parameter num_hidden_layers: 12 -> 4
    METRO INFO: Update config parameter hidden_size: 768 -> 128
    METRO INFO: Update config parameter num_attention_heads: 12 -> 4
    ...
    => loading pretrained model models/hrnet/hrnetv2_w64_imagenet_pretrained.pth
    METRO INFO: => loading hrnet-v2-w64 model
    METRO INFO: Transformers total parameters: 102256130
    METRO INFO: Backbone total parameters: 128059944
    METRO INFO: Loading state dict from checkpoint models/metro_release/metro_3dpw_state_dict.bin
    METRO INFO: => initializing with metro weights from models/metro_release/metro_3dpw_state_dict.bin
    ...
    rich_for_bstro_tsv_db/train.yaml
    rich_for_bstro_tsv_db/val.yaml
    ...
    ```

    The log file `log.txt` and intermediate training files can be found under `${REPO_DIR}/output`.