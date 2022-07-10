# Quick Demo 
We provide a demo inference code to estimate human-scene contact.

Our inference codes will take a image as input and generate the results.

## Human-Scene Contact Detection

This demo runs 3D human-scene contact detection from a single image. 

Our codes require the input images that are already **cropped with the person centered** in the image. The input images can have arbitrary size and the demo code scales it to `224x224`. 
Check `./samples/body-scene-contact/sample1.jpg` for example. 

Run the following script for demo:


```bash
python ./metro/tools/demo_bstro.py 
       --num_hidden_layers 4 
       --num_attention_heads 4 
       --input_feat_dim 2051,512,128 
       --hidden_feat_dim 1024,256,128 
       --input_img samples/body-scene-contact/sample1.png
       --output_dir ./demo 
       --resume_checkpoint models/bstro/hsi_hrnet_3dpw_b32_checkpoint_15.bin
```
After running, it will generate the results in the folder `./demo`. `input.jpg` is the input image in `224x224` size (as a sanity check); `contact_vis.obj` is a body mesh in T-pose where vertices in contact with the scene are in red color:

 <img src="../docs/res_vis_1.png" width="500"> 

Note that BSTRO focuses on estimating *contact*, not poses or shapes. The T-posed mesh is only for visualization purposes. 

## Limitations

 - **This demo doesn't perform human detection**. Our model requires a centered target in the image. 
 - As **BSTRO is a data-driven approach**, it may not perform well if the test samples are very different from the training data. 
 - **BSTRO considers the SMPL mesh topology**. It needs vertex correspondences to transfer the results to other body meshes, e.g., SMPL-X, GHUM. 



