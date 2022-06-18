# --------------------------------
# Setup
# --------------------------------
export REPO_DIR=$PWD
if [ ! -d $REPO_DIR/models ] ; then
    mkdir -p $REPO_DIR/models
fi

# --------------------------------
# Download the ImageNet pre-trained HRNet models 
# The weights are provided by https://github.com/HRNet/HRNet-Image-Classification
# --------------------------------

BLOB='https://datarelease.blob.core.windows.net/metro'

if [ ! -d $REPO_DIR/models/hrnet ] ; then
    mkdir -p $REPO_DIR/models/hrnet
fi
wget -nc $BLOB/models/hrnetv2_w64_imagenet_pretrained.pth -O $REPO_DIR/models/hrnet/hrnetv2_w64_imagenet_pretrained.pth
wget -nc $BLOB/models/hrnetv2_w40_imagenet_pretrained.pth -O $REPO_DIR/models/hrnet/hrnetv2_w40_imagenet_pretrained.pth
wget -nc $BLOB/models/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml -O $REPO_DIR/models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
wget -nc $BLOB/models/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml -O $REPO_DIR/models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml

# --------------------------------
# Download the 3DPW pre-trained METRO models 
# The weights are provided by https://github.com/microsoft/MeshTransformer
# --------------------------------
if [ ! -d $REPO_DIR/models/metro_release ] ; then
    mkdir -p $REPO_DIR/models/metro_release
fi
wget -nc $BLOB/models/metro_3dpw_state_dict.bin -O $REPO_DIR/models/metro_release/metro_3dpw_state_dict.bin
