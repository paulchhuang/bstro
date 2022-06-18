"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training and evaluation codes for 
3D human body mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import torch
import torchvision.models as models
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import numpy as np
import cv2
from metro.modeling.bert import BertConfig, METRO
from metro.modeling.bert.modeling_bstro import BSTRO_BodyHSC_Network as BSTRO_Network
from metro.modeling._smpl import SMPL, Mesh
from metro.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from metro.modeling.hrnet.config import config as hrnet_config
from metro.modeling.hrnet.config import update_config as hrnet_update_config
from metro.utils.image_ops import crop
from metro.utils.logger import setup_logger
from metro.utils.comm import synchronize, is_main_process, get_rank
from metro.utils.miscellaneous import mkdir

def rgb_processing(rgb_img, center, scale, rot, pn, img_res=224):
    """Process rgb image and do augmentation."""
    rgb_img = crop(rgb_img, center, scale, [img_res, img_res], rot=rot)
    
    # in the rgb image we add pixel noise in a channel-wise manner
    rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
    rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
    rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
    # (3,224,224),float,[0,1]
    rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
    return rgb_img

def run_inference(args, BSTRO_model, smpl, mesh_sampler):
    smpl.eval()

    if args.distributed:
        BSTRO_model = torch.nn.parallel.DistributedDataParallel(
            BSTRO_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    BSTRO_model.eval()

    normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        
        img = cv2.imread(args.input_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        center = [ w / 2., h / 2.]
        scale = max(h, w) / 200.0

        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        img = rgb_processing(img, center, sc*scale, rot, pn)
        
        img = torch.from_numpy(img).float()
        # Store image before normalization to use it in visualization
        images = normalize_img(img)
        images = images.cuda(args.device).unsqueeze(0)

        # forward-pass
        if BSTRO_model.config.output_attentions:
            _, _, pred_contact, hidden_states, att = BSTRO_model(images, smpl, mesh_sampler)
        else:
            _, _, pred_contact = BSTRO_model(images, smpl, mesh_sampler)

        
        visual_imgs, pred_contact_meshes = visualize_contact([img],
                                                            pred_contact.detach(), 
                                                            smpl)
        visual_imgs = visual_imgs.transpose(0,1)
        visual_imgs = visual_imgs.transpose(1,2)
        visual_imgs = np.asarray(visual_imgs)

        if is_main_process()==True:
            foldername = './demo'
            if not os.path.exists(foldername):
                os.makedirs(foldername)
            temp_fname = foldername + '/input.jpg'
            cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))

            for _, mesh in enumerate(pred_contact_meshes):
                temp_fname = foldername + f'/contact_vis.obj'
                mesh.export(temp_fname)
    return


def visualize_contact(images,
                    pred_contact, 
                    smpl):
    ref_vert = smpl(torch.zeros((1, 72)).cuda(args.device), torch.zeros((1,10)).cuda(args.device)).squeeze()
    rend_imgs = []
    pred_contact_meshes = []
    batch_size = pred_contact.shape[0]

    import trimesh
    # Do visualization for the first 6 images of the batch

    for i in range(min(batch_size, 50)):
        img = images[i].cpu().numpy()
        # Get predict vertices for the particular example
        contact = pred_contact[i].cpu()
        hit_id = (contact >= 0.5).nonzero()[:,0]

        pred_mesh = trimesh.Trimesh(vertices=ref_vert.detach().cpu().numpy(), faces=smpl.faces.detach().cpu().numpy(), process=False)
        pred_mesh.visual.vertex_colors = (191, 191, 191, 255)
        pred_mesh.visual.vertex_colors[hit_id, :] = (255, 0, 0, 255)
        pred_contact_meshes.append(pred_mesh)

        # Visualize reconstruction and detected pose
        rend_imgs.append(torch.from_numpy(img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs, pred_contact_meshes


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,128', type=str, 
                        help="The Image Feature Dimension.")   
    parser.add_argument("--legacy_setting", default=True, action='store_true',)
    
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=True,
                        help="Path to specific checkpoint for inference.")
    parser.add_argument("--input_img", default='', type=str, required=True,
                        help="The path to the input image.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--model_name_or_path", default='metro/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument("--output_attentions", default=False, action='store_true',) 


    args = parser.parse_args()
    return args


def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)
    if args.distributed:
        # print("Init distributed training on local rank {} ({}), rank {}, world size {}".format(args.local_rank, int(os.environ["LOCAL_RANK"]), int(os.environ["NODE_RANK"]), args.num_gpus))
        print("Init distributed training on local rank {} ({}), rank {}, world size {}".format(args.local_rank, int(os.environ["LOCAL_RANK"]), 0, args.num_gpus))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device("cuda", local_rank)
        synchronize()

    mkdir(args.output_dir)
    logger = setup_logger("BSTRO", args.output_dir, get_rank())
    # set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()

    # Load model
    trans_encoder = []

    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [1]

    for i in range(len(output_feat_dim)):
        config_class, model_class = BertConfig, METRO
        config = config_class.from_pretrained(args.config_name if args.config_name \
                else args.model_name_or_path)

        config.output_attentions = args.output_attentions
        config.hidden_dropout_prob = 0.1
        config.img_feature_dim = input_feat_dim[i] 
        config.output_feature_dim = output_feat_dim[i]
        args.hidden_size = hidden_feat_dim[i]

        if args.legacy_setting==True:
            # During our paper submission, we were using the original intermediate size, which is 3072 fixed
            # We keep our legacy setting here 
            args.intermediate_size = -1
        else:
            # We have recently tried to use an updated intermediate size, which is 4*hidden-size.
            # But we didn't find significant performance changes on Human3.6M (~36.7 PA-MPJPE)
            args.intermediate_size = int(args.hidden_size*4)

        # update model structure if specified in arguments
        update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

        for idx, param in enumerate(update_params):
            arg_param = getattr(args, param)
            config_param = getattr(config, param)
            if arg_param > 0 and arg_param != config_param:
                logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                setattr(config, param, arg_param)

        # init a transformer encoder and append it to a list
        assert config.hidden_size % config.num_attention_heads == 0
        model = model_class(config=config) 
        logger.info("Init model from scratch.")
        trans_encoder.append(model)

    
    # init ImageNet pre-trained backbone model
    if args.arch=='hrnet':
        hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
        logger.info('=> loading hrnet-v2-w40 model')
    elif args.arch=='hrnet-w64':
        hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
        logger.info('=> loading hrnet-v2-w64 model')
    else:
        print("=> using pre-trained model '{}'".format(args.arch))
        backbone = models.__dict__[args.arch](pretrained=True)
        # remove the last fc layer
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

    trans_encoder = torch.nn.Sequential(*trans_encoder)
    total_params = sum(p.numel() for p in trans_encoder.parameters())
    logger.info('Transformers total parameters: {}'.format(total_params))
    backbone_total_params = sum(p.numel() for p in backbone.parameters())
    logger.info('Backbone total parameters: {}'.format(backbone_total_params))

    # build end-to-end METRO network (CNN backbone + multi-layer transformer encoder)
    _bstro_network = BSTRO_Network(args, config, backbone, trans_encoder, mesh_sampler)

    if args.resume_checkpoint!=None and args.resume_checkpoint!='None':# and 'state_dict' not in args.resume_checkpoint:
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        cpu_device = torch.device('cpu')
        state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
        _bstro_network.load_state_dict(state_dict, strict=False)
        del state_dict
    else:
        raise ValueError("Invalid checkpoint {}".format(args.resume_checkpoint))
    
    _bstro_network.to(args.device)
    run_inference(args, _bstro_network, smpl, mesh_sampler)

   

if __name__ == "__main__":
    args = parse_args()
    main(args)
