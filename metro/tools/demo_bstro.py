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
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import numpy as np
import cv2
from metro.modeling._smpl import SMPL, Mesh
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

def run_inference(args, METRO_model, smpl, mesh_sampler):
    smpl.eval()

    if args.distributed:
        METRO_model = torch.nn.parallel.DistributedDataParallel(
            METRO_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    METRO_model.eval()

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
        if METRO_model.config.output_attentions:
            _, _, pred_contact, hidden_states, att = METRO_model(images, smpl, mesh_sampler)
        else:
            _, _, pred_contact = METRO_model(images, smpl, mesh_sampler)

        
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
            temp_fname = foldername + '/visual.jpg'
            cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))

            for mi, mesh in enumerate(pred_contact_meshes):
                temp_fname = foldername + f'/contact_{mi}.obj'
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
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=True,
                        help="Path to specific checkpoint for inference.")
    parser.add_argument("--input_img", default='', type=str, required=True,
                        help="The path to the input image.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    
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
    logger = setup_logger("METRO", args.output_dir, get_rank())
    # set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()

    if args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _metro_network = torch.load(args.resume_checkpoint)
    else:
        raise ValueError("Invalid checkpoint {}".format(args.resume_checkpoint))
    
    _metro_network.to(args.device)
    run_inference(args, _metro_network, smpl, mesh_sampler)

   

if __name__ == "__main__":
    args = parse_args()
    main(args)
