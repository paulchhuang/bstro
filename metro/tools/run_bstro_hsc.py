"""
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
You can only use this computer program if you have closed
a license agreement with MPG or you get the right to use the computer
program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and
liable to prosecution.

Copyright©2022 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems and the Max Planck Institute for Biological
Cybernetics. All rights reserved.

Contact: ps-license@tuebingen.mpg.de

"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import time
import datetime
import torch
import torchvision.models as models
from torchvision.utils import make_grid
import numpy as np
import cv2
from metro.modeling.bert import BertConfig, METRO
from metro.modeling.bert.modeling_bstro import BSTRO_BodyHSC_Network as BSTRO_Network
from metro.modeling._smpl import SMPL, Mesh
from metro.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from metro.modeling.hrnet.config import config as hrnet_config
from metro.modeling.hrnet.config import update_config as hrnet_update_config
from metro.datasets.build import make_data_loader

from metro.utils.logger import setup_logger
from metro.utils.comm import synchronize, is_main_process, get_rank, all_gather
from metro.utils.miscellaneous import mkdir, set_seed
from metro.utils.metric_logger import AverageMeter, DetMetricsLogger

def save_checkpoint(model, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save, op.join(checkpoint_dir, 'model.bin'))
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir

def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.num_train_epochs/2.0)  ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def mean_per_vertex_error(pred, gt, has_smpl):
    """
    Compute mPVE
    """
    pred = (pred[has_smpl == 1] >= 0.5).float()
    gt = gt[has_smpl == 1]
    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def precision_recall_f1score(pred, gt, has_smpl):
    """
    Compute precision, recall, and f1
    """
    pred = pred[has_smpl == 1]
    gt = gt[has_smpl == 1]

    ### precision, recall, f1 scores are computed first for each mesh, then averaged over the whole batch.
    precision_avg = 0
    recall_avg = 0
    f1_avg = 0
    for b in range(gt.shape[0]):
        tp_num = gt[b, pred[b,:,0] >= 0.5, 0].sum()
        precision_denominator = (pred[b, :, 0] >= 0.5).sum()
        recall_denominator = (gt[b, :, 0]).sum()

        precision_ = tp_num / (precision_denominator + 1e-10)
        recall_ = tp_num / (recall_denominator + 1e-10)
        f1_ = 2 * precision_ * recall_ / (precision_ + recall_ + 1e-10)

        precision_avg += precision_
        recall_avg += recall_
        f1_avg += f1_

    # return precision, recall, f1
    return precision_avg / gt.shape[0], recall_avg / gt.shape[0], f1_avg / gt.shape[0]

def det_error_metric(pred, gt, dist_matrix, has_smpl):
    
    pred = pred[has_smpl == 1]
    gt = gt[has_smpl == 1]

    false_positive_dist_avg = 0
    false_negative_dist_avg = 0
    
    for b in range(gt.shape[0]):
        gt_columns = dist_matrix[:, gt[b, :, 0]==1] if any(gt[b, :, 0]==1) else dist_matrix
        error_matrix = gt_columns[pred[b, :, 0] >= 0.5, :] if any(pred[b, :, 0] >= 0.5) else gt_columns

        false_positive_dist = error_matrix.min(dim=1)[0].mean()
        false_negative_dist = error_matrix.min(dim=0)[0].mean()

        false_positive_dist_avg += false_positive_dist
        false_negative_dist_avg += false_negative_dist

    return false_positive_dist_avg / gt.shape[0], false_negative_dist_avg / gt.shape[0]


def run(args, train_dataloader, val_dataloader, METRO_model, smpl, mesh_sampler):
    smpl.eval()
    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs
    if iters_per_epoch<1000:
        args.logging_steps = 500

    optimizer = torch.optim.Adam(params=list(METRO_model.parameters()),
                                           lr=args.lr,
                                           betas=(0.9, 0.999),
                                           weight_decay=0)

    # define loss function (criterion) and optimizer
    criterion_contact = torch.nn.BCELoss().cuda(args.device)

    if args.distributed:
        METRO_model = torch.nn.parallel.DistributedDataParallel(
            METRO_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

        logger.info(
                ' '.join(
                ['Local rank: {o}', 'Max iteration: {a}', 'iters_per_epoch: {b}','num_train_epochs: {c}',]
                ).format(o=args.local_rank, a=max_iter, b=iters_per_epoch, c=args.num_train_epochs)
            )

    start_training_time = time.time()
    end = time.time()
    METRO_model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    log_loss_contact = AverageMeter()
    log_eval_metrics = DetMetricsLogger()

    for iteration, (img_keys, images, annotations) in enumerate(train_dataloader):

        METRO_model.train()
        iteration += 1
        epoch = iteration // iters_per_epoch
        batch_size = images.size(0)
        adjust_learning_rate(optimizer, epoch, args)
        data_time.update(time.time() - end)

        images = images.cuda(args.device)
        mvm_mask = annotations['mvm_mask'].cuda(args.device)

        # generate simplified mesh
        gt_contact = annotations['contact'].cuda(args.device)
        gt_contact_sub2 = mesh_sampler.downsample(gt_contact, n1=0, n2=2)
        gt_contact_sub = mesh_sampler.downsample(gt_contact)

        mvm_mask_ = mvm_mask.expand(-1,-1,2051)
        meta_masks = torch.cat([mvm_mask_], dim=1)

        # forward-pass
        pred_contact_sub2, pred_contact_sub, pred_contact = METRO_model(images, smpl, mesh_sampler, meta_masks=meta_masks, is_train=True)
        
        loss_contact = ( args.vloss_w_sub2 * criterion_contact(pred_contact_sub2, gt_contact_sub2) + \
                            args.vloss_w_sub * criterion_contact(pred_contact_sub, gt_contact_sub) + \
                            args.vloss_w_full * criterion_contact(pred_contact, gt_contact ) )

        loss = args.vertices_loss_weight*loss_contact
        # update logs
        log_loss_contact.update(loss_contact.item(), batch_size)
        log_losses.update(loss.item(), batch_size)

        # back prop
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % args.logging_steps == 0 or iteration == max_iter:
            eta_seconds = batch_time.avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                ' '.join(
                ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}', 'max mem : {memory:.0f}',]
                ).format(eta=eta_string, ep=epoch, iter=iteration, 
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) 
                + '  loss: {:.4f}, contact loss: {:.4f}, compute: {:.4f}, data: {:.4f}, lr: {:.6f}'.format(
                    log_losses.avg, log_loss_contact.avg, batch_time.avg, data_time.avg, 
                    optimizer.param_groups[0]['lr'])
            )

            visual_imgs, pred_contact_meshes, gt_contact_meshes = visualize_contact(annotations['ori_img'].detach(),
                                                                                    annotations['contact'].detach(),
                                                                                    pred_contact.detach(), 
                                                                                    smpl)
            visual_imgs = visual_imgs.transpose(0,1)
            visual_imgs = visual_imgs.transpose(1,2)
            visual_imgs = np.asarray(visual_imgs)

            if is_main_process()==True:
                stamp = '{:03d}_{:05d}'.format(epoch, iteration)
                foldername = os.path.join(args.output_dir, 'train', stamp)
                if not os.path.exists(foldername):
                    os.makedirs(foldername)
                temp_fname = foldername + '/visual.jpg'
                cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))

                for mi, mesh in enumerate(pred_contact_meshes):
                    temp_fname = foldername + '/visual_{:02d}.obj'.format(mi)
                    mesh.export(temp_fname)

                    temp_fname = foldername + '/visual_{:02d}_gt.obj'.format(mi)
                    gt_contact_meshes[mi].export(temp_fname)

                with open(foldername + '/visual.txt','w') as txt_f:
                    txt_f.writelines(img_keys[:len(pred_contact_meshes)])

        if iteration % iters_per_epoch == 0:
            val_mPVE, val_count, val_precision, val_recall, val_f1, val_fp_error, val_fn_error = run_validate(args, val_dataloader, 
                                                                                                            METRO_model,
                                                                                                            epoch, 
                                                                                                            smpl,
                                                                                                            mesh_sampler)

            logger.info(
                ' '.join(['Validation', 'epoch: {ep}',]).format(ep=epoch) 
                + '  mPVE: {:6.2f}, Data Count: {:6.2f}, precision: {:6.4f}, recall: {:6.4f}, f1: {:6.4f}, fp_error: {:8.6f}, fn_error: {:8.5f},'.format(1000*val_mPVE, \
                    val_count, val_precision, val_recall, val_f1, 100 * val_fp_error, 100 * val_fn_error)
            )

            if val_f1 > log_eval_metrics.f1:
                checkpoint_dir = save_checkpoint(METRO_model, args, epoch, iteration)
                log_eval_metrics.update(mPVE=val_mPVE, p=val_precision, r=val_recall, f1=val_f1, 
                                    fp_error=val_fp_error, fn_error=val_fn_error, epoch=epoch)
                
        
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info('Total training time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_training_time / max_iter)
    )
    checkpoint_dir = save_checkpoint(METRO_model, args, epoch, iteration)

    logger.info(
        ' Best Results:'
        + '  F1: {:6.3f}, at epoch {:6.2f}'.format(log_eval_metrics.f1, log_eval_metrics.epoch)
    )


def run_eval_general(args, val_dataloader, METRO_model, smpl, mesh_sampler):
    smpl.eval()

    if args.distributed:
        METRO_model = torch.nn.parallel.DistributedDataParallel(
            METRO_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    METRO_model.eval()

    val_mPVE, val_count, val_precision, val_recall, val_f1, val_fp_error, val_fn_error  = run_evaluation(args, val_dataloader, 
                                                                                            METRO_model, 
                                                                                            smpl,
                                                                                            mesh_sampler)

    logger.info(
        ' '.join(['Validation', ])
                + '  mPVE: {:6.2f}, Data Count: {:6.2f}, precision: {:6.4f}, recall: {:6.4f}, f1: {:6.4f}, fp_error: {:8.6f}, fn_error: {:8.5f},'.format(1000*val_mPVE, \
                    val_count, val_precision, val_recall, val_f1, 100 * val_fp_error, 100 * val_fn_error)
    )
    
    return

def run_validate(args, val_loader, METRO_model, epoch, smpl, mesh_sampler):
    batch_time = AverageMeter()
    mPVE = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    f1 = AverageMeter()
    fp_error = AverageMeter()
    fn_error = AverageMeter()
    # switch to evaluate mode
    METRO_model.eval()
    smpl.eval()

    dist_matrix = np.load('/ps/project/common/tuch/geodesics/smpl/smpl_neutral_geodesic_dist.npy')
    dist_matrix = torch.tensor(dist_matrix).cuda()
    with torch.no_grad():
        # end = time.time()
        for i, (img_keys, images, annotations) in enumerate(val_loader):

            batch_size = images.size(0)
            # compute output
            images = images.cuda(args.device)
            gt_contact = annotations['contact'].cuda(args.device)
            gt_contact_sub2 = mesh_sampler.downsample(gt_contact, n1=0, n2=2)
            gt_contact_sub = mesh_sampler.downsample(gt_contact)

            # forward-pass
            if METRO_model.config.output_attentions:
                pred_contact_sub2, pred_contact_sub, pred_contact, hidden_states, att = METRO_model(images, smpl, mesh_sampler)
            else:
                pred_contact_sub2, pred_contact_sub, pred_contact = METRO_model(images, smpl, mesh_sampler)

            # measure errors
            has_smpl=torch.ones((pred_contact.shape[0]))
            error_vertices = mean_per_vertex_error(pred_contact, gt_contact, has_smpl)
            precision_, recall_, f1_ = precision_recall_f1score(pred_contact, gt_contact, has_smpl)
            fp_error_, fn_error_ = det_error_metric(pred_contact, gt_contact, dist_matrix, has_smpl)
            # print(i, precision_, recall_, f1_)
            
            if len(error_vertices)>0:
                mPVE.update(np.mean(error_vertices), int(torch.sum(has_smpl)) )

            precision.update(precision_)
            recall.update(recall_)
            f1.update(f1_)
            fp_error.update(fp_error_)
            fn_error.update(fn_error_)

            if i % 500 == 0:
            # print(img_keys, i)
                visual_imgs, pred_contact_meshes, gt_contact_meshes = visualize_contact(annotations['ori_img'].detach(),
                                                                                        annotations['contact'].detach(),
                                                                                        pred_contact.detach(), 
                                                                                        smpl)
                visual_imgs = visual_imgs.transpose(0,1)
                visual_imgs = visual_imgs.transpose(1,2)
                visual_imgs = np.asarray(visual_imgs)

                if is_main_process()==True:
                    seq = img_keys[0].split('/')[-6]
                    stamp = '{:03d}_{:05d}_{}'.format(epoch, i, seq)
                    foldername = os.path.join(args.output_dir, 'val', stamp)
                    if not os.path.exists(foldername):
                        os.makedirs(foldername)
                    temp_fname = foldername + '/visual.jpg'
                    cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))

                    for mi, mesh in enumerate(pred_contact_meshes):
                        img_basename = img_keys[mi].split('/')[-1].split('.')[0]
                        temp_fname = foldername + '/{}_{}.obj'.format(img_basename, mi)
                        mesh.export(temp_fname)


    val_mPVE = all_gather(float(mPVE.avg))
    val_mPVE = sum(val_mPVE)/len(val_mPVE)

    val_precision = all_gather(float(precision.avg))
    val_precision = sum(val_precision)/len(val_precision)

    val_recall = all_gather(float(recall.avg))
    val_recall = sum(val_recall)/len(val_recall)

    val_f1 = all_gather(float(f1.avg))
    val_f1 = sum(val_f1)/len(val_f1)

    val_fp_error = all_gather(float(fp_error.avg))
    val_fp_error = sum(val_fp_error)/len(val_fp_error)

    val_fn_error = all_gather(float(fn_error.avg))
    val_fn_error = sum(val_fn_error)/len(val_fn_error)

    val_count = all_gather(float(mPVE.count))
    val_count = sum(val_count)

    return val_mPVE, val_count, val_precision, val_recall, val_f1, val_fp_error, val_fn_error

def run_evaluation(args, val_loader, METRO_model, smpl, mesh_sampler):
    batch_time = AverageMeter()
    mPVE = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    f1 = AverageMeter()
    fp_error = AverageMeter()
    fn_error = AverageMeter()
    # switch to evaluate mode
    METRO_model.eval()
    smpl.eval()

    seqs = []
    if args.subset_path!='': # evaluate a particular subset
        with open(args.subset_path,'r') as f:
            lines = f.readlines()
        seqs = set([line.strip().split('\t')[0] for line in lines])

    dist_matrix = np.load('/ps/project/common/tuch/geodesics/smpl/smpl_neutral_geodesic_dist.npy')
    dist_matrix = torch.tensor(dist_matrix).cuda()
    with torch.no_grad():
        # end = time.time()
        for i, (img_keys, images, annotations) in enumerate(val_loader):

            if seqs:    # if seqs is not empty --> evalute only the samples from the seq in seqs.
                hit_flag = False
                for seq in seqs:
                    if sum([seq in img_fn for img_fn in img_keys]) >= len(img_keys)/2.0:
                        hit_flag = True
                        break
            else:       # if seqs is empty --> evalute all samples.
                hit_flag = True

            if not hit_flag:
                continue        
            
            batch_size = images.size(0)
            # compute output
            images = images.cuda(args.device)
            gt_contact = annotations['contact'].cuda(args.device)
            gt_contact_sub2 = mesh_sampler.downsample(gt_contact, n1=0, n2=2)
            gt_contact_sub = mesh_sampler.downsample(gt_contact)

            # forward-pass
            if METRO_model.config.output_attentions:
                pred_contact_sub2, pred_contact_sub, pred_contact, hidden_states, att = METRO_model(images, smpl, mesh_sampler)
            else:
                pred_contact_sub2, pred_contact_sub, pred_contact = METRO_model(images, smpl, mesh_sampler)

            # measure errors
            has_smpl=torch.ones((pred_contact.shape[0]))
            error_vertices = mean_per_vertex_error(pred_contact, gt_contact, has_smpl)
            precision_, recall_, f1_ = precision_recall_f1score(pred_contact, gt_contact, has_smpl)
            fp_error_, fn_error_ = det_error_metric(pred_contact, gt_contact, dist_matrix, has_smpl)
            # print(i, precision_, recall_, f1_)
            
            if len(error_vertices)>0:
                mPVE.update(np.mean(error_vertices), int(torch.sum(has_smpl)) )

            precision.update(precision_)
            recall.update(recall_)
            f1.update(f1_)
            fp_error.update(fp_error_)
            fn_error.update(fn_error_)

            if i % 100 == 0:

                visual_imgs, pred_contact_meshes, gt_contact_meshes = visualize_contact(annotations['ori_img'].detach(),
                                                                                        annotations['contact'].detach(),
                                                                                        pred_contact.detach(), 
                                                                                        smpl)
                visual_imgs = visual_imgs.transpose(0,1)
                visual_imgs = visual_imgs.transpose(1,2)
                visual_imgs = np.asarray(visual_imgs)

                if is_main_process()==True:
                    subset = os.path.basename(args.subset_path).split('.')[0] if args.subset_path else 'all'
                    seq = img_keys[0].split('/')[-6]
                    stamp = '{:05d}_{}'.format(i, seq)
                    foldername = os.path.join(args.output_dir, 'test', subset, stamp)
                    if not os.path.exists(foldername):
                        os.makedirs(foldername)
                    temp_fname = foldername + '/visual.jpg'
                    cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))

                    for mi, mesh in enumerate(pred_contact_meshes):
                        img_basename = img_keys[mi].split('/')[-1].split('.')[0]
                        temp_fname = foldername + '/{}_{}.obj'.format(img_basename, mi)
                        mesh.export(temp_fname)


    eval_mPVE = all_gather(float(mPVE.avg))
    eval_mPVE = sum(eval_mPVE)/len(eval_mPVE)

    eval_precision = all_gather(float(precision.avg))
    eval_precision = sum(eval_precision)/len(eval_precision)

    eval_recall = all_gather(float(recall.avg))
    eval_recall = sum(eval_recall)/len(eval_recall)

    eval_f1 = all_gather(float(f1.avg))
    eval_f1 = sum(eval_f1)/len(eval_f1)

    eval_fp_error = all_gather(float(fp_error.avg))
    eval_fp_error = sum(eval_fp_error)/len(eval_fp_error)

    eval_fn_error = all_gather(float(fn_error.avg))
    eval_fn_error = sum(eval_fn_error)/len(eval_fn_error)

    eval_count = all_gather(float(mPVE.count))
    eval_count = sum(eval_count)

    return eval_mPVE, eval_count, eval_precision, eval_recall, eval_f1, eval_fp_error, eval_fn_error

def visualize_contact(images,
                    gt_contact,
                    pred_contact, 
                    smpl):
    gt_contact = gt_contact.cpu()
    ref_vert = smpl(torch.zeros((1, 72)).cuda(args.device), torch.zeros((1,10)).cuda(args.device)).squeeze()
    rend_imgs = []
    gt_contact_meshes = []
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

        ## Ground truths
        contact = gt_contact[i]
        hit_id = (contact == 1).nonzero()[:,0]

        gt_mesh = trimesh.Trimesh(vertices=ref_vert.detach().cpu().numpy(), faces=smpl.faces.detach().cpu().numpy(), process=False)
        gt_mesh.visual.vertex_colors = (191, 191, 191, 255)
        gt_mesh.visual.vertex_colors[hit_id, :] = (0, 255, 0, 255)
        gt_contact_meshes.append(gt_mesh)
        
        # Visualize reconstruction and detected pose
        rend_imgs.append(torch.from_numpy(img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs, pred_contact_meshes, gt_contact_meshes


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--data_dir", default='datasets', type=str, required=False,
                        help="Directory with all datasets, each in one subfolder")
    parser.add_argument("--train_yaml", default='imagenet2012/train.yaml', type=str, required=False,
                        help="Yaml file with all data for training.")
    parser.add_argument("--val_yaml", default='imagenet2012/test.yaml', type=str, required=False,
                        help="Yaml file with all data for validation.")
    parser.add_argument("--test_yaml", default='imagenet2012/test.yaml', type=str, required=False,
                        help="Yaml file with all data for test/eval.")
    parser.add_argument("--num_workers", default=4, type=int, 
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.") 
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='metro/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float, 
                        help="The initial lr.")
    parser.add_argument("--num_train_epochs", default=200, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--vertices_loss_weight", default=100.0, type=float)          
    parser.add_argument("--joints_loss_weight", default=1000.0, type=float)
    parser.add_argument("--vloss_w_full", default=0.33, type=float) 
    parser.add_argument("--vloss_w_sub", default=0.33, type=float) 
    parser.add_argument("--vloss_w_sub2", default=0.33, type=float) 
    parser.add_argument("--drop_out", default=0.1, type=float, 
                        help="Drop out ratio in BERT.")
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
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=False, action='store_true',) 
    parser.add_argument('--logging_steps', type=int, default=1000, 
                        help="Log every X steps.")
    parser.add_argument("--subset_path", default='', type=str, required=False,
                        help="Txt files defining the subset of testset being evaluated.")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=0, 
                        help="For distributed training.")

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
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()


    # Load model
    trans_encoder = []

    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [1]
    
    if args.run_eval_only==True and args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _bstro_network = torch.load(args.resume_checkpoint)
    else:
        # init three transformer-encoder blocks in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, METRO
            config = config_class.from_pretrained(args.config_name if args.config_name \
                    else args.model_name_or_path)

            config.output_attentions = args.output_attentions
            config.hidden_dropout_prob = args.drop_out
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

        if config.output_attentions:
            setattr(_bstro_network.trans_encoder[-1].config,'output_attentions', True)
            setattr(_bstro_network.trans_encoder[-1].config,'output_hidden_states', True)
            _bstro_network.trans_encoder[-1].bert.encoder.output_attentions = True
            _bstro_network.trans_encoder[-1].bert.encoder.output_hidden_states =  True
            for iter_layer in range(4):
                _bstro_network.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
            for inter_block in range(3):
                setattr(_bstro_network.trans_encoder[-1].config,'device', args.device)

        if args.resume_checkpoint!=None and args.resume_checkpoint!='None':
            # for fine-tuning or resume training or inference, load weights from checkpoint
            logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
            cpu_device = torch.device('cpu')
            state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
            if not args.run_eval_only and ('3dpw' in args.resume_checkpoint or 'h36m' in args.resume_checkpoint):
                logger.info('=> initializing with metro weights from {}'.format(args.resume_checkpoint))
                # initializing with METRO pretrained on 3dpw or h36m. Only apply to the backbone.
                state_dict = {k: v for k, v in state_dict.items() if k != 'trans_encoder.2.cls_head.weight' and \
                                                                    k != 'trans_encoder.2.cls_head.bias' and \
                                                                    k != 'trans_encoder.2.residual.weight' and \
                                                                    k != 'trans_encoder.2.residual.bias' and \
                                                                    k != 'conv_learn_tokens.weight' and \
                                                                    k != 'conv_learn_tokens.bias'}
            _bstro_network.load_state_dict(state_dict, strict=False)
            del state_dict
    
    _bstro_network.to(args.device)
    logger.info("Training parameters %s", args)

    if args.run_eval_only==True:
        test_dataloader = make_data_loader(args, args.test_yaml, 
                                        args.distributed, is_train=False, scale_factor=args.img_scale_factor, hsi_flag=True)
        run_eval_general(args, test_dataloader, _bstro_network, smpl, mesh_sampler)

    else:
        train_dataloader = make_data_loader(args, args.train_yaml, 
                                            args.distributed, is_train=True, scale_factor=args.img_scale_factor, hsi_flag=True)
        val_dataloader = make_data_loader(args, args.val_yaml, 
                                        args.distributed, is_train=False, scale_factor=args.img_scale_factor, hsi_flag=True)
        run(args, train_dataloader, val_dataloader, _bstro_network, smpl, mesh_sampler)



if __name__ == "__main__":
    args = parse_args()
    main(args)
