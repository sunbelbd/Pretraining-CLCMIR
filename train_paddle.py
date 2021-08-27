# -----------------------------------------------------------
# Part of the implementation is based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Training script"""

import argparse
import logging
import numpy
import os
import paddle
import pprint
# import torch
import random
import shutil
import tensorboard_logger as tb_logger
import time
from paddle.distributed import fleet, get_rank

# sys.path.append('/mnt/home/hongliangfei/research/vision-bert-scan-slurm/paddle_version')
from paddle_version.evaluation import i2t, t2i, AverageMeter, encode_data, shard_xattn_t2i_model
from paddle_version.finetune.dataset.build import make_dataloader as make_dataloader_ft
from paddle_version.finetune.function.config import config as config_ft, update_config as update_config_ft
from paddle_version.model import SCAN
from paddle_version.pretrain.dataset.build import make_dataloaders, make_dataloader
from paddle_version.pretrain.dataset.multi_task_dataloader import MultiTaskDataLoader
from paddle_version.pretrain.function.config import config, update_config


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='pretrain',
                        help='pretrain,retrieval')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=150, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--grad_clip', default=5., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=.0001, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=10, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--log_step', default=100, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=100000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--agg_func', default='Mean', type=str,
                        help='Aggregation function')
    parser.add_argument('--logger_name', default='./runs/runX/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/',
                        help='Path to save the model.')
    parser.add_argument('--auto_resume', action='store_true',
                        help='automatic resume training')
    parser.add_argument('--mlm', action='store_true',
                        help='mask language modeling.')
    parser.add_argument('--mrm', action='store_true',
                        help='mask region modeling.')
    parser.add_argument('--cm', action='store_true',
                        help='contrast modeling.')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Use Slurm for training.')
    parser.add_argument('--fp16', action='store_true',
                        help='Use fp16 training.')
    parser.add_argument('--aux_txt_mlm', action='store_true',
                        help='Auxillary text mask language modeling.')
    parser.add_argument('--aux_t2t_recovery', action='store_true',
                        help='Auxillary text to text recovery.')
    parser.add_argument('--i2t_recovery', action='store_true',
                        help='Image to text recovery.')
    parser.add_argument('--cfg', type=str, help='path to config file')
    parser.add_argument('--cfg_ft', type=str, help='path to finetune config file')

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    opt = parser.parse_args()
    pprint.pprint(opt)
    if opt.cfg is not None:
        update_config(opt.cfg)
    # pprint.pprint(config)
    if opt.cfg_ft is not None:
        update_config_ft(opt.cfg_ft)
    # pprint.pprint(config_ft)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    if opt.multi_gpu:
        strategy = fleet.DistributedStrategy()
        # strategy.find_unused_parameters = True
        fleet.init(is_collective=True, strategy=strategy)
        opt.is_master = get_rank() == 0
    else:
        # single gpu. We don't consider CPU case.
        opt.is_master = True

    # manually set random seed
    if config.RNG_SEED > -1:
        random.seed(config.RNG_SEED)
        numpy.random.seed(config.RNG_SEED)
        paddle.seed(config.RNG_SEED)
        # paddle.cuda.manual_seed_all(config.RNG_SEED)

    # Construct the model
    model = SCAN(opt)  # .half()
    params = model.params
    clip = paddle.nn.ClipGradByNorm(clip_norm=5.0)
    optimizer = paddle.optimizer.Adam(learning_rate=opt.learning_rate, parameters=params, grad_clip=clip)

    if opt.multi_gpu:
        print("Using paddle.distributed.fleet ...")
        model.txt_enc = fleet.distributed_model(model.txt_enc)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)

    scaler = paddle.amp.GradScaler() if opt.fp16 else None

    # Load data loaders
    if opt.task == "pretrain":
        # train_loader = datanew64c32bertccmlm_mrm.get_loaders(
        #     opt.data_name, opt.batch_size * 16, opt.workers, opt)
        # Build multi-task data loader. First a list of data loaders covering img-cap data, mono and parallel text data
        if isinstance(config.DATASET, list):
            train_loaders = make_dataloaders(config, mode='train', distributed=True)
            # Extract train loader
            train_loader = MultiTaskDataLoader(train_loaders)
        else:
            train_loader = make_dataloader(config, mode='train', distributed=True)
        # val_loader_ft = datanew64c32bert.get_test_loader('test', 'coco', opt.batch_size, opt.workers, opt)
        val_loader_ft = make_dataloader_ft(config_ft, mode='test', distributed=False)
    else:
        # train_loader_ft, val_loader_ft = datanew64c32bert.get_loaders(
        #     opt.data_name, opt.batch_size, opt.workers, opt)
        train_loader_ft = make_dataloader_ft(config_ft, mode='train', distributed=True)
        val_loader_ft = make_dataloader_ft(config_ft, mode='test', distributed=False)

    # Smart resume from the latest checkpoint
    start_epoch = 0
    best_rsum = 0
    if opt.auto_resume:
        # very important； mapping saved model in cuda:0 to other devices
        #  map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        for epoch in range(opt.num_epochs, start_epoch, -1):
            model_filename = opt.model_name + '/checkpoint_{}.pdparams'.format(epoch - 1)
            if os.path.exists(model_filename):
                print("=> loading checkpoint '{}'".format(model_filename))
                # checkpoint = torch.load(model_filename)
                checkpoint = paddle.load(model_filename)
                start_epoch = checkpoint['epoch']
                best_rsum = checkpoint['best_rsum']
                model.load_state_dict(checkpoint['model'])
                # Eiters is used to show logs as the continuation of another
                # training
                model.Eiters = checkpoint['Eiters']
                print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                      .format(model_filename, start_epoch, best_rsum))
                # validate(opt, val_loader_ft, model)
                break

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        # print(opt.logger_name)
        # print(opt.model_name)
        # if epoch == 0:
        #    validate(opt, val_loader, model)
        adjust_learning_rate(opt, optimizer, epoch)
        # train for one epoch
        # CC data: MLM obj
        if opt.task == "pretrain":
            trainMLM(opt, train_loader, model, epoch, optimizer, scaler)
        else:
            train(opt, train_loader_ft, model, epoch, optimizer, scaler)
        # trainMLM(opt, train_loader, model, epoch, val_loader)
        # coco data: ranking obj, can switch to MLM as well.
        is_best = False
        if opt.task != "pretrain" and ((0 < epoch < 10 and epoch % 5 == 0) or (epoch >= 10 and epoch % 2 == 0)):
            # evaluate on validation set
            print("Val at epoch %d" % epoch)
            rsum = validate(opt, val_loader_ft, model)
            # remember best R@ sum and save checkpoint
            if rsum > best_rsum:
                is_best = True
                best_rsum = rsum
        if opt.is_master:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint_{}.pdparams'.format(epoch), prefix=opt.model_name + '/')


def train(opt, train_loader, model, epoch, optimizer, scaler):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # train_logger = LogCollector()
    # model.logger = train_logger
    # model.txt_enc.logger = train_logger

    end = time.time()
    for i, train_data in enumerate(train_loader):
        model.train_start()
        # measure data loading time
        data_time.update(time.time() - end)
        # make sure train logger is used
        # Update the model
        model.logger.update('lr', optimizer.get_lr())
        # if opt.gpu_id != -1:
        #     train_data = train_data.cuda()
        loss = model.train_emb(*train_data)
        # optimizer.zero_grad()
        if opt.fp16:
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()  # do backward
            scaler.minimize(optimizer, scaled_loss)
        else:
            loss.backward()
            optimizer.step()
        optimizer.clear_grad()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


def trainMLM(opt, train_loader, model, epoch, optimizer, scaler):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # train_logger = LogCollector()
    # model.logger = train_logger
    # model.txt_enc.logger = train_logger

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        # validate(opt, val_loader, model)
        model.train_start()
        # measure data loading time
        data_time.update(time.time() - end)
        # make sure train logger is used
        # model.logger = train_logger
        # Update the model
        model.logger.update('lr', optimizer.get_lr())
        # if torch.cuda.is_available():
        #     train_data = to_cuda(train_data)
        loss = model.train_embMLM(*train_data)
        if opt.fp16:
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()  # do backward
            scaler.minimize(optimizer, scaled_loss)
        else:
            loss.backward()
            optimizer.step()
        optimizer.clear_grad()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, cap_masks, img_masks = encode_data(model, val_loader, opt.log_step, logging.info)

    # img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    img_embs = img_embs[0::5]
    # img_masks = numpy.array([img_masks[i] for i in range(0, len(img_masks), 5)])
    img_masks = img_masks[0::5]

    start = time.time()
    sims = shard_xattn_t2i_model(model, img_embs, cap_embs, cap_masks, img_masks, opt,
                                 shard_size=config_ft.TEST.BATCH_IMAGES)

    end = time.time()
    print("calculate similarity time:", end - start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pdparams', prefix=''):
    tries = 15
    error = None
    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            paddle.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model_best.pdparams')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.75 ** (epoch // opt.lr_update))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    optimizer.set_lr(lr)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape((1, -1)).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape((-1,)).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
