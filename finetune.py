from models.PASMnet import *
from datasets.messytable import MessytableDataset
from datasets.messytable_test import MessytableTestDataset_TEST
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils.config import cfg
from net_utils import *
import argparse
from loss import *
import os
from tensorboardX import SummaryWriter
from utils.util import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_disp', type=int, default=0, help='prior maximum disparity, 0 for unavailable')

    parser.add_argument('--savepath', default='log/', help='save path')
    #parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    #parser.add_argument('--batch_size', type=int, default=14)
    #parser.add_argument('--n_workers', type=int, default=2, help='number of threads in dataloader')
    #parser.add_argument('--gamma', type=float, default=0.1)
    #parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs to train')
    #parser.add_argument('--n_steps', type=int, default=60, help='number of epochs to update learning rate')
    parser.add_argument('--resume_model', type=str, default=None)
    parser.add_argument('--print_freq', type=int, default=1, help='the frequency of printing losses (epchs)')
    parser.add_argument('--save_freq', type=int, default=40, help='the frequency of saving models (epochs)')
    parser.add_argument('--config-file', type=str, default='./configs/local_train_steps.yaml',
                        metavar='FILE', help='Config files')

    return parser.parse_args()


def train(train_loader, val_loader, args, cfg, writer):
    net = PASMnet().to(args.device)
    net = nn.DataParallel(net, device_ids=[0])
    net.train()
    cudnn.benchmark = True

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.SOLVER.LR_CASCADE)

    loss_epoch = []
    loss_list = []
    EPE_epoch = []
    D3_epoch = []
    EPE_list = []

    epoch_start = 0

    if args.resume_model is not None:

        ckpt = torch.load(cfg.resume_model)

        if isinstance(net, nn.DataParallel):
            net.module.load_state_dict(ckpt['state_dict'])
        else:
            net.load_state_dict(ckpt['state_dict'])

        epoch_start = ckpt['epoch']
        loss_list = ckpt['loss']
        EPE_list = ckpt['EPE']

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in net.parameters()])))

    for epoch in range(epoch_start, cfg.SOLVER.EPOCHS):
        # lr stepwise
        lr = cfg.SOLVER.LR_CASCADE * (cfg.SOLVER.WEIGHT_DECAY ** (epoch // cfg.SOLVER.LR_STEPS))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for iteration, data in enumerate(train_loader):
            img_left, img_right = data['img_sim_L'].to(args.device), data['img_sim_R'].to(args.device)
            disp_gt = data['img_disp_l'].to(args.device)
            
            #print(disp_gt.shape)
            disp_gt = F.interpolate(disp_gt, scale_factor=0.5, mode='nearest',
                             recompute_scale_factor=False)  # [bs, 1, H, W]
            #print(img_left.shape, disp_gt.shape)

            disp, att, att_cycle, valid_mask = net(img_left, img_right, max_disp=args.max_disp)

            # loss-D
            loss_P = loss_disp_unsupervised(img_left, img_right, disp, F.interpolate(valid_mask[-1][0], scale_factor=4, mode='nearest'))

            # loss-S
            loss_S = loss_disp_smoothness(disp, img_left)

            # loss-PAM
            loss_PAM_P = loss_pam_photometric(img_left, img_right, att, valid_mask)
            loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask)
            loss_PAM_S = loss_pam_smoothness(att)
            loss_PAM = loss_PAM_P + 5 * loss_PAM_S + 5 * loss_PAM_C

            # losses
            loss = loss_P + 0.5 * loss_S + loss_PAM
            loss_epoch.append(loss.data.cpu().item())

            # metrics
            mask = disp_gt > 0
            EPE_iter = EPE_metric(disp, disp_gt, mask)
            EPE_epoch += EPE_iter
            #print(disp.shape[0])
            for i in range(disp.shape[0]):
                D3_epoch += D1_metric(disp[i, :, :, :].unsqueeze(0), disp_gt[i, :, :, :].unsqueeze(0), mask[i, :, :, :].unsqueeze(0), 3)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step = epoch * len(train_loader) + iteration
            if global_step % args.print_freq == 0:
                print('Epoch----%d, Iter----%d, loss---%f, EPE---%f, D3---%f' %
                    (epoch + 1,
                    iteration,
                    float(loss_epoch[len(loss_epoch)-1]),
                    float(EPE_epoch[len(loss_epoch)-1]),
                    float(D3_epoch[len(loss_epoch)-1])))

                save_scalars(writer, 'train', {
                    'net_loss': loss.item(), 
                    'EPE': np.array(EPE_iter).mean().item(), 
                    'D3': D3_epoch[len(D3_epoch)-1].item()}, global_step)

                save_images(writer, 'train', {'img_L':[data['img_sim_L'].detach().cpu()]}, global_step)   
                save_images(writer, 'train', {'img_R':[data['img_sim_R'].detach().cpu()]}, global_step)
                save_images(writer, 'train', {'disp_gt':[data['img_disp_l'].detach().cpu()]}, global_step)   
                #print(preds.items())
                save_images(writer, 'train', {'disp_pred': [disp.detach().cpu()]}, global_step)
        
        # print
        #print(loss_epoch[0], type(loss_epoch[0]))
        #print(len(D3_epoch))
        print('Epoch----%5d, loss---%f, EPE---%f, D3---%f' %
            (epoch + 1,
            float(np.array(loss_epoch).mean()),
            float(np.array(EPE_epoch).mean()),
            float(np.array(D3_epoch).mean())))

        if (epoch+1) % args.save_freq == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            EPE_list.append(float(np.array(EPE_epoch).mean()))

            filename = 'PASMnet_epoch' + str(epoch + 1) + '.pth.tar'

            path = os.path.join(args.savepath, 'checkpoint')
            if not os.path.isdir(path):
                os.mkdir(path)
            save_ckpt({
                'epoch': epoch + 1,
                'state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
                'loss': loss_list,
                'EPE': EPE_list
            }, save_path=path, filename=filename)

            loss_epoch = []
            EPE_epoch = []
            D3_epoch = []


def main(args, cfg):
    train_set = MessytableDataset(cfg.SPLIT.TRAIN, gaussian_blur=False, color_jitter=False, debug=False, sub=1)
    val_set = MessytableDataset(cfg.SPLIT.VAL, gaussian_blur=False, color_jitter=False, debug=False, sub=1, isVal=True)

    train_loader = DataLoader(
        train_set,
        cfg.SOLVER.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.SOLVER.NUM_WORKER,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_set,
        cfg.SOLVER.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.SOLVER.NUM_WORKER,
        pin_memory=False,
    )

    writer = SummaryWriter(args.savepath)

    train(train_loader, val_loader, args, cfg, writer)

if __name__ == '__main__':
    args = parse_args()
    cfg.merge_from_file(args.config_file)

    main(args, cfg)

