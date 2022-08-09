import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
import time, shutil
import pytorch_msssim
from tqdm import trange
from torch.optim import Adam, SGD
from torch.autograd import Variable
import gdal, ogr, os, osr
import cv2
from torch.utils.tensorboard import SummaryWriter
from model.PAPS import edge_enhance_multi
from data.data import get_train_data, get_eval_data
import utils.utils as utils


from args import args

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    solver = Solver_edge()
    solver.train()

class Solver_edge(object):
    def __init__(self):
        super(Solver_edge, self).__init__()
        self.nEpochs = args.epochs
        self.checkpoint = args.checkpoint_dir
        self.batch_size = args.batch_size
        self.timestamp = int(time.time())
        self.epoch = 1
        self.lr = args.lr
        # self.model1 = net()
        self.model = edge_enhance_multi(channels=32, num_of_layers=8)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.train_dataset = get_train_data(args.traindata_dir)
        self.train_loader = DataLoader(self.train_dataset, args.batch_size, shuffle=False,
                                       num_workers=6)
        self.eval_dataset = get_eval_data(args.evaldata_dir)
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=1, shuffle=False,
                                      num_workers=6)
        self.writer = SummaryWriter(args.log_dir + '/edge_enhance/o_SSIM')
        self.records = {'Epoch': [], 'Loss': [], 'PSNR': [], 'SSIM': []}

    def train(self):
        # print(self.model)
        # exit()
        if not os.path.exists(os.path.join(args.log_dir, 'edge_enhance')):
            os.makedirs(os.path.join(args.log_dir, 'edge_enhance'))
        if not os.path.exists(os.path.join(args.record_dir, 'edge_enhance')):
            os.makedirs(os.path.join(args.record_dir, 'edge_enhance'))
        if not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint)
        if not os.path.exists(os.path.join(args.checkpoint_backup_dir, 'edge_enhance')):
            os.makedirs(os.path.join(args.checkpoint_backup_dir, 'edge_enhance'))
        self.log_dir = args.log_dir + '/edge_enhance'
        self.record_dir = args.record_dir + '/edge_enhance'
        self.checkpoint_backup_dir = args.checkpoint_backup_dir + '/edge_enhance'
        self.open_type = "w" if os.path.exists(self.record_dir + '/train_loss_record.txt') else "w"
        self.train_loss_record = open('%s/train_loss_record.txt' % self.record_dir, self.open_type)
        self.epoch_time_record = open('%s/epoch_time_record.txt' % self.record_dir, self.open_type)

        # xx
        self.check_pretrained(args.edge_enhance_multi_pretrain_model)
        if args.cuda:
            self.model.cuda()
            # self.model1_path = args.model1_path
            # self.model1 = self.model1.cuda()
            # self.model1.load_state_dict(torch.load(self.model1_path, map_location=lambda storage, loc: storage)['net'])
        self.model.train()
        print(self.model)
        loss = torch.nn.MSELoss().cuda()
        ssim_loss = pytorch_msssim.msssim
        tbar = trange(self.nEpochs)
        print('[*]Start training...')
        print('train:', len(self.train_dataset), '\neval:', len(self.eval_dataset))
        step = 0
        time_sum = 0

        steps_per_epoch = len(self.train_loader)
        total_iterations = self.nEpochs * steps_per_epoch
        print('[*] steps_per_epoch:', steps_per_epoch)
        print('[*] total_iters:', total_iterations)

        while self.epoch <= self.nEpochs:
            start = time.time()
            self.lr = args.lr
            self.lr = self.adjust_learning_rate(self.lr, self.epoch, args.lr_decay_freq)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
            print('[*] Epoch = %d \t lr = %.10f' % (self.epoch, self.optimizer.param_groups[0]["lr"]))

            for i, batch in enumerate(self.train_loader):
                step += 1
                # if i >= 10:
                #     break
                pan_img, lr_img, lr_u_img, ms_img = batch[0], batch[1], batch[2], batch[3]
                # 对数据操作
                if args.cuda:
                    ms_img, pan_img, lr_img, lr_u_img = ms_img.cuda(), pan_img.cuda(), lr_img.cuda(), lr_u_img.cuda()
                ## 网络输出
                # model1_img = self.model1(pan_img, lr_img)
                outputs = self.model(pan_img, lr_img)
                # ssim = ssim_loss(outputs, ms_img, normalize=True)
                train_loss = loss(outputs, ms_img) #+ (1 - ssim)
                self.writer.add_scalar('train_loss', train_loss, step)
                # self.writer.add_scalar('ssim_loss', ssim, step)
                if step % 800 == 0:
                    self.writer.add_image('pan', pan_img[0], step)
                    self.writer.add_image('lr', lr_u_img[0], step)
                    self.writer.add_image('output', outputs[0], step)
                    self.writer.add_image('GT', ms_img[0], step)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

            mesg = "{}\tEpoch {}:\t[{}\{}]\t loss: {:.15f}\t \n".format(
                time.ctime(), self.epoch, self.epoch, self.nEpochs,
                train_loss.item(),
            )
            tbar.set_description(mesg)
            tbar.update()
            self.train_loss_record.write(
                "Epoch[{}/{}]: train_loss: {:.15f}\n".format(self.epoch, self.nEpochs, train_loss.item()))

            # xx
            save_model_filename = "edge_enhance_multi.pth"
            save_model_path = os.path.join(self.checkpoint, save_model_filename)
            self.save_checkpoint(save_model_path)
            # xx
            if self.epoch % args.model_backup_freq == 0:
                save_model_filename = "edge_enhance_multi_epochs{}.pth".format(self.epoch)
                save_model_path = os.path.join(self.checkpoint_backup_dir, save_model_filename)
                self.save_checkpoint(save_model_path)

            if self.epoch % args.eval_freq == 0:
                # xx
                checkpoint = torch.load(args.edge_enhance_multi_pretrain_model)
                self.model.load_state_dict(checkpoint['net'])
                print('[*] Eval the model after training {} epochs'.format(self.epoch))
                self.eval()

            time_epoch = (time.time() - start)
            time_sum += time_epoch
            print('[*] No:{} epoch training costs {:.4f}min'.format(self.epoch, time_epoch / 60))
            self.epoch_time_record.write(
                "No:{} epoch training costs {:.4f}min\n".format(self.epoch, time_epoch / 60))
            self.epoch += 1

        self.writer.close()

    def eval(self):
        self.model.eval()
        self.model.cuda()
        open_type = "w" if os.path.exists(os.path.join(self.record_dir + '/eval_loss_record.txt')) else "w"
        eval_loss_record = open('%s/eval_loss_record.txt' % self.record_dir, open_type)
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for k, data in enumerate(self.eval_loader):
                if k == 200:
                    break
                img_pan, img_lr, img_lr_u, target = data[0], data[1], data[2], data[3]
                img_pan = img_pan.cuda()
                img_lr = img_lr.cuda()
                target = target.cuda()
                img_lr_u = img_lr_u.cuda()
                batch_psnr, batch_ssim = [], []
                # 网络输出
                eval_fused_images = self.model(img_pan, img_lr)
                loss = torch.nn.L1Loss()
                ssim = pytorch_msssim.msssim
                eval_loss = loss(eval_fused_images, target)

                psnr = utils.calculate_psnr(eval_fused_images, target, 1.)
                ssim = utils.calculate_ssim(eval_fused_images, target, 11, 'mean', 1.)
                batch_psnr.append(psnr.cpu())
                batch_ssim.append(ssim.cpu())
                avg_psnr = np.array(batch_psnr).mean()
                avg_ssim = np.array(batch_ssim).mean()
                psnr_list.extend(batch_psnr)
                ssim_list.extend(batch_ssim)
                print("===>Batch:{} Eval.loss: {:.10f} PSNR: {:.10f}, SSIM: {:.10f}".format(k + 1, eval_loss.item(),
                                                                                            avg_psnr, avg_ssim))
                eval_loss_record.write(
                    "Batch:{} Eval.loss: {:.10f} PSNR: {:.10f}, SSIM: {:.10f}\n".format(k + 1, eval_loss.item(),
                                                                                        avg_psnr, avg_ssim))

                print('==>Save the fused_images')
                eval_fused_images, real_images = eval_fused_images.cpu(), target.cpu()
                utils.eval_img_save(eval_fused_images, 'eval_fused_images', k, self.epoch)
                utils.eval_img_save(real_images, 'real_images', k, self.epoch)
            self.records['Epoch'].append(self.epoch)
            self.records['PSNR'].append(np.array(psnr_list).mean())
            self.records['SSIM'].append(np.array(ssim_list).mean())

            self.writer.add_scalar('PSNR_epoch', self.records['PSNR'][-1], self.epoch)
            self.writer.add_scalar('SSIM_epoch', self.records['SSIM'][-1], self.epoch)
            eval_loss_record.close()

    def save_checkpoint(self, save_model_path):
        self.ckp = {
            'epoch': self.epoch,
            'records': self.records
        }
        self.ckp['net'] = self.model.state_dict()
        # self.ckp['optimizer'] = self.optimizer.state_dict()
        torch.save(self.ckp, save_model_path)

        if self.records['PSNR'] != [] and self.records['PSNR'][-1] == np.array(self.records['PSNR']).max():
            shutil.copy(save_model_path, os.path.join(args.checkpoint_dir, 'best_model_edge_enhance_multi.pth'))

    def check_pretrained(self, pretrain_model_path):
        if os.path.exists(pretrain_model_path):
            print('Resuming, initializing using weight from {}.'.format(pretrain_model_path))
            ckpt = torch.load(pretrain_model_path)
            self.epoch = ckpt['epoch']
            self.records = ckpt['records']
            self.model.load_state_dict(ckpt['net'])
            print('[*] Reload epoch {} success!'.format(self.epoch))
            if self.epoch > self.nEpochs:
                raise Exception("Pretrain epoch must less than the max epoch!")
        else:
            self.epoch = 1
            print('[*] No pretrained model! Starting training from epoch {}'.format(self.epoch))

    def get_edge(self, data):
        data = data.numpy()
        rs = np.zeros_like(data)
        N = data.shape[0]
        for i in range(N):
            if len(data.shape) == 3:
                rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5),
                                                            normalize=True)  # 第二个参数的-1表示输出图像使用的深度与输入图像相同
            else:
                rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5), normalize=True)
        return torch.from_numpy(rs)

    def adjust_learning_rate(self, lr, epoch, freq):
        lr = lr * (0.1 ** (epoch // freq))
        return lr

if __name__ == "__main__":
    main()