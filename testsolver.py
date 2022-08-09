import time

import torch
from torch.autograd import Variable
from model.model import net
from model.PAPS import edge_enhance_multi
from utils.utils import calculate_psnr, calculate_ssim, cc, sam, ergas, get_edge, array2raster, denorm
from data.data import get_test_data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from args import args
import numpy as np
import os

def get_metrics(img1, img2):
    # input: img1 {the pan-sharpened image}, img2 {the ground-truth image}
    # return: (larger better) psnr, ssim, scc, (smaller better) sam, ergas
    m1 = calculate_psnr(img1, img2, 1.)
    m2 = calculate_ssim(img1, img2, 11, 'mean', 1.)
    m3 = cc(img1, img2)
    m4 = sam(img1, img2)
    m5 = ergas(img1, img2)
    return [m1.item(), m2.item(), m3.item(), m4.item(), m5.item()]

class Testsolver(object):
    def __init__(self):
        self.batch_size = 1
        # self.model1 = net()
        self.model2 = edge_enhance_multi(channels=32, num_of_layers=8)

        self.testDataset = get_test_data(args.testdata_dir)
        self.test_dataloader = DataLoader(self.testDataset, batch_size=self.batch_size, shuffle=False,
                                          num_workers=6)

    def check(self):
        self.cuda = args.cuda
        if self.cuda:
            # self.model1_path = args.model1_path
            # self.model1 = self.model1.cuda()
            # self.epoch1 = torch.load(self.model1_path, map_location=lambda storage, loc: storage)['epoch']
            # self.model1.load_state_dict(torch.load(self.model1_path, map_location=lambda storage, loc: storage)['net'])

            self.model2 = self.model2.cuda()
            self.epoch2 = torch.load(args.model2_path, map_location=lambda storage, loc: storage)['epoch']
            self.model2.load_state_dict(torch.load(args.model2_path, map_location=lambda storage, loc: storage)['net'])

        else:
            # self.epoch1 = torch.load(args.model1_path, map_location=lambda storage, loc: storage)['epoch']
            # self.model1.load_state_dict(torch.load(args.model1_path, map_location=lambda storage, loc: storage)['net'])
            self.epoch2 = torch.load(args.model2_path, map_location=lambda storage, loc: storage)['epoch']
            self.model2.load_state_dict(
                torch.load(args.model2_path, map_location=lambda storage, loc: storage)['model'])

    def test_all(self):
        count = 0
        # self.model1.eval()
        self.model2.eval()
        avg_time = []
        psnr_list, ssim_list, cc_list, sam_list, ergas_list = [], [], [], [], []
        test_record_name = '/test.txt'
        open_type = "a+" if os.path.exists(os.path.join(args.record_dir + test_record_name)) else "w"
        test_record = open(os.path.join(args.record_dir + test_record_name), open_type)
        for batch in self.test_dataloader:
            count += 1
            img_pan, img_lr, img_lr_u, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])

            time0 = time.time()
            # 把数据集中提取出来的数据放到device上去
            if self.cuda:
                img_pan = img_pan.cuda()
                img_lr = img_lr.cuda()
                img_lr_u = img_lr_u.cuda()
                target = target.cuda()
            batch_psnr, batch_ssim, batch_cc, batch_sam, batch_ergas = [], [], [], [], []
            metrics = torch.zeros(5)
            with torch.no_grad():
                test_fused_images = self.model2(img_pan, img_lr)  # 网络输出
            time1 = time.time()
            print('==>Save the test_fused_images')
            metrics[:] = torch.Tensor(get_metrics(test_fused_images, target))
            batch_psnr.append(metrics[0].cpu())
            batch_ssim.append(metrics[1].cpu())
            batch_cc.append(metrics[2].cpu())
            batch_sam.append(metrics[3].cpu())
            batch_ergas.append(metrics[4].cpu())
            psnr_list.extend(batch_psnr)
            ssim_list.extend(batch_ssim)
            cc_list.extend(batch_cc)
            sam_list.extend(batch_sam)
            ergas_list.extend(batch_ergas)
            # test_record.write(
            #     "Batch:{} PSNR: {:.10f}, SSIM: {:.10f}, CC: {:.10f}, SAM: {:.10f}, ERGAS: {:.10f}\n".format(count,
            #                                                                                                 np.array(batch_psnr).mean(),
            #                                                                                                 np.array(batch_ssim).mean(),
            #                                                                                                 np.array(batch_cc).mean(),
            #                                                                                                 np.array(batch_sam).mean(),
            #                                                                                                 np.array(batch_ergas).mean()))
            test_fused_images = test_fused_images.cpu()
            self.test_img_save(test_fused_images, 'test_fused_images', self.epoch2, count)

            print("===> Save the test_fused_images || Timer: %.4f sec." % (time1 - time0))
            avg_time.append(time1 - time0)
        test_record.write("AVG Timer: %.4f sec." % (np.mean(avg_time)))
        test_record.write(
            "Test avg metrics:\n PSNR: {:.10f}, SSIM: {:.10f}, CC: {:.10f}, SAM: {:.10f}, ERGAS: {:.10f}\n".format(
                np.array(psnr_list).mean(),
                np.array(ssim_list).mean(),
                np.array(cc_list).mean(),
                np.array(sam_list).mean(),
                np.array(ergas_list).mean()))
        test_record.close()
        print("===> AVG Timer: %.4f sec." % (np.mean(avg_time)))

    def test_img_save(self, x, name, epoch, count):
        x = x.detach().numpy()  # [batch_size, channel, h, w]
        x = np.transpose(x, (0, 2, 3, 1))
        test_dir = args.output_dir + '/GF2/PAPS/block_12/'
        real_dir = args.output_dir + '/GF2/PAPS/real/'
        lrms_dir = args.output_dir + '/GF2/PAPS/LRMS/'
        pan_dir = args.output_dir + '/GF2/PAPS/PAN/'
        if not os.path.exists(real_dir):
            os.makedirs(real_dir)
            os.makedirs(lrms_dir)
            os.makedirs(pan_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        if name == 'test_fused_images':
            array2raster(os.path.join(test_dir, 'test_fused_images_{}.tif'.format(count)),
                         [0, 0], 2.4, 2.4, denorm(x[0].transpose(2, 0, 1)), 4)
        elif name == 'real_images':
            array2raster(os.path.join(real_dir, 'test_fused_images_{}.tif'.format(count)),
                         [0, 0], 2.4, 2.4, denorm(x[0].transpose(2, 0, 1)), 4)
        elif name == 'test_pan_images':
            array2raster(os.path.join(pan_dir, 'test_fused_images_{}.tif'.format(count)),
                         [0, 0], 2.4, 2.4, denorm(x[0].reshape(x.shape[1], x.shape[2])), 1)
        else:
            array2raster(os.path.join(lrms_dir, 'test_fused_images_{}.tif'.format(count)),
                         [0, 0], 2.4, 2.4, denorm(x[0].transpose(2, 0, 1)), 4)

