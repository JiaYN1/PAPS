from __future__ import division
import cv2
import os
import gdal, osr
import numpy as np
import random

import sys
sys.path.append('../')
from utils.utils import array2raster

dataDir = '/media/hdr/Elements SE'
satellite = 'GF-2'
# satellite1 = 'WV-2'
blk = 64
ratio = 2
trainIndex = range(3, 10)
testIndex = range(1, 2)
evalIndex = range(2, 3)

if __name__ == '__main__':
    rasterOrigin = (0, 0)
    trainCount = 0
    testCount = 0
    evalCount = 0
    originCount = 0
    path = ["ms", "pan"]
    trainDir = dataDir + '/Dataset/train%d' % blk
    testDir = dataDir + '/Dataset/test_%s' % satellite
    evalDir = dataDir + '/Dataset/eval%d' % blk
    if not os.path.exists(trainDir):
        os.makedirs(trainDir)
    if not os.path.exists(testDir):
        os.makedirs(testDir)
    if not os.path.exists(evalDir):
        os.makedirs(evalDir)

    open_type = "a+" if os.path.exists('./record.txt') else "w"
    record = open('./record.txt', open_type)
    for num in trainIndex:
        mul = '%s/Dataset/%s/prepare_data/%d_ms.tif' % (dataDir, satellite, num)
        lr = '%s/Dataset/%s/prepare_data/%d_lr.tif' % (dataDir, satellite, num)
        lr_u = '%s/Dataset/%s/prepare_data/%d_lr_u.tif' % (dataDir, satellite, num)
        pan = '%s/Dataset/%s/prepare_data/%d_pan.tif' % (dataDir, satellite, num)
        # pan_d = '%s/Dataset/%s/prepare_data/%d_pan_d.tif' % (dataDir, satellite, num)

        dt_mul = gdal.Open(mul)
        dt_lr = gdal.Open(lr)
        dt_lr_u = gdal.Open(lr_u)
        dt_pan = gdal.Open(pan)
        # dt_pan_d = gdal.Open(pan_d)

        img_mul = dt_mul.ReadAsArray()  # (c, h, w)
        img_lr = dt_lr.ReadAsArray()
        img_lr_u = dt_lr_u.ReadAsArray()
        img_pan = dt_pan.ReadAsArray()
        # img_pan_d = dt_pan_d.ReadAsArray()

        XSize = dt_lr.RasterXSize
        YSize = dt_lr.RasterYSize

        sample = int(XSize * YSize / blk / blk * ratio)

        for _ in range(sample):
            x = random.randint(0, XSize - blk)
            y = random.randint(0, YSize - blk)

            array2raster('%s/%d_mul.tif' % (trainDir, trainCount), rasterOrigin, 2.4, 2.4,
                         img_mul[:, y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 4)
            array2raster('%s/%d_lr_u.tif' % (trainDir, trainCount), rasterOrigin, 2.4, 2.4,
                         img_lr_u[:, y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 4)
            array2raster('%s/%d_lr.tif' % (trainDir, trainCount), rasterOrigin, 2.4, 2.4,
                         img_lr[:, y:(y + blk), x:(x + blk)], 4)
            array2raster('%s/%d_pan.tif' % (trainDir, trainCount), rasterOrigin, 2.4, 2.4,
                         img_pan[y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 1)
            # array2raster('%s/%d_pan_d.tif' % (trainDir, trainCount), rasterOrigin, 2.4, 2.4,
            #               img_pan_d[y :(y + blk) , x :(x + blk)], 1)
            trainCount += 1

        print("done %d" % num)

    record.write("%d\n" % trainCount)

    for num in testIndex:
        mul = '%s/Dataset/%s/prepare_data/%d_ms.tif' % (dataDir, satellite, num)
        lr = '%s/Dataset/%s/prepare_data/%d_lr.tif' % (dataDir, satellite, num)
        lr_u = '%s/Dataset/%s/prepare_data/%d_lr_u.tif' % (dataDir, satellite, num)
        pan = '%s/Dataset/%s/prepare_data/%d_pan.tif' % (dataDir, satellite, num)
        # pan_d = '%s/Dataset/%s/prepare_data/%d_pan_d.tif' % (dataDir, satellite, num)

        dt_mul = gdal.Open(mul)
        dt_lr = gdal.Open(lr)
        dt_pan = gdal.Open(pan)
        # dt_pan_d = gdal.Open(pan_d)
        dt_lr_u = gdal.Open(lr_u)

        img_mul = dt_mul.ReadAsArray()
        img_lr = dt_lr.ReadAsArray()
        img_pan = dt_pan.ReadAsArray()
        # img_pan_d = dt_pan_d.ReadAsArray()
        img_lr_u = dt_lr_u.ReadAsArray()

        XSize = dt_lr.RasterXSize
        YSize = dt_lr.RasterYSize

        row = 0
        col = 0

        for y in range(0, YSize, blk):
            if y + blk >= YSize:
                continue
            col = 0

            for x in range(0, XSize, blk):
                if x + blk >= XSize:
                    continue

                array2raster('%s/%d_mul.tif' % (testDir, testCount), rasterOrigin, 2.4, 2.4,
                             img_mul[:, y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 4)
                array2raster('%s/%d_lr_u.tif' % (testDir, testCount), rasterOrigin, 2.4, 2.4,
                             img_lr_u[:, y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 4)
                array2raster('%s/%d_lr.tif' % (testDir, testCount), rasterOrigin, 2.4, 2.4,
                             img_lr[:, y:(y + blk), x:(x + blk)], 4)
                array2raster('%s/%d_pan.tif' % (testDir, testCount), rasterOrigin, 2.4, 2.4,
                             img_pan[y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 1)
                # array2raster('%s/%d_pan_d.tif' % (testDir, testCount), rasterOrigin, 2.4, 2.4,
                #               img_pan_d[y :(y + blk) , x :(x + blk) ], 1)

                testCount += 1
                col += 1

            row += 1
            print(num, row)

        record.write("%d: %d * %d\n" % (num, row, col))

    record.write("%d\n" % testCount)
    record.close()
    print("done")
    
    for num in evalIndex:
        mul = '%s/Dataset/%s/prepare_data/%d_ms.tif' % (dataDir, satellite, num)
        lr = '%s/Dataset/%s/prepare_data/%d_lr.tif' % (dataDir, satellite, num)
        lr_u = '%s/Dataset/%s/prepare_data/%d_lr_u.tif' % (dataDir, satellite, num)
        pan = '%s/Dataset/%s/prepare_data/%d_pan.tif' % (dataDir, satellite, num)
        # pan_d = '%s/Dataset/%s/prepare_data/%d_pan_d.tif' % (dataDir, satellite, num)

        dt_mul = gdal.Open(mul)
        dt_lr = gdal.Open(lr)
        dt_pan = gdal.Open(pan)
        # dt_pan_d = gdal.Open(pan_d)
        dt_lr_u = gdal.Open(lr_u)

        img_mul = dt_mul.ReadAsArray()
        img_lr = dt_lr.ReadAsArray()
        img_pan = dt_pan.ReadAsArray()
        # img_pan_d = dt_pan_d.ReadAsArray()
        img_lr_u = dt_lr_u.ReadAsArray()

        XSize = dt_lr.RasterXSize
        YSize = dt_lr.RasterYSize

        row = 0
        col = 0

        for y in range(0, YSize, blk):
            if y + blk >= YSize:
                continue
            col = 0

            for x in range(0, XSize, blk):
                if x + blk >= XSize:
                    continue

                array2raster('%s/%d_mul.tif' % (evalDir, evalCount), rasterOrigin, 2.4, 2.4,
                             img_mul[:, y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 4)
                array2raster('%s/%d_lr_u.tif' % (evalDir, evalCount), rasterOrigin, 2.4, 2.4,
                             img_lr_u[:, y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 4)
                array2raster('%s/%d_lr.tif' % (evalDir, evalCount), rasterOrigin, 2.4, 2.4,
                             img_lr[:, y:(y + blk), x:(x + blk)], 4)
                array2raster('%s/%d_pan.tif' % (evalDir, evalCount), rasterOrigin, 2.4, 2.4,
                             img_pan[y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 1)
                # array2raster('%s/%d_pan_d.tif' % (testDir, testCount), rasterOrigin, 2.4, 2.4,
                #               img_pan_d[y :(y + blk) , x :(x + blk) ], 1)

                evalCount += 1
                col += 1

            row += 1
            print(num, row)

        record.write("%d: %d * %d\n" % (num, row, col))

    record.write("%d\n" % evalCount)
    record.close()
    print("done")
