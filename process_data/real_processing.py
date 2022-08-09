import gdal, osr
import os
import numpy as np
import cv2

indataDir = '/media/hdr/Elements SE/data'
dataDir = '/media/hdr/Elements SE/'
satellite = 'GF-2'

def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array, bandSize):
    if bandSize == 4:
        cols = array.shape[2]
        rows = array.shape[1]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(newRasterfn, cols, rows, 4, gdal.GDT_UInt16)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        for i in range(1, 5):
            outband = outRaster.GetRasterBand(i)
            outband.WriteArray(array[i - 1, :, :])
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()
    elif bandSize == 1:
        cols = array.shape[1]
        rows = array.shape[0]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_UInt16)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(array[:, :])

def downsample(img, ratio=4):
    h, w = img.shape[:2]
    return cv2.resize(img, (w // ratio, h // ratio))

def upsample(img, ratio = 4):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * ratio, h * ratio))

if __name__ == "__main__":
    rasterOrigin = (0, 0)
    inDir = '%s/%s/' % (indataDir, satellite)
    outDir = '%s/Dataset/%s/' % (dataDir, satellite)
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    # MUL -> ms(1) lr(1/4) lr_u(1/4*4)
    for i in range(13, 15):
        newMul = outDir + str(i) + '_ms.tif'
        newLR = outDir + str(i) + '_lr.tif'
        newLR_U = outDir + str(i) + '_lr_u.tif'
        newPan = outDir + str(i) + '_pan.tif'
        # newPan_D = outDir + str(i) + '_pan_d.tif'

        MS_path = inDir + str(i) + '-MUL.TIF'
        PAN_path = inDir + str(i) + '-PAN.TIF'
        # print(MS_path)
        rawMul = gdal.Open(MS_path).ReadAsArray()
        rawPan = gdal.Open(PAN_path).ReadAsArray()
        print("rawMul:", rawMul.shape, " rawPan:", rawPan.shape)

        rawMul = rawMul.transpose(1, 2, 0)
        h, w = rawMul.shape[:2]
        h = h // 4 * 4
        w = w // 4 * 4
    #     # cv2是(w, h)
        imgMul = cv2.resize(rawMul, (w, h))
        imgLR = cv2.resize(imgMul, (w // 4, h // 4))
        imgLR_U = cv2.resize(imgLR, (w, h))
        imgPan = cv2.resize(rawPan, (w, h))
        # imgPan_D = downsample(imgPan)

        imgMul = imgMul.transpose(2, 0, 1)
        # print(imgMul.shape)
        # exit()
        imgLR = imgLR.transpose(2, 0, 1)
        imgLR_U = imgLR_U.transpose(2, 0, 1)

        array2raster(newMul, rasterOrigin, 2.4, 2.4, imgMul, 4)
        array2raster(newLR_U, rasterOrigin, 2.4, 2.4, imgLR_U, 4)  # lr_u
        array2raster(newLR, rasterOrigin, 2.4, 2.4, imgLR, 4)  # lr
        array2raster(newPan, rasterOrigin, 2.4, 2.4, imgPan, 1)  # pan
        # array2raster(newPan_D, rasterOrigin, 2.4, 2.4, imgPan_D, 1)  # pan_d
        # array2raster(newPan, rasterOrigin, 2.4, 2.4, imgPan, 1)
        # array2raster(newPan_D, rasterOrigin, 2.4, 2.4, imgPan_d_u, 1)
        # print('mul:', imgMul.shape, ' pan:', imgPan.shape)
        print('mul:', imgMul.shape, ' lr_u:', imgLR_U.shape,
              ' lr:', imgLR.shape, ' pan:', imgPan.shape)
        # print('pan_d', imgPan_D.shape)
        print('done%s' % i)
