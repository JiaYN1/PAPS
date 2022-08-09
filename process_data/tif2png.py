import os
import cv2
import numpy as np
import gdal
from PIL import Image

hrms_data_path = '' # input_dir
hrms_out_path = '' # output_dir


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
def imgto8bit(img):
    img_norm = normalization(img)
    img_8 = np.uint8(255 * img_norm)
    return img_8

def tif_png(rasterfile):
    in_ds = gdal.Open(rasterfile)

    XSize = in_ds.RasterXSize
    YSize = in_ds.RasterYSize
    bands = in_ds.RasterCount
    if bands == 4:
        B_band = in_ds.GetRasterBand(1)
        B = B_band.ReadAsArray(0, 0, XSize, YSize).astype(np.int16)
        G_band = in_ds.GetRasterBand(2)
        G = G_band.ReadAsArray(0, 0, XSize, YSize).astype(np.int16)
        R_band = in_ds.GetRasterBand(3)
        R = R_band.ReadAsArray(0, 0, XSize, YSize).astype(np.int16)
        IR_band = in_ds.GetRasterBand(4)
        IR = IR_band.ReadAsArray(0, 0, XSize, YSize).astype(np.int16)
        R1 = imgto8bit(R)
        G1 = imgto8bit(G)
        B1 = imgto8bit(B)
        IR1 = imgto8bit(IR)
        img_out = cv2.merge([R1, G1, B1])
        # img_out = cv2.merge([B1, G1, R1])
        # print(img_out.shape)
        # exit()
        img_out = cv2.resize(img_out, (img_out.shape[0], img_out.shape[1]))
        # print(img_out.shape)
        # exit()
    else:
        band = in_ds.GetRasterBand(1)
        out = band.ReadAsArray(0, 0, XSize, YSize).astype(np.int16)
        img_out = imgto8bit(out)
    return img_out

def main():
    if not os.path.exists(hrms_out_path):
        os.makedirs(hrms_out_path)
    classes = os.listdir(hrms_data_path)
    classes.sort()
    for idx, folder in enumerate(classes):
        if folder.endswith('.tif') or folder.endswith('.tiff') or folder.endswith('.TIF'):
            ori_img = os.path.join(hrms_data_path, folder)
            print(ori_img)
            if folder.endswith('.tiff'):
                result_name = os.path.basename(ori_img)[:-5]
            else:
                result_name = os.path.basename(ori_img)[:-4]
            out = hrms_out_path + '/' + result_name + ".png"
            img = tif_png(ori_img)
            cv2.imencode('.png', img)[1].tofile(out)


if __name__ == '__main__':
    main()