import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
import sdpc


# Mask generation by OTSU algorithm
def get_bg_mask(thumbnail, kernel_size=1):
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    _, threshold = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_OTSU)

    close_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(threshold), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

    return (image_open / 255.0).astype(np.uint8)


parser = argparse.ArgumentParser(description='Code to patch WSI using .sdpc files')
parser.add_argument('--wsi_path', type=str, default=r'G:\SYSFrozenLung_Datasets\version2-slide-available', help='path of .sdpc files')
parser.add_argument('--save_dir', type=str, default=r'G:\SYSFrozenLung_Datasets\version2-patch-256', help='path to store processed tiles')
parser.add_argument('--tile_size', type=int, default=256, help='size for processed tiles')
parser.add_argument('--patch_level', type=int, default=1, help='level for cutting patches')
parser.add_argument('--overlap_size', type=int, default=0, help='overlap size for processed tiles')
parser.add_argument('--blank_TH', type=float, default=0.7, help='cut patches with blank rate lower than blank_TH')

if __name__ == '__main__':

    args = parser.parse_args()
    wsi_path = args.wsi_path
    save_dir = args.save_dir
    tile_size = args.tile_size
    patch_level = args.patch_level
    overlap_size = args.overlap_size
    blank_TH = args.blank_TH

    slide_list = glob.glob(os.path.join(wsi_path, r'*\*.sdpc'))
    idx = 0
    for slide_path in slide_list:
        wsi_name = slide_path.split('\\')[-1].split('.')[0]
        label_name = slide_path.split('\\')[-2]

        wsi = sdpc.Sdpc(slide_path)
        save_path = os.path.join(save_dir, label_name)

        wsi.crop_patches(tile_size=tile_size, overlap_size=overlap_size, patch_level=patch_level,
                         save_dir=save_path, blank_TH=0.7, is_mark=True, is_mask=True, patch_norm=False)

        print(f'{idx} / {len(slide_list)} {((idx / len(slide_list)) * 100):.2f}%')





