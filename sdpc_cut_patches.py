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
parser.add_argument('--wsi_path', type=str, default='', help='path of .sdpc files')
parser.add_argument('--save_path', type=str, default='', help='path to store processed tiles')
parser.add_argument('--tile_size', type=int, default=256, help='size for processed tiles')
parser.add_argument('--thumbnail_level', type=int, default=2, help='thumbnail level')
parser.add_argument('--patch_level', type=int, default=1, help='level for cutting patches')
parser.add_argument('--overlap_size', type=int, default=0, help='overlap size for processed tiles')
parser.add_argument('--blank_TH', type=float, default=0.8, help='cut patches with blank rate lower than blank_TH')


if __name__ == '__main__':

    args = parser.parse_args()
    wsi_path = args.wsi_path
    thumbnail_level = args.thumbnail_level
    save_path = args.save_path
    tile_size = args.tile_size
    patch_level = args.patch_level
    overlap_size = args.overlap_size
    blank_TH = args.blank_TH

    slide_list = glob.glob(os.path.join(wsi_path, '*.sdpc'))
    idx = 0
    for slide_path in slide_list:
        wsi = sdpc.Sdpc(slide_path)
        wsi_name = slide_path.split('/')[-1].split('.')[0]
        zoom_value = wsi.level_downsample[1] / wsi.level_downsample[0]

        thumbnail = np.array(wsi.read_region((0, 0), thumbnail_level, wsi.level_dimensions[thumbnail_level]))

        # Obtain mask & generate mask image
        black_pixel = np.where((thumbnail[:, :, 0] < 50) & (thumbnail[:, :, 1] < 50) & (thumbnail[:, :, 2] < 50))
        thumbnail[black_pixel] = [255, 255, 255]

        bg_mask = get_bg_mask(thumbnail, kernel_size=5)

        os.makedirs(f'{save_path}/{wsi_name}/thumbnail', exist_ok=True)
        cv2.imwrite(f'{save_path}/{wsi_name}/thumbnail/thumbnail.png', thumbnail)
        cv2.imwrite(f'{save_path}/{wsi_name}/thumbnail/mask.png', bg_mask * 255)

        marked_img = thumbnail.copy()

        tile_x = int(tile_size / pow(4, thumbnail_level - patch_level))   # 在patch_level=1的情况下切1000*1000 （20x）  thumbnail 是 patch_level = 2的时候做的
        tile_y = int(tile_size / pow(4, thumbnail_level - patch_level))  # 用于在缩率图中寻找相关位置

        x_overlap = int(overlap_size / pow(4, thumbnail_level - patch_level))
        y_overlap = int(overlap_size / pow(4, thumbnail_level - patch_level))

        thumbnail_x, thumbnail_y = wsi.level_dimensions[thumbnail_level]

        total_num = int(np.floor((thumbnail_x - tile_x) / (tile_x - x_overlap) + 1)) * \
                    int(np.floor((thumbnail_y - tile_y) / (tile_y - y_overlap) + 1))
        
        with tqdm(total=total_num, ncols=100) as pbar:
            for i in range(int(np.floor((thumbnail_x - tile_x) / (tile_x - x_overlap) + 1))):
                for j in range(int(np.floor((thumbnail_y - tile_y) / (tile_y - y_overlap) + 1))):

                    start_x = int(np.floor(i * (tile_x - x_overlap) / thumbnail_x * bg_mask.shape[1]))
                    start_y = int(np.floor(j * (tile_y - y_overlap) / thumbnail_y * bg_mask.shape[0]))

                    end_x = int(np.ceil((i * (tile_x - x_overlap) + tile_x) / thumbnail_x * bg_mask.shape[1]))
                    end_y = int(np.ceil((j * (tile_y - y_overlap) + tile_y) / thumbnail_y * bg_mask.shape[0]))

                    mask = bg_mask[start_y:end_y, start_x:end_x]

                    if np.sum(mask == 0) / mask.size < blank_TH:
                        cv2.rectangle(marked_img, (end_x, end_y), (start_x, start_y), (255, 0, 0), 2)
                        
                        cut_x = int(start_x * pow(4, thumbnail_level))
                        cut_y = int(start_y * pow(4, thumbnail_level))

                        img = wsi.read_region((cut_x, cut_y) , patch_level, (tile_size, tile_size))     

                        save_img = f'{save_path}/{wsi_name}/patch_level-{str(patch_level)}_X-{cut_x}_Y-{cut_y}.png'   # 第0层的坐标X，Y

                        cv2.imwrite(save_img, img)
                    
                    pbar.update(1)

        cv2.imwrite(f'{args.save_path}/{wsi_name}/thumbnail/thumbnail_marked.png', marked_img)
        idx += 1
        print(f'{idx} / {len(slide_list)} {((idx / len(slide_list)) * 100):.2f}%')







