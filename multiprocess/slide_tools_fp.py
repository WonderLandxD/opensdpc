import os
import openslide
import sdpc
import numpy as np
from PIL import Image

from tqdm import tqdm
import cv2
import timm
import glob


# Mask generation by OTSU algorithm
def get_bg_mask(thumbnail, kernel_size=1):
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    ret, threshold = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_OTSU)

    close_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(threshold), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

    return (image_open / 255.0).astype(np.uint8)


def func_patching(args, pair_list, thread_id):
    print(f"Thread {thread_id} is processing {len(pair_list)} WSI.")
    try:
        for item, pair_path in enumerate(pair_list):
            slide_path = pair_path[0]
            save_path = pair_path[1]
            os.makedirs(save_path, exist_ok=True)
            if args.format == 'sdpc':
                slide = sdpc.Sdpc(slide_path)
            else:
                slide = openslide.OpenSlide(slide_path)
            
            thumbnail_level = slide.level_count - args.thumb_n  # Take out thumbnails
            if args.format == 'sdpc':
                thumbnail = slide.read_region((0, 0), thumbnail_level, slide.level_dimensions[thumbnail_level])
            else:
                thumbnail = np.array(slide.read_region((0, 0), thumbnail_level, slide.level_dimensions[thumbnail_level]).convert('RGB'))
            
            black_pixel = np.where((thumbnail[:, :, 0] < 50) & (thumbnail[:, :, 1] < 50) & (thumbnail[:, :, 2] < 50))
            thumbnail[black_pixel] = [255, 255, 255]
            bg_mask = get_bg_mask(thumbnail, kernel_size=args.kernel_size)  # Obtain mask
            zoom_scale = round(slide.level_downsamples[0]) / round(slide.level_downsamples[1])

            x_size = int(args.patch_w / zoom_scale)
            y_size = int(args.patch_h / zoom_scale)
            x_overlap = int(args.overlap_w / zoom_scale)
            y_overlap = int(args.overlap_h / zoom_scale)
            img_x, img_y = slide.level_dimensions[0]

            absolute_coord_list = []
            relative_coord_list = []
            rectangle_list = []
            X_len = int(np.floor((img_x - x_size) / (x_size - x_overlap) + 1))
            Y_len = int(np.floor((img_y - y_size) / (y_size - y_overlap) + 1))
            for i in range(X_len):
                for j in range(Y_len):
                    img_start_x = int(np.floor(i * (x_size - x_overlap) / img_x * bg_mask.shape[1]))
                    img_start_y = int(np.floor(j * (y_size - y_overlap) / img_y * bg_mask.shape[0]))
                    img_end_x = int(np.ceil((i * (x_size - x_overlap) + x_size) / img_x * bg_mask.shape[1]))
                    img_end_y = int(np.ceil((j * (y_size - y_overlap) + y_size) / img_y * bg_mask.shape[0]))
                    mask = bg_mask[img_start_y:img_end_y, img_start_x:img_end_x]

                    if np.sum(mask == 0) / mask.size < args.blank_TH:
                        
                        x_start = int(i * (x_size - x_overlap))
                        y_start = int(j * (y_size - y_overlap))
                        x_offset = int(x_size / pow(1/zoom_scale, args.WSI_level))
                        y_offset = int(y_size / pow(1/zoom_scale, args.WSI_level))

                        relative_coord = (i, j)
                        rectangle = ((x_start, y_start), (x_offset, y_offset))
                        absolute_coord = (x_start, y_start)

                        relative_coord_list.append(relative_coord)
                        absolute_coord_list.append(absolute_coord)
                        rectangle_list.append(rectangle)

            with tqdm(total=len(relative_coord_list), desc=f'THREAD {thread_id} ({item+1} / {len(pair_list)})', position=thread_id, ncols=75) as pbar:
                for idx, absolute_coord in enumerate(absolute_coord_list):
                    try:
                        if args.format == 'sdpc':
                            img = Image.fromarray(slide.read_region(rectangle_list[idx][0], args.WSI_level, rectangle_list[idx][1]))
                        else:
                            img = slide.read_region(rectangle_list[idx][0], args.WSI_level, rectangle_list[idx][1]).convert('RGB')
                        img.save(os.path.join(save_path, f"no{idx:05d}_{absolute_coord[0]:05d}x_{absolute_coord[1]:05d}y.jpg"))
                        pbar.update(1)
                    except:
                        pbar.update(1)
                        continue 
                        
            
            # save thumbnail
            thumbnail_save_path = os.path.join(save_path, 'thumbnail/x20_thumbnail.jpg')
            os.makedirs(os.path.dirname(thumbnail_save_path), exist_ok=True)
            x20_thumbnail = Image.fromarray(thumbnail)
            x20_thumbnail.save(thumbnail_save_path)
    except:
        print(f"Error processing {slide_path} on thread {thread_id}")
