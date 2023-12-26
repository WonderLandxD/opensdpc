# 2022/12/24 Version1.0 Completing the slicing of tissue region by leveraging OTSU algorithm

import argparse
import openslide
import os
import queue
import subprocess
import threading
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import sys


####################################################################################################
parser = argparse.ArgumentParser(description='Code to patch WSI using svs/tif/... files')
parser.add_argument('--data_path', type=str,
                    default=r'/data_sdf/CAMELYON17_TIF/training',
                    help='path of svs/tif/... files')
parser.add_argument('--store_path', type=str,
                    default=r'/data_sdf/CAMELYON17_jpgPATCH/train',
                    help='path to store processed patches')

# GPU settings
parser.add_argument('--gpu_num', type=int, default=0, choices=[0, 1], help='Use gpu numbers or not')
parser.add_argument('--free_rate', type=float, default=0.8, help='use gpu with lower free rate (auto choose)')
parser.add_argument('--choose_method', type=str, default='Min', choices=['Order', 'Max', 'Min'],
                    help='auto choose gpu method')
parser.add_argument('--gpu', type=str, default='', help='specify gpu to use (ignore auto choose)')

# common parameters
parser.add_argument('--multiprocess_num', type=int, default=2,
                    help='the number of multiprocess for saving patches')
parser.add_argument('--patch_w', type=int, default=256, help='the width of patch')
parser.add_argument('--patch_h', type=int, default=256, help='the height of patch')
parser.add_argument('--overlap_w', type=int, default=0, help='the overlap width of patch')
parser.add_argument('--overlap_h', type=int, default=0, help='the overlap height of patch')

parser.add_argument('--magnification', type=int, default=40, help='max magnification of the scanner: 40x, 80x, ...')
parser.add_argument('--zoom_scale', type=float, default=0.5, help='zoom_scale: 0-1')

parser.add_argument('--thumbnail_level', type=int, default=3, choices=[1, 2, 3],
                    help='the top level to catch thumbnail images from svs/tif... files')
parser.add_argument('--stain_removal', type=bool, default=True, choices=[False, True],
                    help='whether to remove black stain generated when making slices (e.g. CAMELYON16 dataset) or not')
parser.add_argument('--marked_thumbnail', type=bool, default=True, choices=[False, True],
                    help='0: no produce, 1: produce marked thumbnail')
parser.add_argument('--mask', type=bool, default=True, choices=[False, True],
                    help='0: no produce, 1: produce mask')
parser.add_argument('--kernel_size', type=int, default=5, help='the kernel size of close and open opts for mask')
parser.add_argument('--blank_TH', type=float, default=0.2, help='cut patches with blank rate lower than blank_TH')


####################################################################################################
def get_nvidia_free_gpu(threshold=0.8, method='Order'):
    def _get_pos(command: str, start_pos: int, val: str):
        pos = command[start_pos:].find(val)
        if pos != -1:
            pos += start_pos
        return pos

    query = subprocess.getoutput('nvidia-smi -q -d Memory |grep -A4 GPU')
    free_id = []
    free_dict = {}
    gpu_id = 0
    str_scan_pos = 0

    while str_scan_pos < len(query):
        total_pos = _get_pos(query, str_scan_pos, 'Total')
        comma_pos = _get_pos(query, total_pos + 1, ':')
        _MiB_pos = _get_pos(query, comma_pos + 1, 'MiB')
        try:
            total_memory = int(query[comma_pos + 1:_MiB_pos])
        except:
            break

        _Free_pos = _get_pos(query, _MiB_pos + 1, 'Free')
        comma_pos = _get_pos(query, _Free_pos + 1, ':')
        _MiB_pos = _get_pos(query, comma_pos + 1, 'MiB')
        try:
            free_memory = int(query[comma_pos + 1:_MiB_pos])

        except:
            break
        free_rate = float(free_memory) / float(total_memory)
        if free_rate > threshold:
            free_id.append(gpu_id)
            free_dict.update({gpu_id: free_memory})
            print("GPU:%d, Free:%d, Total:%d, Free rate:%.2f." % (gpu_id, free_memory, total_memory, free_rate))
        else:
            print("GPU:%d, Free:%d, Total:%d, Free rate:%.2f, Unselected." % (
                gpu_id, free_memory, total_memory, free_rate))
        gpu_id += 1
        str_scan_pos = _MiB_pos + 1
    if method == 'Max':
        free_id = sorted([_id for _id in free_dict], key=lambda _id: free_dict[_id], reverse=True)
    elif method == 'Min':
        free_id = sorted([_id for _id in free_dict], key=lambda _id: free_dict[_id], reverse=False)
    elif method == 'Order':
        pass
    else:
        raise Exception('Wrong Choose GPU method!')
    return free_id


def config_nvidia_env(num=1, threshold=0.8, choose_method='Order', *ids):
    str_gpus = []
    if len(ids) != 0 and '' not in ids:
        if isinstance(ids, int):
            str_gpus = [str(ids)]
        elif isinstance(ids, (list, tuple)):
            for _id in ids:
                str_gpus.append(str(_id))
        else:
            raise Exception('Wrong manual gpu id type')
    else:
        if num == 0:
            str_gpus = ['-1']
        else:
            avail_gpus = get_nvidia_free_gpu(threshold, method=choose_method)
            if not avail_gpus:
                raise Exception('No free GPU with memory more than {0}%'.format(100 * threshold))
            n = 0
            for gpu in avail_gpus:
                if n >= num:
                    break
                str_gpus.append(str(gpu))
                n += 1
            if n < num:
                raise Exception('No enough free GPU with memory more than {0}%'.format(100 * threshold))
    env_val = ','.join(str_gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = env_val
    return env_val


# Mask generation by OTSU algorithm
def get_bg_mask(thumbnail, kernel_size=1):
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    ret, threshold = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_OTSU)

    close_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(threshold), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

    return (image_open / 255.0).astype(np.uint8)


def generate_patch_level(args):
    if args.zoom_scale <= 1 / 16:
        args.WSI_level = 2
    elif args.zoom_scale <= 1 / 2:
        args.WSI_level = 1
    else:
        args.WSI_level = 0
    return args


def save_img(img_dict: dict):
    for k, v in img_dict.items():
        v.save(k)


class MultiProcessSave(threading.Thread):
    def run(self):
        event.wait(timeout=10)  # The main function starts reading patches
        # Recurring save
        while True:
            # The queue is empty and the main function stops reading patches
            if q.empty() and not event.is_set():
                break
            else:
                try:
                    img_dict = q.get(block=event.is_set(), timeout=1000)  # Read the first item of the queue
                    # save_img(img_dict, format="JPEG", quality=75)  # save
                    save_img(img_dict)
                    
                except:
                    break


# class main():  # 调试时注释
class main(threading.Thread):
    def run(self):
        args = parser.parse_args()
        if args.gpu_num != 0:
            gpu_env = config_nvidia_env(args.gpu_num, args.free_rate, args.choose_method, args.gpu)
            print('Using GPU ID:', gpu_env)
            args.device = torch.device('cuda')
        else:
            args.device = torch.device('cpu')

        args = generate_patch_level(args)
        files = os.listdir(args.data_path)

        for i, file in enumerate(files):
            fi_path, _ = file.split('.')
            fi_path = os.path.join(args.store_path, fi_path + '_' + str(int(args.magnification * args.zoom_scale)))
            folder = os.path.exists(fi_path)

            if not folder:
                os.makedirs(fi_path)
            else:
                continue

            print('-----------------* Processing: %s *---------------------' % file)
            self.auto_cut(args, file, fi_path)

        event.clear()

    def auto_cut(self, args, file, file_path):
        slide_path = os.path.join(args.data_path, file)
        slide = openslide.OpenSlide(slide_path)

        thumbnail_level = slide.level_count - args.thumbnail_level  # Take out thumbnails
        thumbnail = np.array(slide.read_region((0, 0), thumbnail_level, slide.level_dimensions[thumbnail_level]).convert('RGB'))

        if args.stain_removal:
            black_pixel = np.where((thumbnail[:, :, 0] < 50) & (thumbnail[:, :, 1] < 50) & (thumbnail[:, :, 2] < 50))
            thumbnail[black_pixel] = [255, 255, 255]

        bg_mask = get_bg_mask(thumbnail, kernel_size=args.kernel_size)  # Obtain mask

        if args.mask:
            os.makedirs(os.path.join(file_path, 'thumbnail'), exist_ok=True)
            plt.imshow(bg_mask)
            f = plt.gcf()  # Acquire current image
            f.savefig(os.path.join(file_path, 'thumbnail', 'mask.png'))
            f.clear()  # Release memory

        # Set exported thumbnails with markers
        marked_img = thumbnail.copy()

        # Scaling to the image size of layer 0 in the Whole Slide Image
        x_size = int(args.patch_w / args.zoom_scale)
        y_size = int(args.patch_h / args.zoom_scale)
        x_overlap = int(args.overlap_w / args.zoom_scale)
        y_overlap = int(args.overlap_h / args.zoom_scale)
        img_x, img_y = slide.level_dimensions[0]
        total_num = int(np.floor((img_x - x_size) / (x_size - x_overlap) + 1)) * \
                    int(np.floor((img_y - y_size) / (y_size - y_overlap) + 1))

        with tqdm(total=total_num, file=sys.stdout, colour='blue', desc='Processing', ncols=100) as pbar:
            for i in range(int(np.floor((img_x - x_size) / (x_size - x_overlap) + 1))):
                for j in range(int(np.floor((img_y - y_size) / (y_size - y_overlap) + 1))):
                    img_start_x = int(np.floor(i * (x_size - x_overlap) / img_x * bg_mask.shape[1]))
                    img_start_y = int(np.floor(j * (y_size - y_overlap) / img_y * bg_mask.shape[0]))
                    img_end_x = int(np.ceil((i * (x_size - x_overlap) + x_size) / img_x * bg_mask.shape[1]))
                    img_end_y = int(np.ceil((j * (y_size - y_overlap) + y_size) / img_y * bg_mask.shape[0]))
                    mask = bg_mask[img_start_y:img_end_y, img_start_x:img_end_x]

                    if np.sum(mask == 0) / mask.size < args.blank_TH:
                        cv2.rectangle(marked_img, (img_start_x, img_start_y), (img_end_x, img_end_y), (255, 0, 0), 2)
                        x_start = int(i * (x_size - x_overlap))
                        y_start = int(j * (y_size - y_overlap))
                        x_offset = int(x_size / pow(2, args.WSI_level))
                        y_offset = int(y_size / pow(2, args.WSI_level))
                        img = slide.read_region((x_start, y_start), args.WSI_level, (x_offset, y_offset)).convert('RGB')

                        if rgb_filter(img):
                            save_path = os.path.join(file_path, str(i) + '_' + str(j) + '.jpg')
                            im = process_data(args, img)
                            q.put({save_path: im})
                            event.set()

                    pbar.update(1)

        if args.marked_thumbnail:
            os.makedirs(os.path.join(file_path, 'thumbnail'), exist_ok=True)
            Image.fromarray(marked_img).save(os.path.join(file_path, 'thumbnail', 'thumbnail.png'))
        slide.close()


def process_data(args, data: np.array):
    data.thumbnail((args.patch_w, args.patch_h))
    return data


def rgb_filter(data):
    flag = True
    mean_threshold = 240
    var_threshold = 100
    if np.mean(np.array(data)[:, :, :]) > mean_threshold and np.var(np.array(data)[:, :, :]) < var_threshold:
        flag = False

    return flag


if __name__ == '__main__':
    q = queue.Queue(-1)
    event = threading.Event()
    threads = [main()]
    Process_num = parser.parse_args().multiprocess_num
    for i in range(Process_num):
        threads.append(MultiProcessSave())
    for t in threads:
        t.start()
