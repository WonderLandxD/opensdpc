'''
Multithreaded patch generation for hitopathology whole slide images
Date: 2024/09/22
Author: Jiawen Li
'''

import argparse
import glob
import os
import numpy as np
import csv
from tqdm import tqdm
import multiprocessing

from concurrent.futures import ProcessPoolExecutor, as_completed
from slide_tools_fp import func_patching


def distribute_processing_wsi2patch(pair_list, num_thread, args):
    sub_pair_list = np.array_split(pair_list, num_thread)

    with ProcessPoolExecutor(max_workers=num_thread) as executor:
        futures = {executor.submit(func_patching, args, sublist, i): i for i, sublist in enumerate(sub_pair_list)}

        for future in as_completed(futures):
            thread_index = futures[future]
            try:
                result = future.result()
                print(result)  
            # except:
            #     print(f'Thread {thread_index} generated an exception, continue')
            except Exception as exc:
                print(f'Thread {thread_index} generated an exception: {exc}')


def parse():
    parser = argparse.ArgumentParser(description='Patching for WSI group for Project [SlideLoRA]')

    ##### cropping patches (CPUs) parameters that need to be adjusted #####
    parser.add_argument('--n_thread', type=int, default=8, 
                        help='[MAIN] select the number of threads you want to use') 
    
    ##### patching parameters that need to be adjusted #####
    parser.add_argument('--kernel_size', type=int, default=5, 
                        help='the kernel size of close and open opts for mask')
    parser.add_argument('--patch_w', type=int, default=256,
                        help='the width of patch')
    parser.add_argument('--patch_h', type=int, default=256,
                        help='the height of patch')
    parser.add_argument('--overlap_w', type=int, default=0,
                        help='the overlapped width of patch')
    parser.add_argument('--overlap_h', type=int, default=0,
                        help='the overlapped height of patch')
    parser.add_argument('--blank_TH', type=float, default=0.7, 
                        help='cut patches with blank rate lower than blank_TH')
    parser.add_argument('--WSI_level', type=float, default=1, 
                        help='WSI level for cropping patches, [IMPORTANT] It must be set to [1] !!!')
    parser.add_argument('--thumb_n', type=float, default=1, 
                        help='Get the (slide.level_count - args.thumb_n)th layer thumbnail')
    
    ##### WSI Data parameters #####
    parser.add_argument('--format', type=str, default='sdpc', 
                        choices=['ndpi', 'svs', 'tiff', 'tif', 'mrxs', 'sdpc'], help='select the format of whole slide images')
    
    return parser.parse_args()


if __name__ == '__main__':

    args = parse()
    
    multiprocessing.set_start_method('spawn', force=True)

    '''NOTE: dataset settings, custom design'''
    csv_file_path = ''  # your own csv file

    slide_list = []
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in tqdm(reader):
            slide_list.append(row[0])
    args.save_dir = '/home/ljw/net_data/cervical_fp'
    '''NOTE: dataset settings, custom design'''

    pair_list = []
    for slide_path in tqdm(slide_list):
        slide_name = slide_path.split('/')[-1].split('.')[0]
        save_path = os.path.join(args.save_dir, f'{slide_name}')
        thumb_path = os.path.join(save_path, 'thumbnail/x20_thumbnail.jpg')
        if not os.path.exists(thumb_path):
            pair_list.append([slide_path, save_path])
        else:
            continue

    
    print(f'All data number: {len(slide_list)}, unprocessed data number: {len(pair_list)}')

    num_thread = args.n_thread
    # print(pair_list)
    distribute_processing_wsi2patch(pair_list, num_thread, args)

    







