import cv2
import numpy.ctypeslib as npCtypes
import ctypes
from ctypes import *
import gc
import os
import sys
from sdpc.Sdpc_struct import SqSdpcInfo
from PIL import Image
import numpy as np
from sdpc.normalizeStaining import normalizeStaining
import time


if os.name == 'nt':
    # Environment configuration under WINDOWS requires calling .dll file
    dirname, _ = os.path.split(os.path.abspath(__file__))
    os.chdir(os.path.join(dirname, 'WINDOWS\\dll'))
    sys.path.append(os.path.join(dirname, 'WINDOWS\\dll'))
    soPath = os.path.join(dirname, 'WINDOWS\\dll\\DecodeSdpcDll.dll')
elif os.name == 'posix':
    # Environment configuration under LINUX requires calling .so file
    dirname, _ = os.path.split(os.path.abspath(__file__))
    so_lacation = os.path.join(dirname, 'LINUX')
    sys.path.append(os.path.join(dirname, 'LINUX'))
    sys.path.append(os.path.join(dirname, 'LINUX/ffmpeg'))
    sys.path.append(os.path.join(dirname, 'LINUX/jpeg'))
    soPath = os.path.join(dirname, 'LINUX/libDecodeSdpc.so')
    os.environ['LINUX_PATH'] = os.path.join(dirname, 'LINUX')
    os.environ['FFMPEG_PATH'] = os.path.join(dirname, 'LINUX/ffmpeg')
    os.environ['JPEG_PATH'] = os.path.join(dirname, 'LINUX/jpeg')
else:
    raise RuntimeError(f'Unsupported operating system for {os.name}')


# Interface pointer settings
so = ctypes.CDLL(soPath)
so.GetLayerInfo.restype = POINTER(c_char)
so.SqGetRoiRgbOfSpecifyLayer.argtypes = [POINTER(SqSdpcInfo), POINTER(POINTER(c_uint8)),
                                             c_int, c_int, c_uint, c_uint, c_int]
so.SqGetRoiRgbOfSpecifyLayer.restype = c_int
so.SqOpenSdpc.restype = POINTER(SqSdpcInfo)


class Sdpc:
    def __init__(self, sdpcPath):
        self.sdpcPath = sdpcPath
        self.sdpc = self.readSdpc(self.sdpcPath)
        self.level_count = self.getLevelCount()
        self.level_downsamples = self.getLevelDownsamples()
        self.level_dimensions = self.getLevelDimensions()
        self.scan_magnification = self.readSdpc(self.sdpcPath).contents.picHead.contents.rate
        self.sampling_rate = self.readSdpc(self.sdpcPath).contents.picHead.contents.scale

    def getRgb(self, rgbPos, width, height):
        intValue = npCtypes.as_array(rgbPos, (height, width, 3))
        return intValue

    def readSdpc(self, fileName):
        if os.name == 'nt':
            sdpc = so.SqOpenSdpc(c_char_p(bytes(fileName, 'gbk')))
            sdpc.contents.fileName = bytes(fileName, 'gbk')
        else:
            sdpc = so.SqOpenSdpc(c_char_p(bytes(fileName, 'utf-8')))
            sdpc.contents.fileName = bytes(fileName, 'utf-8')
        return sdpc

    def getLevelCount(self):
        return self.sdpc.contents.picHead.contents.hierarchy

    def getLevelDownsamples(self):
        levelCount = self.getLevelCount()
        rate = self.sdpc.contents.picHead.contents.scale
        rate = 1 / rate
        _list = []
        for i in range(levelCount):
            _list.append(rate ** i)
        return tuple(_list)

    def get_best_level_for_downsample(self, downsample):
        preset = [i for i in self.level_downsamples]
        err = [abs(i - downsample) for i in preset]
        level = err.index(min(err))
        return level

    def read_region(self, location, level, size):
        startX, startY = location
        scale = self.level_downsamples[level]
        startX = int(startX / scale)
        startY = int(startY / scale)

        width, height = size

        rgbPos = POINTER(c_uint8)()
        rgbPosPointer = byref(rgbPos)
        so.SqGetRoiRgbOfSpecifyLayer(self.sdpc, rgbPosPointer, width, height, startX, startY, level)
        rgb = self.getRgb(rgbPos, width, height)[..., ::-1]
        rgbCopy = rgb.copy()

        so.Dispose(rgbPos)
        del rgbPos
        del rgbPosPointer
        gc.collect()

        return rgbCopy

    def getLevelDimensions(self):
        def findStrIndex(subStr, str):
            index1 = str.find(subStr)
            index2 = str.find(subStr, index1 + 1)
            index3 = str.find(subStr, index2 + 1)
            index4 = str.find(subStr, index3 + 1)
            return index1, index2, index3, index4

        levelCount = self.getLevelCount()
        levelDimensions = []
        for level in range(levelCount):
            layerInfo = so.GetLayerInfo(self.sdpc, level)
            count = 0
            byteList = []
            while (ord(layerInfo[count]) != 0):
                byteList.append(layerInfo[count])
                count += 1

            strList = [byteValue.decode('utf-8') for byteValue in byteList]
            str = ''.join(strList)

            equal1, equal2, equal3, equal4 = findStrIndex("=", str)
            line1, line2, line3, line4 = findStrIndex("|", str)

            rawWidth = int(str[equal1 + 1:line1])
            rawHeight = int(str[equal2 + 1:line2])
            boundWidth = int(str[equal3 + 1:line3])
            boundHeight = int(str[equal4 + 1:line4])
            w, h = rawWidth - boundWidth, rawHeight - boundHeight
            levelDimensions.append((w, h))
        return tuple(levelDimensions)

    def get_thumbnail(self, thumbnail_level):
        thumbnail = np.array(self.read_region((0, 0), thumbnail_level, self.level_dimensions[thumbnail_level]))
        return thumbnail

    def get_bg_mask(self, thumbnail, kernel_size=1):
        hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
        _, threshold = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_OTSU)

        close_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(threshold), cv2.MORPH_CLOSE, close_kernel)
        open_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

        return (image_open / 255.0).astype(np.uint8)

    def crop_patches(self, tile_size, overlap_size, patch_level, save_dir, blank_TH=0.7, is_mark=True,
                         is_mask=True, patch_norm=False):
            if patch_level + 2 > self.level_count - 1:
                thumbnail_level = self.level_count - 1
            else:
                thumbnail_level = patch_level + 2

            thumbnail = np.array(self.read_region((0, 0), thumbnail_level, self.level_dimensions[thumbnail_level]))

            # Obtain mask & generate mask image
            black_pixel = np.where((thumbnail[:, :, 0] < 50) & (thumbnail[:, :, 1] < 50) & (thumbnail[:, :, 2] < 50))
            thumbnail[black_pixel] = [255, 255, 255]

            bg_mask = self.get_bg_mask(thumbnail, kernel_size=5)

            marked_img = thumbnail.copy()
            zoom_value = round(1 / self.sampling_rate)
            crop_magnification = self.scan_magnification * self.sampling_rate

            wsi_name, _ = os.path.splitext(os.path.basename(self.sdpcPath))
            save_patch = os.path.join(save_dir, wsi_name, f'{round(crop_magnification)}x')
            os.makedirs(save_patch, exist_ok=True)
            if os.listdir(save_patch):
                response = input(
                    "There are files in the folder, do you want to continue cropping? (Y/N): ").strip().upper()
                if response == 'Y':
                    print("Continue...")
                elif response == 'N':
                    print('Cancel...')
                    sys.exit(0)
                else:
                    print("Invalid input, cancel...")
                    sys.exit(0)

            tile_x = int(tile_size / pow(zoom_value,
                                         thumbnail_level - patch_level))  # 在patch_level=1的情况下切1000*1000 （20x）  thumbnail 是 patch_level = 2的时候做的
            tile_y = int(tile_size / pow(zoom_value, thumbnail_level - patch_level))  # 用于在缩率图中寻找相关位置

            x_overlap = int(overlap_size / pow(zoom_value, thumbnail_level - patch_level))
            y_overlap = int(overlap_size / pow(zoom_value, thumbnail_level - patch_level))

            thumbnail_x, thumbnail_y = self.level_dimensions[thumbnail_level]

            total_num = int(np.floor((thumbnail_x - tile_x) / (tile_x - x_overlap) + 1)) * \
                        int(np.floor((thumbnail_y - tile_y) / (tile_y - y_overlap) + 1))

            idx = 0
            start_time = time.time()
            print(f'Start croping patches...Normalized patches is {str(patch_norm)}')
            for i in range(int(np.floor((thumbnail_x - tile_x) / (tile_x - x_overlap) + 1))):
                for j in range(int(np.floor((thumbnail_y - tile_y) / (tile_y - y_overlap) + 1))):

                    start_x = int(np.floor(i * (tile_x - x_overlap) / thumbnail_x * bg_mask.shape[1]))
                    start_y = int(np.floor(j * (tile_y - y_overlap) / thumbnail_y * bg_mask.shape[0]))

                    end_x = int(np.ceil((i * (tile_x - x_overlap) + tile_x) / thumbnail_x * bg_mask.shape[1]))
                    end_y = int(np.ceil((j * (tile_y - y_overlap) + tile_y) / thumbnail_y * bg_mask.shape[0]))

                    mask = bg_mask[start_y:end_y, start_x:end_x]

                    if np.sum(mask == 0) / mask.size < blank_TH:
                        cv2.rectangle(marked_img, (end_x, end_y), (start_x, start_y), (255, 0, 0), 2)

                        cut_x = int(start_x * pow(zoom_value, thumbnail_level))
                        cut_y = int(start_y * pow(zoom_value, thumbnail_level))

                        img_np = np.array(self.read_region((cut_x, cut_y), patch_level, (tile_size, tile_size)))  # numpy format

                        if patch_norm == True:
                            Inorm, H, E = normalizeStaining(img=img_np, Io=240, alpha=1, beta=0.15)  # all numpy
                            img = Image.fromarray(Inorm)
                        else:
                            img = Image.fromarray(img_np)

                        img.save(f'{save_patch}/{i}_{j}.jpg', format='JPEG')

                        elapsed_time = time.time() - start_time
                        elapsed_minutes = int(elapsed_time // 60)
                        elapsed_seconds = int(elapsed_time % 60)

                        print(
                            f"[{wsi_name}] Processing: {idx + 1}/{total_num}, --- {elapsed_minutes} min {elapsed_seconds} s ",
                            end='\r')
                        idx += 1

            end_time = time.time() - start_time
            end_minutes = int(end_time // 60)
            end_seconds = int(end_time % 60)
            ratio = (idx / total_num) * 100
            print(f'\nFinish [{wsi_name}]......Saving path: {save_patch}')
            print(
                f'The total number of patches: {idx} ({ratio:.2f}% of {total_num}) TIME: {end_minutes} min {end_seconds} s')

            if is_mark == True:
                os.makedirs(f'{save_patch}/thumbnail', exist_ok=True)
                marked_img = Image.fromarray(marked_img)
                marked_img.save(f'{save_patch}/thumbnail/mark_level{thumbnail_level}.jpg', format='JPEG')
                print('Saved marked thumbnail....')

            if is_mask == True:
                os.makedirs(f'{save_patch}/thumbnail', exist_ok=True)
                mask_img = Image.fromarray(bg_mask * 255)
                mask_img.save(f'{save_patch}/thumbnail/mask_level{thumbnail_level}.jpg', format='JPEG')
                print('Saved mask thumbnail....')

            print('End croping patches...')

    def close(self):
        so.SqCloseSdpc(self.sdpc)
