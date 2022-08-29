import ctypes
import gc
import numpy as np
import numpy.ctypeslib as npCtypes
from ctypes import *
import copy
import json
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import sys
import glob
import argparse
from tqdm import tqdm
import time
from Sdpc_struct import SqSdpcInfo

####################################################################################################
parser = argparse.ArgumentParser(description='Code to patch WSI using SDPC files')
parser.add_argument('--data_path', type=str, default='/data_sda/yra/sss',
                    help='path of SDPC files')
# 在winsdow/linux上切patch
parser.add_argument('--wins_linux', type=int, default=1, help='0:wins; 1:linux')
parser.add_argument('--dll_location', type=str, default='E:/DLL/dll',
                    help='where DecodeSdpcDll.dll located')
parser.add_argument('--so_location', type=str, default='/data_sda/yra/DecodeSdpcDemo/DecodeSdpcDemo/lib/LINUX',
                    help='where libDecodeSdpc.so located')
parser.add_argument('--patch_w', type=int, default=224, help='the width of patch')
parser.add_argument('--patch_h', type=int, default=224, help='the height of patch')
parser.add_argument('--magnification', type=int, default=10, help='magnification: 0-100x')

# 模式1：用sdpl切patch；模式2：自动切patch
parser.add_argument('--build_mode', type=int, default=1, help='1:use sdpl; 2:no sdpl')
# mode 1 parameters：magnification是想要切的patch的倍率，rect_mag是参考矩形所在的倍率
parser.add_argument('--rect_mag', type=int, default=40, help='the magnification of standard rectangular: 0-100x')
# mode 2 parameters
parser.add_argument('--RGB_TH', type=int, default=240, help='the threshold of rgb to screen out valid patches')
parser.add_argument('--RGB_var_TH', type=int, default=500, help='the threshold of rgb to screen out valid patches')


####################################################################################################

# 创建Sdpc类以读取Sdpc文件
class Sdpc:

    def __init__(self, sdpcPath):
        print(sdpcPath)
        self.sdpc = self.readSdpc(sdpcPath)
        self.level_count = self.getLevelCount()
        self.level_downsample = self.getLevelDownsamples()
        self.level_dimensions = self.getLevelDimensions()

    def getRgb(self, rgbPos, width, height):

        intValue = npCtypes.as_array(rgbPos, (height, width, 3))
        return intValue

    def readSdpc(self, fileName):

        sdpc = so.SqOpenSdpc(c_char_p(bytes(fileName, 'gbk')))
        sdpc.contents.fileName = bytes(fileName, 'gbk')

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

    def read_region(self, location, level, size):

        startX, startY = location
        scale = self.level_downsample[level]
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

    def close(self):

        so.SqCloseSdpc(self.sdpc)


def polygon_preprocessor(points):
    n = len(points[0])

    # 对平行的线段施加扰动
    for i in range(n - 2):
        if points[1][i] == points[1][i + 1]:
            points[1][i + 1] += 1
    if points[1][n - 2] == points[1][n - 1]:
        if points[1][n - 2] == points[1][n - 3] - 1:
            points[1][n - 2] -= 1
        else:
            points[1][n - 2] += 1

    # 将第二个点添加到末行以备用
    points[0].append(points[0][1])
    points[1].append(points[1][1])

    return points


def inorout(points, x, y):
    n = len(points[0])
    count = 0

    for i in range(n - 2):
        # 排除掉出界的情况
        if points[1][i] > points[1][i + 1]:
            if y >= points[1][i] or y < points[1][i + 1]:
                continue
            if y == points[1][i + 1] and y < points[1][i + 2]:
                continue  # 排除掉可能相切的情况
        else:
            if y <= points[1][i] or y > points[1][i + 1]:
                continue
            if y == points[1][i + 1] and y > points[1][i + 2]:
                continue  # 排除掉可能相切的情况

        # 讨论线段垂直的情况
        if points[0][i] == points[0][i + 1]:
            if x < points[0][i]:
                count += 1

        # 讨论一般情况
        else:
            slope = (points[1][i + 1] - points[1][i]) / (points[0][i + 1] - points[0][i])  # 计算线段的斜率
            y_hat = slope * (x - points[0][i]) + points[1][i]
            if slope * (y - y_hat) > 0:
                count += 1

    return count


def getcoords(sdpc_path, label_dic, save_dir, patch_w, patch_h, patch_sizeup=1, wins_linux=0, patch_level=0):
    template = None
    std_scale = None
    colors = []
    std_color = None
    # 寻找标注的矩形模板，并以该倍率作为参考坐标系
    for counter in tqdm(label_dic['LabelRoot']['LabelInfoList'][0:]):
        # 设置矩形模板的公用参数
        if counter['LabelInfo']['ToolInfor'] == "btn_rect":
            std_color = counter['LabelInfo']['PenColor']
            template = copy.deepcopy(counter)
            rec_index = len(label_dic['LabelRoot']['LabelInfoList']) + 1
            template['LabelInfo']['Id'] = rec_index
            template['LabelInfo']['PenColor'] = "Red"
            template['LabelInfo']['Dimensioning'] = 1
            # 提取该模板在绝对坐标系下的倍率
            std_scale = template['LabelInfo']['ZoomScale']
        elif counter['LabelInfo']['PenColor'] not in colors:
            colors.append(counter['LabelInfo']['PenColor'])
    if template == None:
        print('Please add a standard rectangle')
        sys.exit()
    colors.sort()
    # 读取sdpc文件
    sdpc = Sdpc(sdpc_path)
    # 对所有标注遍历
    for counter in tqdm(label_dic['LabelRoot']['LabelInfoList'][0:]):
        # 对笔刷标注切片
        if counter['LabelInfo']['ToolInfor'] == "btn_brush" or counter['LabelInfo']['ToolInfor'] == "btn_pen":
            color_index = colors.index(counter['LabelInfo']['PenColor'])
            Points = list(zip(*[list(map(int, point.split(', '))) for point in counter['PointsInfo']['ps']]))
            # 获取笔刷所在的屏幕位置
            Ref_x, Ref_y, _, _ = counter['LabelInfo']['CurPicRect'].split(', ')
            Ref_x, Ref_y = int(Ref_x), int(Ref_y)
            # 计算参考坐标系与绝对坐标系的比率
            ratio = counter['LabelInfo']['ZoomScale'] / std_scale
            # 将标注的点放缩到参考坐标系下
            Pointsx = []
            Pointsy = []
            for i in range(len(Points[0])):
                Pointsx.append(int((Points[0][i] + Ref_x) / ratio) - Ref_x)
                Pointsy.append(int((Points[1][i] + Ref_y) / ratio) - Ref_y)
            # 获取闭合曲线在参考坐标系下的左上角起点spa，右下角终点spb
            SPA_x, SPA_y = (min(Pointsx), min(Pointsy))
            SPB_x, SPB_y = (max(Pointsx), max(Pointsy))
            Pointslist = polygon_preprocessor([Pointsx, Pointsy])
            # *************************************************
            # 以标注的中心点作为原点，计算patch下标
            x0 = np.mean(np.array(Pointsx[:-1]))
            y0 = np.mean(np.array(Pointsy[:-1]))
            start_kx = -np.ceil((x0 - SPA_x - patch_w / 2) / patch_w)
            end_kx = np.ceil((SPB_x - x0 - patch_w / 2) / patch_w)
            start_ky = -np.ceil((y0 - SPA_y - patch_h / 2) / patch_h)
            end_ky = np.ceil((SPB_y - y0 - patch_h / 2) / patch_h)

            # 循环操作，生成标注
            for x in range(int(start_kx), int(end_kx) + 1):
                for y in range(int(start_ky), int(end_ky) + 1):
                    # 每个patch的中心点
                    testx = (x0 + x * patch_w)
                    testy = (y0 + y * patch_h)
                    # 判断中心点是否在框中
                    count = inorout(Pointslist, testx, testy)
                    if (count % 2) == 0:
                        continue
                    # 如果在框中，更新标注信息
                    rect_x = testx - patch_w / 2
                    rect_y = testy - patch_h / 2
                    # 设置要添加的标注信息
                    template1 = copy.deepcopy(template)
                    template1['LabelInfo']['StartPoint'] = '%d, %d' % (50, 50)
                    template1['LabelInfo']['EndPoint'] = '%d, %d' % (50 + patch_w, 50 + patch_h)
                    template1['LabelInfo']['Rect'] = '%d, %d, %d, %d' % (50, 50, patch_w, patch_h)
                    template1['LabelInfo']['CurPicRect'] = '%d, %d, %d, %d' % (
                        Ref_x + rect_x - 50, Ref_y + rect_y - 50, 0, 0)
                    try:
                        img = sdpc.read_region(((Ref_x + rect_x) / std_scale, (Ref_y + rect_y) / std_scale),
                                               patch_level, (int(patch_w / std_scale), int(patch_h / std_scale)))
                    except:
                        print('(%d, %d) is out of the WSI, continue...' % (
                            (Ref_x + rect_x) * 2, (Ref_y + rect_y) * 2))
                        continue
                    save_dir1 = copy.deepcopy(save_dir) + "/" + str(color_index) + "_" + colors[color_index]
                    if not os.path.exists(save_dir1):
                        os.mkdir(save_dir1)
                    if wins_linux == 1:
                        pre_file = save_dir.split('/')[-1]
                    else:
                        pre_file = save_dir.replace('/', '\\').split('\\')[-1]
                    image_file = pre_file + '_%d.png' % template1['LabelInfo']['Dimensioning']
                    save_path = os.path.join(save_dir1, image_file)
                    im = Image.fromarray(img)
                    im.thumbnail((patch_w / patch_sizeup, patch_h / patch_sizeup))
                    if wins_linux == 1:
                        im.save(save_path)
                    else:
                        im.save(save_path.replace('/', '\\'))

                    # 将其添加到LabelList中
                    label_dic['LabelRoot']['LabelInfoList'].append(template1)
                    # 模版迭代
                    rec_index += 1
                    template['LabelInfo']['Id'] = rec_index
                    template['LabelInfo']['Dimensioning'] = template['LabelInfo']['Dimensioning'] + 1


if __name__ == '__main__':
    args = parser.parse_args()
    if args.wins_linux == 1:  # linux下环境配置，调用.so文件
        sys.path.append(args.so_location)
        soPath = args.so_location + '/libDecodeSdpc.so'
    else:  # windows下环境配置，调用.dll文件
        os.chdir(args.dll_location)
        sys.path.append(args.dll_location)
        soPath = args.dll_location + '/DecodeSdpcDll.dll'
    # 接口指针设置
    so = ctypes.CDLL(soPath)
    so.GetLayerInfo.restype = POINTER(c_char)
    so.SqGetRoiRgbOfSpecifyLayer.argtypes = [POINTER(SqSdpcInfo), POINTER(POINTER(c_uint8)),
                                             c_int, c_int, c_uint, c_uint, c_int]
    so.SqGetRoiRgbOfSpecifyLayer.restype = c_int
    so.SqOpenSdpc.restype = POINTER(SqSdpcInfo)

    data_path = args.data_path
    patch_w, patch_h = args.patch_w, args.patch_h

    files = os.listdir(data_path)
    for i, file in enumerate(files):
        if file.endswith('.sdpc'):
            print('file = ', file)
            fi_path_, _ = file.split('.')
            fi_path = os.path.join(data_path + '/' + fi_path_)
            folder = os.path.exists(fi_path)
            # 新建保存patch的文件夹
            if not folder:
                os.makedirs(fi_path)
            else:
                print('...There is this folder')
                continue

            print('---------------* Processing: %s *---------------' % file)
            time_start = time.time()
            if args.build_mode == 1:
                if not os.path.exists(os.path.join(data_path + '/' + file.replace('sdpc', 'sdpl'))):
                    continue
                # 已经得到了每张wsi的各个patch的坐标位置
                sdpc_path = os.path.join(data_path + '/' + file)
                sdpl_path = fi_path + '.sdpl'
                # 保存原有的sdpl至*_old.sdpl
                if args.wins_linux == 1:
                    os.system('cp %s %s' % (sdpl_path, sdpl_path.replace('.sdpl', '_old.sdpl')))  # Linux下拷文件
                else:
                    os.system('copy %s %s' % (sdpl_path.replace('/', '\\'),
                                              sdpl_path.replace('/', '\\').replace('.sdpl', '_old.sdpl')))  # window下拷文件
                with open(sdpl_path, 'r', encoding='UTF-8') as f:
                    label_dic = json.load(f)
                    # 计算在第0层需要切patch的大小
                    sizeup = int(args.rect_mag / args.magnification)
                    cutpatch_w = patch_w * sizeup
                    cutpatch_h = patch_h * sizeup
                    getcoords(sdpc_path=sdpc_path, label_dic=label_dic, save_dir=fi_path, patch_w=cutpatch_w,
                              patch_h=cutpatch_h, patch_sizeup=sizeup, wins_linux=args.wins_linux)
                    # 保存新的sdpl覆盖*.sdpl
                    with open(sdpl_path, 'w') as f:
                        json.dump(label_dic, f)


            elif args.build_mode == 2:
                sdpc_path = os.path.join(data_path + '/' + file)
                slide = Sdpc(sdpc_path)

                sizeup = 100 / args.magnification  # 默认第0层是100倍
                x_size = int(args.patch_w * sizeup)
                y_size = int(args.patch_h * sizeup)
                x_final_size = args.patch_w
                y_final_size = args.patch_h

                img_x, img_y = slide.level_dimensions[0]

                for i in range(int(np.floor(img_x / x_size))):
                    for j in range(int(np.floor(img_y / y_size))):
                        img = slide.read_region((i * x_size, j * y_size), 0, (x_size, y_size))
                        img_array = np.array(img)
                        img_RGB_mean = np.mean(img_array[:, :, :])
                        img_RGB_var = np.var(img_array[:, :, :])
                        print('Location: (%d, %d), Mean: %s, Var: %s'
                              % (i * x_size * sizeup, j * y_size * sizeup, str(img_RGB_mean), str(img_RGB_var)))
                        if img_RGB_mean < args.RGB_TH and img_RGB_var > args.RGB_var_TH:
                            save_path = fi_path + '/' + str(i) + '_' + str(j) + '.png'
                            im = Image.fromarray(img_array)
                            im.thumbnail((x_final_size, x_final_size))
                            if args.wins_linux == 1:
                                im.save(save_path)
                            else:
                                im.save(save_path.replace('/', '\\'))
            time_end = time.time()
            print('total time:', time_end - time_start)
    print('-----------------* Patching Finished *---------------------')
