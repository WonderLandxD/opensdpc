### [Updating News (2024.09.22)]
1. Some users' machines are not compatible with the latest compiled packages. For this reason, you can still use old versions. See: [old version for sdpc library](https://github.com/WonderLandxD/sdpc-for-python/tree/4c03a32473eb88f24283446c0967e5053f083896).
2. Due to copyright restrictions, I cannot directly provide the software for converting sdpc to svs. I am actively communicating with the company and providing an open source interface. Please contact the slide-scanner after-sales staff to request it. See [sqray.com](https://www.sqray.com/service/scanFilm) for more details.
3. Provide an example for generating patches with multiple thread in order to provide pre-processing for huge datasets. See [this subsection](https://github.com/WonderLandxD/sdpc-for-python/tree/main?tab=readme-ov-file#demo-code-of-using-sdpc-and-openslide-library-to-crop-the-patches).

[Note]
 please use **version==1.5** if you want to use sdpc-linux. I may be slow to reply, thank you for the patience 😊.

If you don't know how to deal with the old version, here is a simple plug-and-play process:
- Step1: Click the [Tsinghua cloud link](https://cloud.tsinghua.edu.cn/f/d1da5598c9c849d98e3f/?dl=1) to download LINUX.zip directly;
- Step2: Replace the unzipped LINUX folder with the LINUX folder of the sdpc package in your own environment;
- Step3: Write the absolute path of the LINUX folder and the ffmpeg folder inside it into the environment variable, see the issue [#2](https://github.com/WonderLandxD/sdpc-for-python/issues/2) for more details
- Step4: Just enter `import sdpc` in python to use it. 

### [Updating News (2024.02.03)]
1. Updated Part "Troubleshooting"

### [Updating News (2023.12.26)]
1. **sdpc-linux** and **sdpc-win** are no longer be updated. The latest library **sdpc-for-python** is a new version for Sdpc Python library, which can be used in both Windows and Linux Systems.
2. The color correction has been updated. Now the color of the cropped patches are consistent with the color in the reading software.
3. Changed function name `level_downsample` to `level_downsamples`.
4. Chineses paths have been supported.
5. Added a function to view the magnification directly: `wsi.scan_magnification()`.
6. Added a function to view the sampling ratio directly：`wsi.sampling_rate()`.
7. Added a function of obtaining the thumbnail image: `wsi.get_thumbnail(thumbnail_level)`.
8. The `wsi.crop_patches()` function has been added. Now you can call the function directly in the code to separate the foreground tissue area and crop the patches (Using Pillow Library to save patches).
9. Added a option that can normalize images in `wsi.crop_patches()` function, it normalizes by H and E channels.
    
## About sdpc-for-python

Sdpc-for-python is a python library for processing whole slide images (WSIs) in **sdpc** format. To read WSIs in sdpc format in Windows platform, download the [TEKSQRAY reading software](https://www.sqray.com/Download).

|  Download link | Extraction code | Instruction |
|  ----  | ----  | ----  |
| [Baidu Cloud](https://pan.baidu.com/s/1A4oOSlS2pCTsSRmQ_eCljQ)  | sq12 | Lite version |
| Please see the [sqray.com](https://www.sqray.com/Download) | - | Full version |

## Installation

|  Platform   |  PyPI installer |
|  ----  | ----  |
| Windows/Linux  | `pip install sdpc-for-python` |


## How to use Sdpc library

- **Import Sdpc Library:**

```
import sdpc
```

- **Read the WSI**

```
wsi = sdpc.Sdpc(wsi_path)
```

- **Get the number of levels of the WSI**

```
level_count = wsi.level_count
```

- **By how much are levels downsampled from the original image?**

```
level_downsamples = wsi.level_downsamples
```

- **Get the dimensions of data in each level**

```
level_dimensions = wsi.level_dimensions
```

- **Get the patch under a certain position at a certain level in the WSI**

```
img = wsi.read_region((Position_X, Position_Y), patch_level, (size_H, size_W))   # patch_size = (size_H, size_W)
```

- **Obtain the thumbnail (numpy format)**

```
thumbnail_np = wsi.get_thumbnail(thumbnail_level)
```
*thumbnail_level*: the level of the wanted thumbnail

- **Obtain the scaned magnification**

```
magnification = wsi.scan_magnification()
```

- **Obtain the downsampling ratio**

```
ration = wsi.sampling_rate()
```

- **Crop the foreground tissue to obtain the patches automatically**

```
wsi.crop_patches(tile_size, overlap_size, patch_level, save_dir, blank_TH=0.7, is_mark=True, is_mask=True, patch_norm=False)
```
*tile_size*: patch size

*overlap_size*: overlapped size: recommend: 0

*save_dir*: path to save

*blank_TH*: crop patches with blank rate lower than blank_TH, recommend: 0.7, default=0.7

*is_mark*: whether the marked thumbnail is needed, default=True

*is_mask*: whether the mask of the marked thumbnail is needed, default=True

*patch_norm*: whether the normalized color of patches is needed, default=False. (select this will reduce time greatly)

## Demo code of using Sdpc and OpenSlide library to crop the patches

- Using **single thread** Sdpc library to crop patches.
```
python sdpc_crop_patches.py
```

- Using **single thread** OpenSlide library to crop patches.
```
python openslide_crop_patches.py
```

- Using **multiple thread** to crop patches (***Recommend*** if the WSI number is large).
```
cd multiprocess
python generate_patches.py --format sdpc  # please see the line 74-83 to custom your own setting
```
Here, I give an example (see [create_wsi_list.py](https://github.com/WonderLandxD/sdpc-for-python/blob/main/multiprocess/create_wsi_list.py)) for create csv file and load to start cropping patches.

## Other method to cut patches with Sdpc Library

- Multi-thread processing

Build patches from WSIs in sdpc format in [Build-patch-for-sdpc repository](https://github.com/RenaoYan/Build-Patch-for-Sdpc).

Two approaches (build *w/wo* .sdpl) to build patches are given for two different platforms (Windows/Linux).


## Troubleshooting

1. `OSError: libDecodeHevc.so: cannot open shared object file: No such file or directory`

See the issue [#2](https://github.com/WonderLandxD/sdpc-for-python/issues/2).

2. `OSError: version 'GLIBC_2.33' not found`

See the issue [#12](https://github.com/WonderLandxD/sdpc-for-python/issues/12).

*Jiawen Li, H&G Pathology AI Research Team*
