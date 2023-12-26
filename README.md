# [Updating News (2023.12.26)]
1. **sdpc-linux** and **sdpc-win** are no longer be updated. The latest library **sdpc-for-python** is a new version for Sdpc Python library, which can be used in both Windows and Linux Systems.
2. The color correction has been updated. Now the color of the cropped patches are consistent with the color in the reading software.
3. Changed function name `level_downsample` to `level_downsamples`.
4. Chineses paths have been supported.
5. Added a function to view the magnification directly: `wsi.scan_magnification()`.
6. Added a function to view the sampling ratio directly：`wsi.sampling_rate()`.
7. Added a function of obtaining the thumbnail image: `wsi.get_thumbnail(thumbnail_level)`.
8. The `wsi.crop_patches()` function has been added. Now you can call the function directly in the code to separate the foreground tissue area and crop the patches (Using Pillow Library to save patches).
9. Added a option that can normalize images in `wsi.crop_patches()` function, it normalizes by H and E channels.
    
# About sdpc-for-python

Sdpc-for-python is a python library for processing whole slide images (WSIs) in **sdpc** format. To read WSIs in sdpc format in Windows platform, download the [TEKSQRAY reading software](https://www.sqray.com/yprj).

|  Download link | Extraction code | Instruction |
|  ----  | ----  | ----  |
| [Baidu Cloud](https://pan.baidu.com/s/1A4oOSlS2pCTsSRmQ_eCljQ)  | sq12 | Lite version |
| [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/4b533c59a2b74e099d08/?dl=1) | - | Full version |

# Installation

|  Platform   |  PyPI installer |
|  ----  | ----  |
| Windows/Linux  | `pip install sdpc-for-python` |


# How to use Sdpc library

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
*is_mark*： whether the marked thumbnail is needed, default=True
*is_mask*： whether the mask of the marked thumbnail is needed, default=True
*patch_norm*: whether the normalized color of patches is needed, default=False. (select this will reduce time greatly)

# Other method to cut patches with Sdpc Library

- Multi-thread processing

Build patches from WSIs in sdpc format in [Build-patch-for-sdpc repository](https://github.com/RenaoYan/Build-Patch-for-Sdpc).

Two approaches (build *w/wo* .sdpl) to build patches are given for two different platforms (Windows/Linux).


# Troubleshooting

1. `OSError: libDecodeHevc.so: cannot open shared object file: No such file or directory`

See the issue [#2](https://github.com/WonderLandxD/sdpc-for-python/issues/2).

*Jiawen Li, H&G Pathology AI Research Team*
