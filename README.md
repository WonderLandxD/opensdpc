# Sdpc library on windows system

Authors: Yiqing Liu, Qiming He, Renao Yan, Jiawen Li

*H&G Pathology AI Research Team*

# Installation (based on PyPI installer)

`pip install sdpc-win`

# How to use Sdpc library

- Import Sdpc Library:

```
from sdpc.Sdpc import Sdpc
```

- Read WSI 

```
wsi = Sdpc(wsi_path)
```

- Get the number of levels of WSI

```
level_count = wsi.level_count
```

- By how much are levels downsampled from the original image?

```
level_downsamples = wsi.level_downsample
```

- Get the dimensions of data in each level

```
level_dimensions = wsi.level_dimensions
```

- Get the patch under a certain position at a certain level in WSI

```
img = wsi.read_region((Position_X, Position_Y), patch_level, (size_H, size_W))   # patch_size = (size_H, size_W)
```
