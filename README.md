# NeuralVolumeCompression
## Requirements
- Python 3.9
- Libraries: torch, tqdm, numpy, torchmetrics
  
## General
Implementation of a modified U-Net model, used for compression of volumetric data.
Works with BVP files, but can easily be modified to use raw files. 

## Usage
### Compress
```
python compress.py <source_file> <destination_dir>
```
### Extract
```
python extract.py <source_file> <destination_dir>
```