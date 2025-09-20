# SplatTalk: 3D VQA with Gaussian Splatting

## Installation

To get started, create a virtual environment using the provided `environment.yml` file:

```bash
git clone https://github.com/ngailapdi/SplatTalk.git
cd SplatTalk
conda env create -f environment.yml
conda activate splattalk
```

This environment should work for systems with CUDA 12.X.

<details>
<summary>Troubleshooting</summary>
<br>

The Gaussian splatting CUDA code (`diff-gaussian-rasterization`) must be compiled using the same version of CUDA that PyTorch was compiled with. If your system does not use CUDA 12.X by default, you can try the following:

- Install a version of PyTorch that was built using your CUDA version. For example, to get PyTorch with CUDA 11.8, use the following command (more details [here](https://pytorch.org/get-started/locally/)):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- Install CUDA Toolkit 12.X on your system. One approach (*try this at your own risk!*) is to install a second CUDA Toolkit version using the `runfile (local)` option. For instance, to install CUDA Toolkit 12.1, download from [here](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local). When you run the installer, disable the options that install GPU drivers and update the default CUDA symlinks. If you do this, you can point your system to CUDA 12.1 during installation as follows:

```bash
LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64 pip install -r requirements.txt
# If everything else was installed but you're missing diff-gaussian-rasterization, do:
LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64 pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
```
</details>

## Acquiring Datasets

SplatTalk is trained using scenes from [ScanNet](http://www.scan-net.org).

The downloaded dataset under path ```datasets/``` should look like:
```
datasets
├─ scannet
│  ├─ train
│  ├  ├─sceneXXXX_XX
|  ├  ├  ├─ color (RGB images)
│  ├  ├  ├─ depth (depth images)
│  ├  ├  ├─ intrinsic (intrinsics)
│  ├  ├  └─ extrinsics.npy (camera extrinsics)
│  ├  ├─ sceneYYYY_YY
│  ├  ...
│  ├─ test
│  ├  ├─
│  ├  ...
│  ├─ train_idx.txt (training scenes list)
│  └─ test_idx.txt (testing scenes list)
└─
```
To obtain `extrinsics.npy` from the raw ScanNet data, run
```
python convert_poses.py
```

## Acquiring Pre-trained Checkpoints

TODO.

## Running the Code

### Training

The main entry point is `src/main.py`. To train on 100 views, run the following command:

```bash
python -m src.main +experiment=scannet/fvt +output_dir=train_fvt_full_100v
```
You can modify the number of training views with the following command (replace XX with your desired number of views):
```bash
python -m src.main +experiment=scannet/fvt +output_dir=train_fvt_full_100v dataset.view_sampler.num_context_views=XX
```
The output will be saved in path ```outputs/<output_dir>```.

We trained our model using one H100 GPU for 7 days.

### Evaluation

To evaluate pre-trained model on the ```[N]```-views setting on ```[DATASET]```, you can call:

```bash
python -m src.main +experiment=scannet/fvt +output_dir=[OUTPUT_PATH] mode=test dataset/view_sampler=evaluation checkpointing.load=[PATH_TO_CHECKPOINT] dataset.view_sampler.num_context_views=[N]
```


## BibTeX
If you find our work helpful, please consider citing our paper. Thank you!
```
@article{thai2025splattalk,
  title={Splattalk: 3d vqa with gaussian splatting},
  author={Thai, Anh and Peng, Songyou and Genova, Kyle and Guibas, Leonidas and Funkhouser, Thomas},
  journal={arXiv preprint arXiv:2503.06271},
  year={2025}
}
```

## Acknowledgements

Our code is largely based on [FreeSplat](https://github.com/wangys16/FreeSplat). Thanks for their great work!
