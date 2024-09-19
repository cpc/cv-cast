# CV-Cast

Linear Video Coding and Transmission (LVCT) schemes optimized by spectral sensitivity of neural networks using a method from
[*Xiufeng Xie and Kyu-Han Kim. 2019. **Source Compression with Bounded DNN Perception Loss for IoT Edge Computer Vision.** In The 25th Annual International Conference on Mobile Computing and Networking (MobiCom '19). Association for Computing Machinery, New York, NY, USA, Article 47, 1–16.*](http://xiufengxie.com/papers/XXie_MobiCom19_GRACE.pdf).

## Structure

The project is fairly large and consists of multiple submodules.

The most important modules are:
* `experiments` is the top level module that runs various experiments and saves results to files. `generate_results.ipynb` then processes the results and prints/plots them
* `model` module contains tested models and an API to probe/evaluate them. The most important functions are `run_models()` and `probe_models()`
* `lvc` module contains the implementation of LVCT. The most important function here is `create_lvc()`

Other important modules:
* `datasets` contains code related to setting up datasets
* `digcom` contains Sionna implementation of a 5G channel (as well as other misc. experiments)
* `transforms` contains various transforms and image error metrics
* `utils` contains misc. utilities

Not important modules:
* `gradients` contains old code for model probing, was later integrated to `model`

Submodules (install with `pip install -e .` after `cd`-ing to them):
* `ultralytics` is a custom fork of https://github.com/ultralytics/ultralytics implementing the YOLOv8 model

Other:
* `reference`: Used for evaluating the LVCT code against a MATLAB reference
* `combine_imgs.py`: Used to generate split cropped plots from multiple images
* `*.ipynb`: Different notebooks that were created to test different things during development

## Prerequisites

1. Linux, miniconda, Python (3.10)
1. It assumes a NVIDIA GPU
1. [Install pytorch with conda](https://pytorch.org), e.g.:
```shell
conda create -n <env-name>
conda activate <env-name>
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
# or: conda install -n <env-name> pytorch torchvision torchaudio cudatoolkit -c pytorch-nightly
```
1. Verify paths. If they don't point to `miniconda3/envs/<env-name>/bin`, move that folder to the top of your PATH
```shell
which python
which pip
```
1. Install pip packages (make sure the pip is conda's not the system one), e.g.:
```shell
pip install torchsummary
```
1. Install other packages (ipython is optional but I like it for quick testing), e.g.:
```shell
conda install matplotlib scipy ipython
```
1. On Linux, `conda install -c conda-forge libstdcxx-ng` might be necessary to fix wrong glibc++ version.

Over the time multiple packages were added, see the environment.yml. Some random points:
* Tensorflow might require env activations setup https://www.tensorflow.org/install/pip
* The local ultralytics submodule needs to be installed with `pip install -e .`
* Other libraries can be installed via `conda` or `pip`
* All dependencies can be seen in the `environment.yml` file

### environment.yml

(Assuming using Nushell as the shell.)

To generate `environment.yml`:
```
conda env export | save --raw environment.yml
```

To create environment from `environment.yml`:
```
conda env create -f environment.yml
```

## Profilers used

* scalene
    * `scalene --html --outfile profile.html bench.py --reduced-profile`
* py-spy
* cProfile
* pprofile
    * callgrind output, use kcachegrind to view
