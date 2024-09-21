# CV-Cast

Working repository for our paper **"CV-Cast: Computer Vision–Oriented Linear Coding and Transmission"** by Jakub Žádník, Michel Kieffer, Anthony Trioux, Markku Mäkitalo, and Pekka Jääskeläinen (under review for Transactions on Mobile Computing).

The paper proposes new Linear Coding and Transmission [1] schemes that are optimized for minimal neural network loss instead of per-pixel distortion. 
Inspired by [2], we use gradient with respect to NN loss to derive these new schemes.
On the tested computer vision (CV) tasks, CV-Cast achieves higher accuracy than the baseline LCT, and similar to the baseline LCT, its performance degrades smoothly with increased noise levels, unlike classical entropy-coded compression transmitted via a digital channel.

For evaluation of baseline LCT schemes in terms of CV task accuracy under channel impairments, see [3].

[1] Szymon Jakubczak, Dina Katabi. ["SoftCast: One-Size-Fits-All Wireless Video."](https://dl.acm.org/doi/pdf/10.1145/3300061.3345448) _SIGCOMMM_, 2010. [(longer version)](https://core.ac.uk/download/pdf/83176145.pdf)

[2] Xiufeng Xie and Kyu-Han Kim. ["Source Compression with Bounded DNN Perception Loss for IoT Edge Computer Vision."](https://dl.acm.org/doi/pdf/10.1145/3300061.3345448) _MobiCom_, 2019.

[3] Jakub Žádník, Anthony Trioux, Michel Kieffer, Markku Mäkitalo, François-Xavier Coudoux, Patrick Corlay, and Pekka Jääskeläinen. ["Performance of Linear Coding and Transmission in Low-Latency Computer Vision Offloading."](https://dl.acm.org/doi/pdf/10.1145/3300061.3345448) _WCNC_, 2024.


## Structure

The project is fairly large and consists of multiple modules.

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

Other:
* `reference`: Used for evaluating the LVCT code against a MATLAB reference
* `combine_imgs.py`: Used to generate split cropped plots from multiple images
* `*.ipynb`: Different notebooks that were created to test different things during development
* `scripts`: Misc utilities
* `submodules`: 3rd-party repositories or custom forks of the following tools and DL models (some are unused):
    * https://github.com/ultralytics/ultralytics
    * https://gitlab.com/torchjpeg/torchjpeg
    * https://github.com/LiheYoung/Depth-Anything
    * https://github.com/fyu/drn
    * https://github.com/final-0/ICM-v1
    * https://github.com/jmliu206/LIC_TCM

## Prerequisites

Conda is used to manage the Python environments.
The easiest way to install dependencies is using the `scripts/install-conda-env.nu` script (you'll need Nushell to run it).
You can also inspect the script and install dependencies manually.
There is also `environment.yml` containing the latest snapshot of used dependencies (not tested).
Note that some dependencies, like `ultralytics` or `torchjpeg`, need to be installed via `pip install -e .` from their respective directories under `submodules/` (see the install script for details).

### environment.yml

(Assuming using Nushell used as the shell.)

To generate `environment.yml`:
```
conda env export | save --raw environment.yml
```

To create environment from `environment.yml`:
```
conda env create -f environment.yml
```

## Quick Start

To run experiments, use the `run_all()` function inside `experiments/__init__.py`.
You'll need pre-trained FastSeg and YOLOv8 models and update the paths in that file to point at them.
Most relevant experiments:
* `probe_only()`: Runs only the probing to get gradient of the NN wrt. DCT coefficients.
* `predict()`: Run inference and save the resulting images (requires probing to be run beforehand).
* `get_accuracies_single()`: Probe, run iference and store the results.
* `q_search()`: Search for JPEG Q parameter given a CR. Required if you want to run the above with JPEG or GRACE codecs.
In general, the coding and transmissin parameters are set within `experiments/__init__.py` and a single run can iterate through a multiple of them.
The experiments store results into a directory `<outdir>/runXY` as one or many `.pt` files.
These files can be re-used later in other experiments, or read manually with `torch.load(xyz.pt)`.

For plotting / interpretting the results, see `generate_results.ipynb`.
It looks for `runXY` folders generated during experiments specified by `OUTDIRS` inside `results.py`.
You'll need to update the paths there to point at your own generated results.
For every key in the `OUTDIRS` dictionary, the results are concatenated together.

## Profilers used

* scalene
    * `scalene --html --outfile profile.html bench.py --reduced-profile`
* py-spy
* cProfile
* pprofile
    * callgrind output, use kcachegrind to view
