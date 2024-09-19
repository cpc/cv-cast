# Dependency installation for nn-spectral-sensitivity
#
# To run the script, you need to have Nushell. Alternatively, you can just
# look at the contents of `main` and follow the steps manually. Make sure to
# first update paths according to your system.
#
# Requirements:
# * conda
# * at the time of writing Python 3.11 due to OpenCV not supporting 3.12 yet
# * quirks related to LLVM and libstdc++ (see the end of the script)
#
# If installing torchjpeg fails with missing setuptools, remove the pyproject.toml.

use std assert

const CONDA_MODULE_PATH = ($nu.default-config-dir | path join scripts conda.nu)
use $CONDA_MODULE_PATH

def env-path []: [string -> nothing, string -> path] {
    let env_name = $in

    conda env list --json
    | from json
    | get envs
    | where ($it | path basename) == $env_name
    | get -i 0
}

# Create a conda environment and install all dependencies in it
export def main [
    env_name: string,    # name of the environment
    py_version: string,  # Python version to use (tested 3.11)
    --force(-f)          # remove the environment if it already exists
    --cpu(-c)            # install CPU versions of packages where possible
    --tf-cpu             # force TensorFlow CPU version
] {
    # required for sionna, probably needed only at runtime, just making sure
    $env.DRJIT_LIBLLVM_PATH = "/usr/lib/llvm-14/lib/libLLVM.so"

    if $force {
        conda remove -y -n $env_name --all
    }

    let env_path = $env_name | env-path

    if ($env_path | is-empty) {
        conda create -y -n $env_name $'python=($py_version)'
    }

    let env_path = $env_name | env-path
    assert ($env_path | is-not-empty)

    conda activate $env_name
    assert ($env_name in (which python).path.0)
    assert ($env_name in (which pip).path.0)

    conda install -y opencv

    if $cpu {
        conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
    } else {
        conda install -y python torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    }

    conda install -y plotly -c plotly
    conda install -y nbformat  # related to showing plotly plots
    conda install -y scipy numpy ipython matplotlib ipykernel black setuptools

    pip install pylance

    let submodules_path = $env.FILE_PWD | path join .. submodules | path expand

    do {
        cd ($submodules_path | path join torchjpeg)
        pip install -e '.'
    }

    do {
        cd ($submodules_path | path join ultralytics)
        pip install -e '.'
    }

    do {
        cd ($submodules_path | path join Depth-Anything)
        pip install -r requirements.txt
    }

    if $cpu or $tf_cpu {
        pip install tensorflow-cpu
    } else {
        pip install tensorflow==2.12  # exact version to required with CUDA 11.8
        let LD_LIBRARY_PATH = ($env.LD_LIBRARY_PATH
            | prepend ($env_path | path join lib)
            | str join (char esep))
        conda env config vars set $'LD_LIBRARY_PATH=($LD_LIBRARY_PATH)'
        # set path to CUDA
        conda env config vars set XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
    }

    pip install sionna

    pip install geffnet
    pip install git+https://github.com/lilohuang/PyTurboJPEG.git
    pip install pprofile pytorch-msssim
    pip install bjontegaard kaleido
    pip install compressai timm  # for ICM / LIC

    # (if you get GLIBCXX version errors:)
    # Fix conda installing weird libstdc++ with outdated glibc++
    ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ($env_path | path join lib libstdc++.so.6)
    ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ($env_path | path join lib libstdc++.so)

    # Set relevant environment variables

    # required for sionna, set according to your system, or try if it works without:
    conda env config vars set $'DRJIT_LIBLLVM_PATH=($env.DRJIT_LIBLLVM_PATH)'

    # to remove determinism errors
    conda env config vars set CUBLAS_WORKSPACE_CONFIG=:4096:8
}
