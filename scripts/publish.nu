use std assert

const to_reject = [
    archive.nu
    publish.nu
    .gitignore
    .gitmodules
    reference/LLSE_in.mat
    reference/LLSE_mat.mat
    reference/LLSE_out.mat
    reference/husky_cif_dct_gop01.mat
    reference/husky_cif_frame01.mat
    reference/husky_cif_g_gop01.mat
    reference/husky_cif_mean_gop01_frame01.mat
    reference/husky_cif_noise_gop01.mat
    reference/husky_cif_var_gop01_frame01.mat
    reference/kodim23_cif_dct_gop01.mat
    reference/kodim23_cif_frame01.mat
    reference/kodim23_cif_g_gop01.mat
    reference/kodim23_cif_mean_gop01_frame01.mat
    reference/kodim23_cif_noise_gop01.mat
    reference/kodim23_cif_var_gop01_frame01.mat
    reference/sigma_noise.mat
    reference/var.mat
    submodules/Depth-Anything
    submodules/README.md
    submodules/drn
]

export def main [--zip, --out-dir(-o): string, --dry] {
    assert ('.git' | path exists) "Run from CV-Cast repo root"
    assert ($zip or ($out_dir | is-not-empty)) "Choose either zip or output folder"

    let all = git ls-files | lines
    let files = $all | where $it not-in $to_reject
    print $files

    if $zip {
        let now = (date now | date to-record)
        let name = $'cvcast-($now.year)-($now.month)-($now.day).zip'
        print $name

        if not $dry {
            ^zip $name  ...$files
        }
    }

    if ($out_dir | is-not-empty) {
        if not $dry {
            print $out_dir
            mkdir $out_dir
            for file in $files {
                let dir = $file | path dirname
                if ($dir | is-empty) {
                    cp $file $out_dir
                } else {
                    let outdir = $out_dir | path join $dir
                    mkdir $outdir
                    cp -r $file $outdir
                }
            }
        }
    }
}
