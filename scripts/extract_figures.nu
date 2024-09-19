# Update figures in a target directory with generated figures
export def main [tgt_dir: path --dry] {
    ls ($tgt_dir | path join '*.png' | into glob)
    | get name
    | path basename
    | each {|p|
        let new_path = '../nn-spectral-sensitivity/experiments/plots' | path join $p

        if ($new_path | path exists) {
            let old_path = $new_path | path parse | update parent $tgt_dir | path join
            let new_hash = open --raw $new_path | hash md5
            let old_hash = open --raw $old_path | hash md5

            if not $dry {
                cp $new_path $tgt_dir
            }

            {
                name: $p,
                src_exists: ($new_path | path exists)
                updated: ($new_hash != $old_hash)
            }
        } else {
            {
                name: $p,
                src_exists: ($new_path | path exists)
                updated: false
            }
        }
    }
    | sort-by src_exists updated
}
