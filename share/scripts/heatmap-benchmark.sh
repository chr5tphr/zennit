#!/usr/bin/env bash


COMPOSITES=(
    epsilon_gamma_box
    epsilon_plus
    epsilon_plus_flat
    epsilon_alpha2_beta1
    epsilon_alpha2_beta1_flat
)

ABSCOMPOSITES=(
    guided_backprop
    excitation_backprop
    deconvnet
)

ATTRIBUTORS=(
    smoothgrad
    integrads
    gradient
)

EXTRA=(
    occlusion
)

ALLNAMES=(
    input
    "${COMPOSITES[@]}"
    "${ABSCOMPOSITES[@]}"
    "${ATTRIBUTORS[@]}"
    "${EXTRA[@]}"
)


UNSIGNED_CMAPS=(
    hot
    cold
    wred
    wblue
    gray
)

SIGNED_CMAPS=(
    coldnhot
    bwr
)

SUFFIX=''

die() {
    echo >&2 -e "${1-}"
    exit "${2-1}"
}


usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [-h] [-v] [-o output]

Compute heatmaps for builtin attribution methods.

Available options:

-o, --output    Path to save heatmaps (default = 'results')
-p, --python    Path to python binay (default = '.venv/bin/python')
-s, --script    Path to feed_forward.py script (default = 'feed_forward.py')
-d, --dataset   Path to dataset (default = 'data/lighthouses')
-c, --s-cmap    Color-map for signed attributions (default = 'coldnhot')
-u, --u-cmap    Color-map for unsigned attributions (default = 'hot')
-h, --help      Print this help and exit
-v, --verbose   Print script debug info
--pdb           Run debugger
EOF
  exit
}


parse_params() {
    output=""
    python=".venv/bin/python"
    script="feed_forward.py"
    dataset="data/lighthouses"
    python_args=()
    unsigned_cmap='hot'
    signed_cmap='coldnhot'
    attributions=()

    MODEL='vgg16'
    PARAMS='params/vgg16-397923af.pth'

    args=()

    while (($#)); do
        case "${1-}" in
            -h | --help) usage ;;
            -v | --verbose) set -x ;;
            -o | --output)
                output="${2-}"
                shift
                ;;
            -p | --python)
                python="${2-}"
                shift
                ;;
            -s | --script)
                script="${2-}"
                shift
                ;;
            -d | --dataset)
                dataset="${2-}"
                shift
                ;;
            -m | --model)
                case "${2-}" in
                    vgg16 | vgg16_bn | resnet50 | resnet18)
                        MODEL="${2-}"
                        PARAMS="$(find params -name "${MODEL}-*.pth" | head -n 1)"
                        [[ -z "$PARAMS" ]] && die "No parameters found for model '${MODEL}'"
                        ;;
                    *) die "Unknown model: ${2-}" ;;
                esac
                shift
                ;;
            -a | --attribution)
                mapfile -td, -O "${#attributions[@]}" attributions < <(echo -n "${2}")
                shift
                ;;
            -c | --signed-cmap)
                signed_cmap="${2-}"
                shift
                ;;
            -u | --unsigned-cmap)
                unsigned_cmap="${2-}"
                shift
                ;;
            --pdb) python_args+=('-m' 'pdb') ;;
            --) args+=("${@:2}"); break ;;
            -?*) die "Unknown option: ${1}" ;;
            *) args+=("${1-}") ;;
            # *) break ;;
        esac
        shift
    done

    # args=("${@}")

    [[ -z "${output}" ]] && output="$(uv pip show zennit | awk 'NR==2{print $2}')"
    # (( ${#args[@]} )) && die "Too many positional arguments"
    (( ${#args[@]} )) || args=('attribution' 'montage')
    (( ${#attributions[@]} )) || attributions=("${ALLNAMES[@]}")

    return 0
}


attribution(){
    "$python" "${python_args[@]}" "$script" \
        "${dataset}" \
        "${output}/${MODEL}_${1}_{sample:02d}.png" \
        --model "$MODEL" \
        --parameters "$PARAMS" \
        "${@:2}"
}

print_filenames(){
    for method in "${@}"; do
        printf "${output}/${MODEL}_${method}_%02d.png\n" 0 1 2 3 4 5 6 7
    done
}

print_caption_filenames(){
    for method in "${@}"; do
        echo -ne "caption:${method//_/\ }\n"
        printf "${output}/${MODEL}_${method}_%02d.png\n" 0 1 2 3 4 5 6 7
    done
}

all_attributions(){
    for composite in "${COMPOSITES[@]}"; do
        attribution \
            "${composite}" \
            --composite "${composite}" \
            --relevance-norm symmetric \
            --cmap "${signed_cmap}"
    done

    for composite in "${ABSCOMPOSITES[@]}"; do
        attribution \
            "${composite}" \
            --composite "${composite}" \
            --relevance-norm absolute \
            --cmap "${unsigned_cmap}"
    done

    for attributor in "${ATTRIBUTORS[@]}"; do
        attribution \
            "${attributor}" \
            --attributor "${attributor}" \
            --relevance-norm absolute \
            --cmap "${unsigned_cmap}"
    done

    attribution \
        "occlusion" \
        --inputs "${output}/${MODEL}_input_{sample:02d}.png" \
        --attributor "occlusion" \
        --relevance-norm unaligned \
        --cmap "${unsigned_cmap}"
}

full_montage(){
    mapfile -t filenames < <(print_caption_filenames "${@}")
    montage \
        -size 96x96 \
        -gravity center \
        -pointsize 16 \
        -background '#77f' \
        "${filenames[@]}" \
        -geometry 96x96+1+1 \
        -tile "9x${#}" \
        -define webp:lossless=true \
        "${output}/full_montage_${MODEL}${SUFFIX}.webp"
}

color_montage(){
    s_names=(
        "${ABSCOMPOSITES[@]}"
        "${ATTRIBUTORS[@]}"
        "${EXTRA[@]}"
    )
    mapfile -t filenames_ < <(print_filenames "${s_names[@]}")
    for cmap in "${SIGNED_CMAPS[@]}"; do
        "$python" palette_swap.py --cmap "${cmap}" "${filenames_[@]}"
        SUFFIX="_${cmap}"
        full_montage input "${s_names[@]}"
    done
    "$python" palette_swap.py --cmap "${signed_cmap}" "${filenames_[@]}"

    u_names=(
        "${COMPOSITES[@]}"
    )
    mapfile -t filenames_ < <(print_filenames "${u_names[@]}")
    for cmap in "${UNSIGNED_CMAPS[@]}"; do
        "$python" palette_swap.py --cmap "${cmap}" "${filenames_[@]}"
        SUFFIX="_${cmap}"
        full_montage input "${u_names[@]}"
    done
    "$python" palette_swap.py --cmap "${unsigned_cmap}" "${filenames_[@]}"

    SUFFIX=''
}

parse_params "$@"

mkdir -p "${output}"

for action in "${args[@]}"; do
    case "${action}" in
        attribution)
            all_attributions
            ;;
        montage)
            full_montage "${attributions[@]}"
            ;;
        color-montage)
            color_montage
            ;;
        *)
            die "No such action: '${action}'"
            ;;
    esac
done
