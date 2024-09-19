import itertools
from pathlib import Path
from typing import List, Tuple, Literal

import bjontegaard as bd
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import block_process

from .results import (
    collect_df,
    get_baseline,
    MODEL_ORDER,
    DEFAULT_ID_COLS,
    PROBE_DIR_OLD,
)
from gradients.grace import get_quant_table, get_quant_table_approx

DEFAULT_SCALE = 4.0

MARKER_SIZE = 5
MARKER_WIDTH = 0
DASH = "4px"
CSNR_MIN = -5
CSNR_MAX = 25


def _filter_df(df, estimator, mode, block_dct, nchunks=None):
    if nchunks is None:
        return df[
            (df["estimator"] == estimator)
            & (df["mode"] == mode)
            & (df["block_dct"] == block_dct)
            & df["grad_w"]
            & df["grad_sel"]
            & df["grad_alloc"]
        ]
    else:
        return df[
            (df["estimator"] == estimator)
            & (df["mode"] == mode)
            & (df["block_dct"] == block_dct)
            & (df["nchunks"] == nchunks)
            & df["grad_w"]
            & df["grad_sel"]
            & df["grad_alloc"]
        ]


def _add_baseline(
    fig,
    baseline,
    model_ids=["fastseg_small", "fastseg_large", "yolov8_n", "yolov8_s", "yolov8_l"],
    col_major=False,
    pos="top right",
):
    for i, model_id in enumerate(model_ids, start=1):
        row = i if col_major else 0
        col = 0 if col_major else i
        try:
            fig.add_hline(
                y=baseline[model_id],
                line_dash="dot",
                annotation_text=f"{baseline[model_id] * 100:.2f}%",
                annotation_position=pos,
                annotation_xshift=-15,
                row=row,
                col=col,
            )
            # fig.add_hline(
            #     y=baseline[model_id],
            #     line_dash="dot",
            #     annotation_text=f"{baseline[model_id]:.4f}",
            #     annotation_position=pos,
            #     row=1,
            #     col=i,
            # )
            # fig.add_hline(
            #     y=baseline[model_id],
            #     line_dash="dot",
            #     annotation_text=f"{baseline[model_id]:.4f}",
            #     annotation_position=pos,
            #     row=2,
            #     col=i,
            # )
        except KeyError:
            continue


def _remove_duplicates(
    df: pd.DataFrame, subset: List[str] = DEFAULT_ID_COLS
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    duplicates = df[df.duplicated(subset=subset, keep=False)]
    df = df.drop_duplicates(subset=subset)
    return df, duplicates


def _get_bd(df) -> pd.DataFrame:
    """Calculate BD metrics from a given data frame"""

    estimators = pd.unique(df["estimator"])
    modes = pd.unique(df["mode"])
    nchunks = pd.unique(df["nchunks"])
    csnr_dbs = [val for val in pd.unique(df["csnr_db"]) if val != "inf"]
    block_dcts = pd.unique(df["block_dct"])
    interp_method = "pchip"  # akima, pchip

    rows = []
    for model_id, estimator, mode, nc, csnr_db, block_dct in itertools.product(
        MODEL_ORDER.keys(), estimators, modes, nchunks, csnr_dbs, block_dcts
    ):
        df_test = df[
            (df["csnr_db"] == csnr_db)
            & (df["model_id"] == model_id)
            & (df["estimator"] == estimator)
            & (df["nchunks"] == nc)
            & (df["block_dct"] == block_dct)
            & df["grad_w"]
            & df["grad_sel"]
            & df["grad_alloc"]
        ][["cr", "score0_lvc", "score0_lvc_g"]]

        # TODO: Fix non-monotonic curves causing errors

        anchor = df_test[["cr", "score0_lvc"]]
        # anchor = anchor[anchor >= anchor.cummax()]
        test = df_test[["cr", "score0_lvc_g"]]
        # test = test[test >= test.cummax()]

        try:
            bdrate = bd.bd_rate(
                anchor["cr"].to_numpy(),
                anchor["score0_lvc"].to_numpy(),
                test["cr"].to_numpy(),
                test["score0_lvc_g"].to_numpy(),
                method=interp_method,
            )
        except (IndexError, ValueError, AssertionError):
            bdrate = np.nan

        try:
            bdacc = bd.bd_psnr(
                anchor["cr"].to_numpy(),
                anchor["score0_lvc"].to_numpy(),
                test["cr"].to_numpy(),
                test["score0_lvc_g"].to_numpy(),
                method=interp_method,
            )
        except (IndexError, ValueError, AssertionError):
            bdacc = np.nan

        row = {
            "model_id": model_id,
            "estimator": estimator,
            "mode": mode,
            "nchunks": nc,
            "csnr_db": csnr_db,
            "block_dct": block_dct,
            "bdrate": bdrate,
            "bdacc": bdacc * 100,
        }

        # outdir = Path("experiments_tupu/runs/plots/test_bd")
        # outdir.mkdir(exist_ok=True, parents=True)
        # try:
        #     bd.compare_methods(
        #         anchor["cr"].to_numpy(),
        #         anchor["score0_lvc"].to_numpy(),
        #         test["cr"].to_numpy(),
        #         test["score0_lvc_g"].to_numpy(),
        #         rate_label="CR",
        #         distortion_label="Acc",
        #         figure_label="test",
        #         filepath=f"{outdir}/test_bd_{model_id}_{estimator}_{mode}_{nc}_{csnr_db}db_{'bb' if block_dct else 'ff'}.png",
        #     )
        # except (IndexError, ValueError) as e:
        #     print(f"{row}: Error: {e}")

        rows.append(row)

    return pd.DataFrame(rows)


def _get_bd_default(df) -> pd.DataFrame:
    """Calculate BD metrics from a given data frame"""

    estimators = pd.unique(df["estimator"])
    modes = pd.unique(df["mode"])
    nchunks = pd.unique(df["nchunks"])
    csnr_dbs = [val for val in pd.unique(df["csnr_db"]) if val != "inf"]
    block_dcts = pd.unique(df["block_dct"])
    interp_method = "pchip"  # akima, pchip

    rows = []
    for model_id, estimator, mode, nc, csnr_db, block_dct in itertools.product(
        MODEL_ORDER.keys(), estimators, modes, nchunks, csnr_dbs, block_dcts
    ):
        df_anchor = df[
            (df["csnr_db"] == csnr_db)
            & (df["model_id"] == model_id)
            & (df["estimator"] == "zf")
            & (df["nchunks"] == 256)
            & (df["block_dct"] == False)
            & df["grad_w"]
            & df["grad_sel"]
            & df["grad_alloc"]
        ][["cr", "score0_lvc", "score0_lvc_g"]]

        df_256 = df[
            (df["csnr_db"] == csnr_db)
            & (df["model_id"] == model_id)
            & (df["estimator"] == estimator)
            & (df["nchunks"] == 256)
            & (df["block_dct"] == block_dct)
            & df["grad_w"]
            & df["grad_sel"]
            & df["grad_alloc"]
        ][["cr", "score0_lvc", "score0_lvc_g"]]

        df_ff = df[
            (df["csnr_db"] == csnr_db)
            & (df["model_id"] == model_id)
            & (df["estimator"] == estimator)
            & (df["nchunks"] == nc)
            & (df["block_dct"] == False)
            & df["grad_w"]
            & df["grad_sel"]
            & df["grad_alloc"]
        ][["cr", "score0_lvc", "score0_lvc_g"]]

        df_zf = df[
            (df["csnr_db"] == csnr_db)
            & (df["model_id"] == model_id)
            & (df["estimator"] == "zf")
            & (df["nchunks"] == nc)
            & (df["block_dct"] == block_dct)
            & df["grad_w"]
            & df["grad_sel"]
            & df["grad_alloc"]
        ][["cr", "score0_lvc", "score0_lvc_g"]]

        df_test = df[
            (df["csnr_db"] == csnr_db)
            & (df["model_id"] == model_id)
            & (df["estimator"] == estimator)
            & (df["nchunks"] == nc)
            & (df["block_dct"] == block_dct)
            & df["grad_w"]
            & df["grad_sel"]
            & df["grad_alloc"]
        ][["cr", "score0_lvc", "score0_lvc_g"]]

        # TODO: Fix non-monotonic curves causing errors

        anchor_default = df_anchor[["cr", "score0_lvc"]]
        anchor_256 = df_256[["cr", "score0_lvc_g"]]
        anchor_ff = df_ff[["cr", "score0_lvc_g"]]
        anchor_zf = df_zf[["cr", "score0_lvc_g"]]
        anchor = df_test[["cr", "score0_lvc"]]
        test = df_test[["cr", "score0_lvc_g"]]

        def get_bd_series(
            anchor_x: pd.Series,
            anchor_y: pd.Series,
            test_x: pd.Series,
            test_y: pd.Series,
            method=interp_method,
        ):
            # Get pareto front
            # pareto_idx_ref = []
            # for cur_i, (cur_ref_x, cur_ref_y) in enumerate(zip(anchor_x, anchor_y)):
            #     is_pareto = True

            #     for i, (ref_x, ref_y) in enumerate(zip(anchor_x, anchor_y)):
            #         if cur_i == i:
            #             continue

            #         if (cur_ref_x >= ref_x) and (cur_ref_y <= ref_y):
            #             is_pareto = False
            #             break

            #     if is_pareto:
            #         pareto_idx_ref.append(cur_i)

            # anchor_x = anchor_x.loc[pareto_idx_ref]
            # anchor_y = anchor_y.loc[pareto_idx_ref]

            # pareto_idx = []
            # for cur_i, (cur_x, cur_y) in enumerate(zip(test_x, test_y)):
            #     is_pareto = True

            #     for i, (x, y) in enumerate(zip(test_x, test_y)):
            #         if cur_i == i:
            #             continue

            #         if (cur_x >= x) and (cur_y <= y):
            #             is_pareto = False
            #             break

            #     if is_pareto:
            #         pareto_idx.append(cur_i)

            # test_x = test_x.loc[pareto_idx]
            # test_y = test_y.loc[pareto_idx]

            # Get BD curves
            try:
                bdrate = bd.bd_rate(
                    anchor_x.to_numpy(),
                    anchor_y.to_numpy(),
                    test_x.to_numpy(),
                    test_y.to_numpy(),
                    method=method,
                )
            except (IndexError, ValueError, AssertionError):
                bdrate = np.nan

            try:
                bdpsnr = bd.bd_psnr(
                    anchor_x.to_numpy(),
                    anchor_y.to_numpy(),
                    test_x.to_numpy(),
                    test_y.to_numpy(),
                    method=interp_method,
                )
            except (IndexError, ValueError, AssertionError):
                bdpsnr = np.nan

            return bdrate, bdpsnr

        bdrate_default, bdacc_default = get_bd_series(
            anchor_default["cr"],
            anchor_default["score0_lvc"],
            anchor["cr"],
            anchor["score0_lvc"],
        )

        bdrate, bdacc = get_bd_series(
            anchor["cr"],
            anchor["score0_lvc"],
            test["cr"],
            test["score0_lvc_g"],
        )

        bdrate_total, bdacc_total = get_bd_series(
            anchor_default["cr"],
            anchor_default["score0_lvc"],
            test["cr"],
            test["score0_lvc_g"],
        )

        bdrate_256, bdacc_256 = get_bd_series(
            anchor_256["cr"],
            anchor_256["score0_lvc_g"],
            test["cr"],
            test["score0_lvc_g"],
        )

        bdrate_ff, bdacc_ff = get_bd_series(
            anchor_ff["cr"],
            anchor_ff["score0_lvc_g"],
            test["cr"],
            test["score0_lvc_g"],
        )

        bdrate_zf, bdacc_zf = get_bd_series(
            anchor_zf["cr"],
            anchor_zf["score0_lvc_g"],
            test["cr"],
            test["score0_lvc_g"],
        )

        row_default = {
            "model_id": model_id,
            "estimator": estimator,
            "mode": mode,
            "nchunks": nc,
            "csnr_db": csnr_db,
            "block_dct": block_dct,
            "bdrate": bdrate_default,
            "bdacc": bdacc_default * 100,
            "result": "lvc",
        }

        row = {
            "model_id": model_id,
            "estimator": estimator,
            "mode": mode,
            "nchunks": nc,
            "csnr_db": csnr_db,
            "block_dct": block_dct,
            "bdrate": bdrate,
            "bdacc": bdacc * 100,
            "result": "lvc_g",
        }

        row_total = {
            "model_id": model_id,
            "estimator": estimator,
            "mode": mode,
            "nchunks": nc,
            "csnr_db": csnr_db,
            "block_dct": block_dct,
            "bdrate": bdrate_total,
            "bdacc": bdacc_total * 100,
            "result": "total",
        }

        row_256 = {
            "model_id": model_id,
            "estimator": estimator,
            "mode": mode,
            "nchunks": nc,
            "csnr_db": csnr_db,
            "block_dct": block_dct,
            "bdrate": bdrate_256,
            "bdacc": bdacc_256 * 100,
            "result": "lvc_g_256",
        }

        row_ff = {
            "model_id": model_id,
            "estimator": estimator,
            "mode": mode,
            "nchunks": nc,
            "csnr_db": csnr_db,
            "block_dct": block_dct,
            "bdrate": bdrate_ff,
            "bdacc": bdacc_ff * 100,
            "result": "lvc_g_ff",
        }

        row_zf = {
            "model_id": model_id,
            "estimator": estimator,
            "mode": mode,
            "nchunks": nc,
            "csnr_db": csnr_db,
            "block_dct": block_dct,
            "bdrate": bdrate_zf,
            "bdacc": bdacc_zf * 100,
            "result": "lvc_g_zf",
        }

        # outdir = Path("experiments_tupu/runs/plots/test_bd")
        # outdir.mkdir(exist_ok=True, parents=True)
        # try:
        #     bd.compare_methods(
        #         anchor["cr"].to_numpy(),
        #         anchor["score0_lvc"].to_numpy(),
        #         test["cr"].to_numpy(),
        #         test["score0_lvc_g"].to_numpy(),
        #         rate_label="CR",
        #         distortion_label="Acc",
        #         figure_label="test",
        #         filepath=f"{outdir}/test_bd_{model_id}_{estimator}_{mode}_{nc}_{csnr_db}db_{'bb' if block_dct else 'ff'}.png",
        #     )
        # except (IndexError, ValueError) as e:
        #     print(f"{row}: Error: {e}")

        rows.append(row_default)
        rows.append(row)
        rows.append(row_total)
        rows.append(row_256)
        rows.append(row_ff)
        rows.append(row_zf)

    return pd.DataFrame(rows)


def _pareto(
    df: pd.DataFrame, low_cols: List[str] = [], high_cols: List[str] = []
) -> pd.DataFrame:
    """Get Pareto front of selected columns in a dataframe"""

    pareto_idx = []
    for cur_row in df.itertuples():
        cur_idx = cur_row.Index
        cur_low_vals = [cur_row._asdict()[col] for col in low_cols]
        cur_high_vals = [cur_row._asdict()[col] for col in high_cols]
        # cur_t_host = cur_row.t_host
        # cur_pow_mean = cur_row.pow_mean
        # cur_iou_mean = cur_row.iou_mean
        is_pareto = True

        for row in df.itertuples():
            idx = row.Index

            if cur_idx == idx:
                continue

            # t_host = row.t_host
            # pow_mean = row.pow_mean
            # iou_mean = row.iou_mean
            low_vals = [row._asdict()[col] for col in low_cols]
            high_vals = [row._asdict()[col] for col in high_cols]

            if all(
                [cur_val >= val for cur_val, val in zip(cur_low_vals, low_vals)]
            ) and all(
                [cur_val <= val for cur_val, val in zip(cur_high_vals, high_vals)]
            ):
                is_pareto = False
                break

        if is_pareto:
            pareto_idx.append(cur_idx)

    return df.loc[pareto_idx]


def _heatmaps(
    image: np.ndarray,
    w: int,
    h: int,
    scale: float,
    rows: int,
    cols: int,
    label: str | None,
    show: bool = True,
    save: str | Path | None = None,
):
    fig = make_subplots(rows=rows, cols=cols)

    if image.shape[0] != (rows * cols):
        raise ValueError("Number of channels does not correspond to number of subplots")

    # https://community.plotly.com/t/how-to-set-log-scale-for-z-axis-on-a-heatmap/292/8
    def colorbar(nmin, nmax):
        labels = np.sort(
            np.concatenate(
                [
                    np.linspace(10**nmin, 10**nmax, 10),
                    10 ** np.linspace(nmin, nmax, 10),
                ]
            )
        )
        # vals = np.linspace(nmin, nmax, nmax+nmin+1)

        return dict(
            tick0=nmin,
            # title="Log Scale",
            tickmode="array",
            tickvals=np.log10(labels),
            ticktext=[f"{x:.2e}" for x in labels],
            # tickvals=vals,
            # ticktext=[f"{10**x:.2e}" for x in labels],
            # tickvals=np.linspace(nmin, nmax, nmax - nmin + 1),
            # ticktext=[
            #     f"{x:.0e}" for x in 10 ** np.linspace(nmin, nmax, nmax - nmin + 1)
            # ],
        )

    zero_mask = np.logical_or(image == 0.0, image == np.nan)
    img_nz = image[~zero_mask]
    image[zero_mask] = np.nan

    gmin = img_nz.min()
    gmax = img_nz.max()
    nmin = int(np.floor(np.log10(gmin)))
    nmax = int(np.ceil(np.log10(gmax)))

    for ch, (r, c) in zip(
        image, itertools.product(range(1, rows + 1), range(1, cols + 1))
    ):
        fig.add_trace(
            go.Heatmap(
                z=np.log10(ch),
                customdata=ch,
                hovertemplate="x: %{x} <br>" + "y: %{y} <br>" + "z: %{customdata:.2e}",
                # colorbar=colorbar(nmin, nmax),
                colorbar=dict(title="10^"),
                # colorscale="Inferno",
                # reversescale=True,
                zmin=np.log10(gmin),
                zmax=np.log10(gmax),
            ),
            row=r,
            col=c,
        )

    fig.update_yaxes(
        autorange="reversed", scaleanchor="x", scaleratio=1, constrain="domain"
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1, constrain="domain")

    if label is not None:
        fig.update_layout(title=label)

    fig.update_layout(
        width=w,
        height=h,
        font=dict(size=25),
        autosize=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    if label is not None:
        fig.update_layout(
            autosize=False,
            margin=dict(l=0, r=0, t=50, b=0),
        )

    if save:
        fig.write_image(save, scale=scale)
        fig.write_html(Path(save).with_suffix(".html"))

    if show:
        fig.show()


def acc_vs_csnr(
    outdir: Path,
    w: int,
    h: int,
    scale: float = DEFAULT_SCALE,
    do_print: bool = True,
    do_show: bool = True,
    do_reload: bool = False,
) -> Path:
    if do_print:
        print("Accuracy vs CSNR")

    probe_results_list, df_full = collect_df("acc_vs_csnr", do_reload=do_reload)

    df_full, duplicates = _remove_duplicates(df_full)
    if do_print:
        print("Duplicates:", len(duplicates))

    df_full = df_full.sort_values(
        by=["probe_model_id", "model_id"], key=lambda x: x.map(MODEL_ORDER)
    )
    df_full = df_full.sort_values(by=DEFAULT_ID_COLS[2:])

    (baseline0, _) = get_baseline(probe_results_list)
    if do_print:
        print(baseline0)

    estimator = "zf"
    mode = 444
    block_dct = False
    nchunks = 256

    # msg = f"{estimator}, {mode}, {'bb' if block_dct else 'ff'}"
    df = df_full[df_full["csnr_db"] != "inf"]
    df = _filter_df(df, estimator, mode, block_dct, nchunks)[
        ["model_id", "nchunks", "cr", "csnr_db", "score0_lvc", "score0_lvc_g"]
    ]
    df_g = (
        df.copy()
        .drop("score0_lvc", axis=1)
        .rename(columns={"score0_lvc_g": "accuracy"})
    )
    df_g["scheme"] = "CV-Cast"
    df = df.drop("score0_lvc_g", axis=1).rename(columns={"score0_lvc": "accuracy"})
    df["scheme"] = "LCT"

    df = pd.concat([df, df_g])
    df = df.query(f"csnr_db >= {CSNR_MIN} and csnr_db <= {CSNR_MAX}")

    df = df.replace("fastseg_small", "fastseg_small (mIoU)")
    df = df.replace("fastseg_large", "fastseg_large (mIoU)")
    df = df.replace("yolov8_n", "yolov8_n (mAP)")
    df = df.replace("yolov8_s", "yolov8_s (mAP)")
    df = df.replace("yolov8_l", "yolov8_l (mAP)")
    df = df.rename(columns={"cr": "CR", "csnr_db": "CSNR (dB)"})

    fig = px.line(
        df,
        x="CSNR (dB)",
        y="accuracy",
        color="CR",
        facet_col="model_id",
        facet_col_spacing=0.025,
        # facet_row="nchunks",
        # symbol="scheme",
        labels="scheme",
        line_dash="scheme",
        line_dash_sequence=[DASH, "solid"],
        # symbol_sequence=["cross-thin-open", "x-thin-open"],
        markers=True,
        range_x=[CSNR_MIN - 1, CSNR_MAX + 1],
        range_y=[0.0, None],
    )

    _add_baseline(fig, baseline0)

    fig.update_xaxes(
        zerolinecolor="grey",
        zeroline=True,
        gridcolor="grey",
        griddash="dot",
        tickmode="array",
        tickvals=[-10, -5, 0, 5, 10, 15, 20, 25, 30],
        ticktext=["-10", "", "0", "", "10", "", "20", "", "30"],
    )
    fig.update_yaxes(
        tickformat=".0%",
        zerolinecolor="grey",
        zeroline=True,
        matches=None,
        showticklabels=True,
        gridcolor="grey",
        griddash="dot",
        minor=dict(showgrid=True),
    )
    fig.update_layout(
        plot_bgcolor="white",
        width=w,
        height=h,
        showlegend=False,
        # legend=dict(orientation="h", x=0.0, y=-0.2),
        margin=dict(l=0, r=0, t=0, b=80),
        yaxis1=dict(range=[0.0, baseline0["fastseg_small"] * 1.1]),
        yaxis2=dict(range=[0.0, baseline0["fastseg_large"] * 1.1]),
        yaxis3=dict(range=[0.0, baseline0["yolov8_n"] * 1.1]),
        yaxis4=dict(range=[0.0, baseline0["yolov8_s"] * 1.1]),
        yaxis5=dict(range=[0.0, baseline0["yolov8_l"] * 1.1]),
    )
    fig.update_traces(
        marker_size=MARKER_SIZE,
        marker_line_width=MARKER_WIDTH,
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    fname = f"{outdir}/score_csnr_zf_444_ff"
    fig.write_image(f"{fname}.png", scale=scale)
    fig.write_html(f"{fname}.html")

    if do_show:
        fig.show()

    return Path(f"{fname}.png")


def acc_vs_csnr_sionna(
    outdir: Path,
    w: int,
    h: int,
    scale: float = DEFAULT_SCALE,
    do_print: bool = True,
    do_show: bool = True,
    do_reload: bool = False,
):
    if do_print:
        print("Sionna Accuracy vs CSNR")

    probe_results_list, df_sionna = collect_df(
        "acc_vs_csnr_sionna", do_reload=do_reload
    )

    df_sionna, duplicates = _remove_duplicates(df_sionna)
    if do_print:
        print("Duplicates:", len(duplicates))

    df_sionna = df_sionna.sort_values(by=DEFAULT_ID_COLS[2:])
    df_sionna = df_sionna.sort_values(
        by=["probe_model_id", "model_id"], key=lambda x: x.map(MODEL_ORDER)
    )

    (baseline0, _) = get_baseline(probe_results_list)
    if do_print:
        print(baseline0)

    estimator = "zf"
    mode = 444
    block_dct = False
    nchunks = 256

    df = df_sionna[(df_sionna["csnr_db"] != "inf")]
    df = _filter_df(df, estimator, mode, block_dct, nchunks)[
        ["model_id", "nchunks", "cr", "csnr_db", "score0_lvc", "score0_lvc_g"]
    ]
    df_g = (
        df.copy()
        .drop("score0_lvc", axis=1)
        .rename(columns={"score0_lvc_g": "accuracy"})
    )
    df_g["scheme"] = "CV-Cast"
    df = df.drop("score0_lvc_g", axis=1).rename(columns={"score0_lvc": "accuracy"})
    df["scheme"] = "LCT"

    df = pd.concat([df, df_g])
    df = df.query(f"csnr_db >= {CSNR_MIN} and csnr_db <= {CSNR_MAX}")

    df = df.replace("fastseg_small", "fastseg_small (mIoU)")
    df = df.replace("fastseg_large", "fastseg_large (mIoU)")
    df = df.replace("yolov8_n", "yolov8_n (mAP)")
    df = df.replace("yolov8_s", "yolov8_s (mAP)")
    df = df.replace("yolov8_l", "yolov8_l (mAP)")
    df = df.rename(columns={"cr": "CR"})
    df = df.rename(columns={"csnr_db": "Eb/N0 (dB)"})

    fig = px.line(
        df,
        x="Eb/N0 (dB)",
        y="accuracy",
        color="CR",
        facet_col="model_id",
        facet_col_spacing=0.025,
        # facet_row="model_id",
        # facet_row="nchunks",
        # symbol="scheme",
        labels="scheme",
        line_dash="scheme",
        line_dash_sequence=[DASH, "solid"],
        # symbol_sequence=["cross-thin-open", "x-thin-open"],
        markers=True,
        range_x=[CSNR_MIN - 1, CSNR_MAX + 1],
        range_y=[0.0, None],
    )

    _add_baseline(
        fig,
        baseline0,
        # model_ids=["fastseg_large", "fastseg_small"],
        # col_major=True,
    )

    fig.update_xaxes(
        title="Eb/N0 (dB)",
        zerolinecolor="grey",
        zeroline=True,
        gridcolor="grey",
        griddash="dot",
        tickmode="array",
        tickvals=[-10, -5, 0, 5, 10, 15, 20, 25, 30],
        ticktext=["-10", "", "0", "", "10", "", "20", "", "30"],
        # minor=dict(showgrid=True),
    )
    fig.update_yaxes(
        tickformat=".0%",
        zerolinecolor="grey",
        zeroline=True,
        matches=None,
        showticklabels=True,
        gridcolor="grey",
        griddash="dot",
        minor=dict(showgrid=True),
    )
    fig.update_layout(
        plot_bgcolor="white",
        width=w,
        height=h,
        showlegend=False,
        # legend=dict(orientation="h", x=0.0, y=-0.2),
        margin=dict(l=0, r=0, t=0, b=80),
        yaxis1=dict(range=[0.0, baseline0["fastseg_small"] * 1.1]),
        yaxis2=dict(range=[0.0, baseline0["fastseg_large"] * 1.1]),
        yaxis3=dict(range=[0.0, baseline0["yolov8_n"] * 1.1]),
        yaxis4=dict(range=[0.0, baseline0["yolov8_s"] * 1.1]),
        yaxis5=dict(range=[0.0, baseline0["yolov8_l"] * 1.1]),
    )
    fig.update_traces(
        marker_size=MARKER_SIZE,
        marker_line_width=MARKER_WIDTH,
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    fname = f"{outdir}/score_sionna_csnr_zf_444_ff"
    fig.write_image(f"{fname}.png", scale=scale)
    fig.write_html(f"{fname}.html")

    if do_show:
        fig.show()

    return Path(f"{fname}.png")


def dummy_legends(
    outdir: Path,
    w: int,
    h: int,
    scale: float = DEFAULT_SCALE,
    do_print: bool = True,
    do_show: bool = True,
    do_reload: bool = False,
) -> Tuple[Path, Path]:
    if do_print:
        print("(Dummy Lengends) Accuracy vs CSNR")

    probe_results_list, df_full = collect_df("acc_vs_csnr", do_reload=do_reload)

    df_full, duplicates = _remove_duplicates(df_full)
    if do_print:
        print("Duplicates:", len(duplicates))

    df_full = df_full.sort_values(
        by=["probe_model_id", "model_id"], key=lambda x: x.map(MODEL_ORDER)
    )
    df_full = df_full.sort_values(by=DEFAULT_ID_COLS[2:])

    (baseline0, _) = get_baseline(probe_results_list)
    if do_print:
        print(baseline0)

    estimator = "zf"
    mode = 444
    block_dct = False
    nchunks = 256

    # msg = f"{estimator}, {mode}, {'bb' if block_dct else 'ff'}"
    df = df_full[df_full["csnr_db"] != "inf"]
    df = _filter_df(df, estimator, mode, block_dct, nchunks)[
        ["model_id", "nchunks", "cr", "csnr_db", "score0_lvc", "score0_lvc_g"]
    ]
    df_g = (
        df.copy()
        .drop("score0_lvc", axis=1)
        .rename(columns={"score0_lvc_g": "accuracy"})
    )
    df_g["scheme"] = "CV-Cast"
    df = df.drop("score0_lvc_g", axis=1).rename(columns={"score0_lvc": "accuracy"})
    df["scheme"] = "LCT"

    df = pd.concat([df, df_g])
    df = df.replace("fastseg_small", "fastseg_small (mIoU)")
    df = df.replace("fastseg_large", "fastseg_large (mIoU)")
    df = df.replace("yolov8_n", "yolov8_n (mAP)")
    df = df.replace("yolov8_s", "yolov8_s (mAP)")
    df = df.replace("yolov8_l", "yolov8_l (mAP)")
    df = df.rename(columns={"cr": "CR", "csnr_db": "CSNR (dB)"})

    # Scheme legend
    fig = px.line(
        df,
        x="CSNR (dB)",
        y="accuracy",
        # color="CR",
        facet_col="model_id",
        facet_col_spacing=0.025,
        # facet_row="nchunks",
        # symbol="scheme",
        labels="scheme",
        line_dash="scheme",
        line_dash_sequence=[DASH, "solid"],
        # symbol_sequence=["cross-thin-open", "x-thin-open"],
        markers=True,
        range_y=[0.0, None],
    )

    _add_baseline(fig, baseline0)

    fig.update_xaxes(
        zerolinecolor="grey",
        zeroline=True,
        gridcolor="grey",
        griddash="dot",
        tickmode="array",
        tickvals=[-10, -5, 0, 5, 10, 15, 20, 25, 30],
        ticktext=["-10", "", "0", "", "10", "", "20", "", "30"],
    )
    fig.update_yaxes(
        tickformat=".0%",
        zerolinecolor="grey",
        zeroline=True,
        matches=None,
        showticklabels=True,
        gridcolor="grey",
        griddash="dot",
        minor=dict(showgrid=True),
    )
    fig.update_layout(
        legend_title="Line type:",
        plot_bgcolor="white",
        width=w,
        height=h,
        legend=dict(orientation="h", x=0.0, y=-0.2),
        margin=dict(l=20, r=20, t=20, b=0),
        yaxis1=dict(range=[0.0, baseline0["fastseg_small"] * 1.1]),
        yaxis2=dict(range=[0.0, baseline0["fastseg_large"] * 1.1]),
        yaxis3=dict(range=[0.0, baseline0["yolov8_n"] * 1.1]),
        yaxis4=dict(range=[0.0, baseline0["yolov8_s"] * 1.1]),
        yaxis5=dict(range=[0.0, baseline0["yolov8_l"] * 1.1]),
    )
    fig.update_traces(
        marker_size=MARKER_SIZE,
        marker_line_width=MARKER_WIDTH,
        line_color="#000000",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    fname_scheme = f"{outdir}/dummy_legends_scheme"
    fig.write_image(f"{fname_scheme}.png", scale=scale)
    fig.write_html(f"{fname_scheme}.html")

    if do_show:
        fig.show()

    # CR legend
    dft = df.query("scheme == 'LCT'").astype({"CR": str})
    dft.loc[:, "CR"] = dft["CR"].map(lambda x: f"CR={x}")

    fig = px.line(
        dft,
        x="CSNR (dB)",
        y="accuracy",
        color="CR",
        facet_col="model_id",
        facet_col_spacing=0.025,
        # facet_row="nchunks",
        # symbol="scheme",
        # labels="scheme",
        # line_dash="scheme",
        # line_dash_sequence=[DASH, "solid"],
        # symbol_sequence=["cross-thin-open", "x-thin-open"],
        # markers=True,
        range_y=[0.0, None],
    )

    _add_baseline(fig, baseline0)

    fig.update_xaxes(
        zerolinecolor="grey",
        zeroline=True,
        gridcolor="grey",
        griddash="dot",
        tickmode="array",
        tickvals=[-10, -5, 0, 5, 10, 15, 20, 25, 30],
        ticktext=["-10", "", "0", "", "10", "", "20", "", "30"],
    )
    fig.update_yaxes(
        tickformat=".0%",
        zerolinecolor="grey",
        zeroline=True,
        matches=None,
        showticklabels=True,
        gridcolor="grey",
        griddash="dot",
        minor=dict(showgrid=True),
    )
    fig.update_layout(
        legend_title="Color:",
        plot_bgcolor="white",
        width=w,
        height=h,
        legend=dict(orientation="h", x=0.0, y=-0.2),
        margin=dict(l=20, r=20, t=20, b=0),
        yaxis1=dict(range=[0.0, baseline0["fastseg_small"] * 1.1]),
        yaxis2=dict(range=[0.0, baseline0["fastseg_large"] * 1.1]),
        yaxis3=dict(range=[0.0, baseline0["yolov8_n"] * 1.1]),
        yaxis4=dict(range=[0.0, baseline0["yolov8_s"] * 1.1]),
        yaxis5=dict(range=[0.0, baseline0["yolov8_l"] * 1.1]),
    )
    # fig.update_traces(
    #     marker_size=MARKER_SIZE,
    #     marker_line_width=MARKER_WIDTH,
    #     # line_color="#000000",
    # )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    fname_cr = f"{outdir}/dummy_legends_cr"
    fig.write_image(f"{fname_cr}.png", scale=scale)
    fig.write_html(f"{fname_cr}.html")

    if do_show:
        fig.show()

    return (Path(f"{fname_scheme}.png"), Path(f"{fname_cr}.png"))


def apply_dummy_legends(
    outdir: Path,
    w: int,
    h: int,
    inp_files: List[Path],
    fname_scheme: Path,
    fname_cr: Path,
    scale: float = DEFAULT_SCALE,
    do_print: bool = True,
    do_show: bool = True,
    do_reload: bool = False,
):
    if do_print:
        print("Replacing legends...")

    w = int(w * scale)
    h = int(h * scale)
    scheme_crop = Image.open(fname_scheme).crop((0, h - 90, w, h - 10))
    scheme_crop.save(outdir / "legend_scheme.png")
    cr_crop = Image.open(fname_cr).crop((0, h - 90, w, h - 10))
    cr_crop.save(outdir / "legend_cr.png")

    for inp_file in inp_files:
        canvas = Image.open(inp_file)
        canvas.paste(scheme_crop, (0, h - 145))
        canvas.paste(cr_crop, (0, h - 70))
        canvas.save(inp_file)


def disparity_10db_0_25(
    outdir: Path,
    w: int,
    h: int,
    scale: float = DEFAULT_SCALE,
    do_print: bool = True,
    do_show: bool = True,
    do_save: bool = True,
    do_reload: bool = False,
):
    if do_print:
        print("Disparity (10 dB, 0.25):")

    _, df_disparity = collect_df("disparity", do_reload=do_reload)

    df_disparity, duplicates = _remove_duplicates(df_disparity)
    if do_print:
        print("Duplicates:", len(duplicates))

    # mode_renamer = {"probe_model_id": "probe", "model_id": "eval"}
    model_renamer = {
        "fastseg_small": "fss",
        "fastseg_large": "fsl",
        "yolov8_n": "y8n",
        "yolov8_s": "y8s",
        "yolov8_l": "y8l",
    }

    df_disp = df_disparity.query("csnr_db == 10 and cr == 0.25")[
        ["probe_model_id", "model_id", "score0_lvc", "score0_lvc_g"]
    ].sort_values(by=["probe_model_id", "model_id"], key=lambda x: x.map(MODEL_ORDER))
    gt = df_disp.query("probe_model_id == model_id").set_index("probe_model_id")

    df_disp_table = df_disp.pivot(
        index="probe_model_id", columns="model_id", values="score0_lvc_g"
    )
    df_disp_table = df_disp_table.sort_index(
        axis="index", key=lambda x: x.map(MODEL_ORDER)
    )
    df_disp_table = df_disp_table.sort_index(
        axis="columns", key=lambda x: x.map(MODEL_ORDER)
    )

    df_disp_table_diff = df_disp_table.sub(gt["score0_lvc_g"], axis="columns")
    df_disp_table_diff = (df_disp_table_diff * 100).round(2)

    df_disp_table_diff = df_disp_table_diff.rename(
        columns=model_renamer, index=model_renamer
    )
    df_disp_table_diff.index.names = ["probe"]
    df_disp_table_diff.columns.names = ["eval"]

    if do_print:
        print(df_disp_table_diff)

    fig = px.imshow(df_disp_table_diff, text_auto=True)
    fig.update_xaxes(side="top")
    fig.update_layout(
        plot_bgcolor="white",
        width=w,
        height=h,
        coloraxis_showscale=False,
        margin=dict(l=20, r=5, t=20, b=0),
    )

    if do_save:
        fname = outdir / "disparity_10db_cr0_25"
        fig.write_image(f"{fname}.png", scale=scale)
        fig.write_html(f"{fname}.html")

    if do_show:
        fig.show()


def disparity_10db_bd(
    outdir: Path,
    w: int,
    h: int,
    h2: int,
    scale: float = DEFAULT_SCALE,
    do_print: bool = True,
    do_show: bool = True,
    do_save: bool = True,
    do_reload: bool = False,
):
    if do_print:
        print("Disparity (10 dB, BD metrics):")

    _, df_disparity = collect_df("disparity", do_reload=do_reload)

    df_disparity, duplicates = _remove_duplicates(df_disparity)
    if do_print:
        print("Duplicates:", len(duplicates))

    # mode_renamer = {"probe_model_id": "probe", "model_id": "eval"}
    model_renamer = {
        "fastseg_small": "fss",
        "fastseg_large": "fsl",
        "yolov8_n": "y8n",
        "yolov8_s": "y8s",
        "yolov8_l": "y8l",
    }

    df_disp = df_disparity.query("csnr_db == 10")[
        ["probe_model_id", "model_id", "cr", "score0_lvc", "score0_lvc_g"]
    ].sort_values(by=["probe_model_id", "model_id"], key=lambda x: x.map(MODEL_ORDER))
    gt = df_disp.query("probe_model_id == model_id").set_index("probe_model_id")

    df_ref_dict = {}

    # First, collect reference results
    for model_id in model_renamer.keys():
        for probe_model_id in model_renamer.keys():
            if probe_model_id != model_id:
                continue

            df_ref_dict[model_id] = df_disp.query(
                "probe_model_id == @probe_model_id and model_id == @model_id"
            ).sort_values(by="cr")

    interp_method = "pchip"  # akima, pchip

    res = []

    # Then, compare all results to the reference results
    for model_id in model_renamer.keys():
        for probe_model_id in model_renamer.keys():
            df_ref = df_ref_dict[model_id]
            df_sub = df_disp.query(
                "probe_model_id == @probe_model_id and model_id == @model_id"
            ).sort_values(by="cr")

            bdrate = bd.bd_rate(
                rate_anchor=df_ref["cr"].to_numpy(),
                dist_anchor=df_ref["score0_lvc_g"].to_numpy(),
                rate_test=df_sub["cr"].to_numpy(),
                dist_test=df_sub["score0_lvc_g"].to_numpy(),
                method=interp_method,
            )

            bdacc = bd.bd_psnr(
                rate_anchor=df_ref["cr"].to_numpy(),
                dist_anchor=df_ref["score0_lvc_g"].to_numpy(),
                rate_test=df_sub["cr"].to_numpy(),
                dist_test=df_sub["score0_lvc_g"].to_numpy(),
                method=interp_method,
            )

            bdrate_lvc = bd.bd_rate(
                rate_anchor=df_ref["cr"].to_numpy(),
                dist_anchor=df_ref["score0_lvc"].to_numpy(),
                rate_test=df_sub["cr"].to_numpy(),
                dist_test=df_sub["score0_lvc_g"].to_numpy(),
                method=interp_method,
            )

            bdacc_lvc = bd.bd_psnr(
                rate_anchor=df_ref["cr"].to_numpy(),
                dist_anchor=df_ref["score0_lvc"].to_numpy(),
                rate_test=df_sub["cr"].to_numpy(),
                dist_test=df_sub["score0_lvc_g"].to_numpy(),
                method=interp_method,
            )

            # outdir = Path("experiments_tupu/runs/plots/test_bd")
            # outdir.mkdir(exist_ok=True, parents=True)
            # try:
            #     bd.compare_methods(
            #         df_ref["cr"].to_numpy(),
            #         df_ref["score0_lvc_g"].to_numpy(),
            #         df_sub["cr"].to_numpy(),
            #         df_sub["score0_lvc_g"].to_numpy(),
            #         rate_label="CR",
            #         distortion_label="Acc",
            #         figure_label="test",
            #         filepath=f"{outdir}/bd_disparity_probe_{probe_model_id}_eval_{model_id}.png",
            #     )
            # except (IndexError, ValueError) as e:
            #     print(f"probe {probe_model_id} eval {model_id}: Error: {e}")

            res.append(
                {
                    "probe_model_id": probe_model_id,
                    "model_id": model_id,
                    "bdrate": bdrate,
                    "bdacc": bdacc,
                    "bdrate_lvc": bdrate_lvc,
                    "bdacc_lvc": bdacc_lvc,
                }
            )

    df_disp = pd.DataFrame(res)
    if do_print:
        print(df_disp)

    x_axis = "evaluated model"
    y_axis = "optimization target"

    df_disp_table = df_disp.pivot(
        index="probe_model_id", columns="model_id", values="bdrate"
    )
    df_disp_table = df_disp_table.sort_index(
        axis="index", key=lambda x: x.map(MODEL_ORDER)
    )
    df_disp_table = df_disp_table.sort_index(
        axis="columns", key=lambda x: x.map(MODEL_ORDER)
    )
    df_disp_table = df_disp_table.rename(columns=model_renamer, index=model_renamer)
    df_disp_table.index.names = [y_axis]
    df_disp_table.columns.names = [x_axis]
    # df_disp_table = df_disp_table * 100
    df_disp_table = df_disp_table.round(1)

    df_disp_table_lvc = df_disp.pivot(
        index="probe_model_id", columns="model_id", values="bdrate_lvc"
    )
    df_disp_table_lvc = df_disp_table_lvc.sort_index(
        axis="index", key=lambda x: x.map(MODEL_ORDER)
    )
    df_disp_table_lvc = df_disp_table_lvc.sort_index(
        axis="columns", key=lambda x: x.map(MODEL_ORDER)
    )
    df_disp_table_lvc = df_disp_table_lvc.rename(
        columns=model_renamer, index=model_renamer
    )
    df_disp_table_lvc.index.names = [y_axis]
    df_disp_table_lvc.columns.names = [x_axis]
    # df_disp_table_lvc = df_disp_table_lvc * 100
    df_disp_table_lvc = df_disp_table_lvc.round(1)

    if do_print:
        print("CV-Cast probe vs CV-Cast eval:")
        print(df_disp_table)
        print("LCT probe vs CV-Cast eval:")
        print(df_disp_table_lvc)

    disp_table = np.array(
        [
            df_disp_table.to_numpy(),
            df_disp_table_lvc.to_numpy(),
        ]
    )

    colorscale = px.colors.diverging.RdBu[::-1]
    # min/max values symmetric around 0
    absmax = max(abs(disp_table.min()), abs(disp_table.max()))
    zmin = -absmax
    zmax = absmax
    fontsize = 22

    # CV-Cast vs CV-Cast
    fig = px.imshow(
        df_disp_table,
        text_auto=True,
        zmin=zmin,
        zmax=zmax,
        color_continuous_scale=colorscale,
    )
    fig.update_xaxes(side="top")
    fig.update_layout(
        plot_bgcolor="white",
        width=w,
        height=h,
        coloraxis_showscale=False,
        margin=dict(l=20, r=5, t=20, b=0),
        font=dict(size=fontsize),
    )

    if do_save:
        fname = outdir / "disparity_10db_bd"
        fig.write_image(f"{fname}.png", scale=scale)
        fig.write_html(f"{fname}.html")

    if do_show:
        fig.show()

    # LCT vs CV-Cast
    fig = px.imshow(
        df_disp_table_lvc,
        text_auto=True,
        zmin=zmin,
        zmax=zmax,
        color_continuous_scale=colorscale,
    )
    fig.update_xaxes(side="top")
    fig.update_layout(
        plot_bgcolor="white",
        width=w,
        height=h,
        coloraxis_showscale=False,
        margin=dict(l=20, r=5, t=20, b=0),
        font=dict(size=fontsize),
    )

    if do_save:
        fname = outdir / "disparity_10db_bd_lct"
        fig.write_image(f"{fname}.png", scale=scale)
        fig.write_html(f"{fname}.html")

    if do_show:
        fig.show()

    # Combined
    fig = px.imshow(
        disp_table,
        text_auto=True,
        facet_col=0,
        labels={"x": x_axis, "y": y_axis},
        x=["fss", "fsl", "y8n", "y8s", "y8l"],
        y=["fss", "fsl", "y8n", "y8s", "y8l"],
        zmin=zmin,
        zmax=zmax,
        color_continuous_scale=colorscale,
    )
    fig.update_xaxes(side="top")
    fig.update_layout(
        plot_bgcolor="white",
        width=w,
        height=h2,
        coloraxis_showscale=False,
        margin=dict(l=20, r=5, t=20, b=0),
        # font=dict(size=fontsize),
    )

    for anno in fig["layout"]["annotations"]:
        anno["text"] = ""

    if do_show:
        fig.show()

    if do_save:
        fname = outdir / "disparity_10db_bd_comb"
        fig.write_image(f"{fname}.png", scale=scale)
        fig.write_html(f"{fname}.html")


# def grad_norm(
#     outdir: Path,
#     w: int,
#     h: int,
#     scale: float = DEFAULT_SCALE,
#     do_print: bool = True,
#     do_show: bool = True,
#     do_save: bool = True,
#     do_reload: bool = False,
# ):
#     for probe_results, outdir in zip(probe_results_list, outdirs):
#         print(f"--- {outdir} ---")

#         for model_id, res in probe_results.items():
#             for nchunks, grads_norm in res["grads_norm"].items():
#                 plot_channels(
#                     grads_norm.numpy(),
#                     f"grads_norm ({model_id}, {nchunks})",
#                     show=False,
#                     save=outdir / f"grad_norm_{model_id}_{nchunks}.png",
#                 )


def jpeg_grace(
    outdir: Path,
    w: int,
    h: int,
    scale: float = DEFAULT_SCALE,
    do_print: bool = True,
    do_show: bool = True,
    do_save: bool = True,
    do_reload: bool = False,
):
    if do_print:
        print("JPEG vs. GRACE")

    probe_results_list, df_jpeg_grace = collect_df("jpeg_grace", do_reload=do_reload)

    id_cols_jpeg = DEFAULT_ID_COLS + ["codec", "param"]

    df_jpeg_grace = df_jpeg_grace.sort_values(by=id_cols_jpeg[2:]).sort_values(
        by=["probe_model_id", "model_id"], key=lambda x: x.map(MODEL_ORDER)
    )

    (baseline0, _) = get_baseline(probe_results_list)
    if do_print:
        print(baseline0)

    df_jpeg_grace_comp = df_jpeg_grace[
        [
            "model_id",
            "cr",
            "eq_cr",
            "csnr_db",
            "codec",
            "param",
            "enc_size",
            "nbits_per_sym",
            "score0_lvc",
            "score0_lvc_g",
        ]
    ].query("csnr_db == 'inf' and codec in ['grace', 'jpeg']")

    df_jpeg_grace_comp = (
        df_jpeg_grace_comp.drop_duplicates(subset=["model_id", "codec", "enc_size"])
        .sort_values(by=["codec", "enc_size"])
        .sort_values(by=["model_id"], key=lambda x: x.map(MODEL_ORDER))
    )

    # df_jpeg_grace_comp["modulation"] = [f"{2**nb}-QAM" for nb in df_jpeg_grace_comp["nbits_per_sym"]]
    df_jpeg_grace_comp = df_jpeg_grace_comp.round({"eq_cr": 3})

    df_grace_comp = (
        df_jpeg_grace_comp.query("codec == 'grace'")
        .copy()
        .drop("score0_lvc", axis=1)
        .rename(columns={"score0_lvc_g": "accuracy"})
    )
    df_grace_comp["scheme"] = "unbounded GRACE"

    df_jpeg_comp = (
        df_jpeg_grace_comp.query("codec == 'jpeg'")
        .copy()
        .drop("score0_lvc_g", axis=1)
        .rename(columns={"score0_lvc": "accuracy"})
    )
    df_jpeg_comp["scheme"] = "JPEG"

    df_jpeg_grace_comp = pd.concat([df_jpeg_comp, df_grace_comp])

    fig = px.scatter(
        df_jpeg_grace_comp,
        x="enc_size",
        y="accuracy",
        # color="param",
        color="scheme",
        facet_col="model_id",
        facet_col_spacing=0.04,
        # facet_row="nchunks",
        # symbol="modulation",
        # labels="modulation",
        # line_dash="eq_cr",
        # line_dash_sequence=["solid", DASH, "dot"],
        # symbol_sequence=["cross-thin-open", "x-thin-open", "y-down"],
        # markers=True,
        range_y=[0.0, None],
        # title="JPEG vs GRACE",
    )

    _add_baseline(fig, baseline0)

    fig.update_xaxes(
        zerolinecolor="grey",
        zeroline=True,
        gridcolor="grey",
        griddash="dot",
        tickmode="array",
        # tickvals=[0, 5, 10, 15, 20, 30],
        matches=None,
    )
    fig.update_yaxes(
        tickformat=".0%",
        zerolinecolor="grey",
        zeroline=True,
        matches=None,
        showticklabels=True,
        gridcolor="grey",
        griddash="dot",
        minor=dict(showgrid=True),
    )
    fig.update_layout(
        plot_bgcolor="white",
        width=w,
        height=h,
        legend=dict(orientation="h", x=0.0, y=-0.2),
        margin=dict(l=20, r=20, t=20, b=5),
        font=dict(size=15),
    )

    try:
        fig.update_layout(
            yaxis1=dict(range=[0.0, baseline0["fastseg_small"] * 1.1]),
            yaxis2=dict(range=[0.0, baseline0["fastseg_large"] * 1.1]),
            yaxis3=dict(range=[0.0, baseline0["yolov8_n"] * 1.1]),
            yaxis4=dict(range=[0.0, baseline0["yolov8_s"] * 1.1]),
            yaxis5=dict(range=[0.0, baseline0["yolov8_l"] * 1.1]),
        )
    except KeyError:
        pass

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    if do_save:
        fname = f"{outdir}/jpeg_grace_comparison"
        fig.write_image(f"{fname}.png", scale=scale)
        fig.write_html(f"{fname}.html")

    if do_show:
        fig.show()


def jpeg_grace_sionna(
    outdir: Path,
    w: int,
    h: int,
    scale: float = DEFAULT_SCALE,
    codec: Literal["jpeg_grace", "tcm"] = "jpeg_grace",
    do_print: bool = True,
    do_show: bool = True,
    do_save: bool = True,
    do_reload: bool = False,
):
    if do_print:
        print(f"Sionna {codec.upper()}")

    # Collect JPEG
    res_name = f"{codec}_sionna"
    probe_results_list, df_full_jpeg = collect_df(res_name, do_reload=do_reload)

    id_cols_jpeg = DEFAULT_ID_COLS + ["codec", "param"]

    df_full_jpeg = df_full_jpeg.sort_values(by=id_cols_jpeg[2:]).sort_values(
        by=["probe_model_id", "model_id"], key=lambda x: x.map(MODEL_ORDER)
    )

    (baseline0, _) = get_baseline(probe_results_list)
    if do_print:
        print(baseline0)

    if codec == "jpeg_grace":
        qcodecs = "['grace', 'jpeg']"
    elif codec == "tcm":
        qcodecs = "['tcm']"

    df_jpeg_grace = df_full_jpeg[
        [
            "model_id",
            "nchunks",
            "cr",
            "csnr_db",
            "score0_lvc",
            "score0_lvc_g",
            "codec",
            "param",
            "nbits_per_sym",
            "eq_cr",
        ]
    ].query(f"csnr_db != 'inf' and codec in {qcodecs} and csnr_db >= 0")

    df_jpeg_grace["modulation"] = [
        f"{2**nb}-QAM" for nb in df_jpeg_grace["nbits_per_sym"]
    ]
    df_jpeg_grace = df_jpeg_grace.round({"eq_cr": 3})

    # df_grace = (
    #     df_jpeg_grace.query("codec == 'grace'")
    #     .copy()
    #     .drop("score0_lvc", axis=1)
    #     .rename(columns={"score0_lvc_g": "accuracy"})
    # )
    # df_grace["scheme"] = "GRACE"

    if codec == "jpeg_grace":
        qcodec = "jpeg"
    elif codec == "tcm":
        qcodec = "tcm"

    df_jpeg = (
        df_jpeg_grace.query(f"codec == '{qcodec}'")
        .copy()
        .drop("score0_lvc_g", axis=1)
        .rename(columns={"score0_lvc": "accuracy", "csnr_db": "Eb/N0 (dB)"})
    )

    if codec == "jpeg_grace":
        df_jpeg["scheme"] = "JPEG"
        param = "Q"
        param_name = "Q"
    elif codec == "tcm":
        df_jpeg["scheme"] = "LIC-TCM"
        param = "lambda"
        param_name = "λ"

    df_jpeg = df_jpeg.rename(columns={"param": param})

    if codec == "tcm":
        df_jpeg[param] = df_jpeg[param].map(lambda x: Path(Path(x).stem).stem)

    df_jpeg[param] = df_jpeg[param].map(lambda par: f"{param_name}={par}")

    group_cols = ["model_id", param, "scheme", "modulation", "eq_cr"]
    print("Mean eq_cr over all Eb/N0:")
    print(df_jpeg[group_cols].groupby(group_cols[:-1]).mean().to_string())

    # Collect CV-Cast
    _, df_sionna = collect_df("acc_vs_csnr_sionna_extra", do_reload=do_reload)

    df_sionna, duplicates = _remove_duplicates(df_sionna)
    if do_print:
        print("Duplicates:", len(duplicates))

    df_sionna = df_sionna.sort_values(by=DEFAULT_ID_COLS[2:])
    df_sionna = df_sionna.sort_values(
        by=["probe_model_id", "model_id"], key=lambda x: x.map(MODEL_ORDER)
    )

    estimator = "zf"
    mode = 444
    block_dct = False
    nchunks = 256

    df = df_sionna[(df_sionna["csnr_db"] != "inf")]
    df = _filter_df(df, estimator, mode, block_dct, nchunks)[
        ["model_id", "nchunks", "cr", "csnr_db", "score0_lvc", "score0_lvc_g"]
    ]
    df_g = (
        df.copy()
        .drop("score0_lvc", axis=1)
        .rename(columns={"score0_lvc_g": "accuracy"})
    )
    df_g["scheme"] = "CV-Cast"

    variants = [
        ("fastseg", ["fastseg_small", "fastseg_large"]),
        ("yolov8", ["yolov8_n", "yolov8_s", "yolov8_l"]),
    ]


    for variant, model_ids in variants:
        y8_rmargin = 0
        if codec == "jpeg_grace":
            cr = 1.0
            label = f"CV-Cast CR={cr:.1f}"
            if variant == "fastseg":
                w2 = w
                rmargin = w / 3 + y8_rmargin / 2
                # rmargin = y8_rmargin
                spacing = 0.15
            elif variant == "yolov8":
                w2 = w
                rmargin = y8_rmargin
                spacing = 0.1
        elif codec == "tcm":
            if variant == "fastseg":
                cr = 14 / 256
                label = f"CV-Cast CR={cr:.3f}"
                w2 = w
                rmargin = w / 3 + y8_rmargin / 2
                # rmargin = y8_rmargin
                spacing = 0.15
            elif variant == "yolov8":
                cr = 0.25
                label = f"CV-Cast CR={cr:.2f}"
                w2 = w
                rmargin = y8_rmargin
                spacing = 0.1

        dash = "6px"

        fig = px.line(
            df_jpeg.query("model_id in @model_ids"),
            x="Eb/N0 (dB)",
            y="accuracy",
            # color="param",
            # color="scheme",
            facet_col="model_id",
            facet_col_spacing=spacing,
            # facet_row="nchunks",
            symbol=param,
            labels=param,
            line_dash="modulation",
            line_dash_sequence=["solid", dash, "dot"],
            # symbol_sequence=["cross-thin-open", "x-thin-open", "y-down-open"],
            symbol_sequence=["circle", "circle", "circle"],
            markers=True,
            range_y=[0.0, None],
        )

        showlegend = True

        for row_idx, row_figs in enumerate(fig._grid_ref):
            for col_idx, col_fig in enumerate(row_figs):
                model_id = model_ids[col_idx]
                df_sub = df_g.query(
                    f"model_id == @model_id and cr == {str(cr)} and csnr_db >= 0 and csnr_db <= 10"
                )
                df_sub = df_sub.rename(columns={"csnr_db": "Eb/N0 (dB)"})
                fig.add_trace(
                    go.Scatter(
                        name=label,
                        x=df_sub["Eb/N0 (dB)"],
                        y=df_sub["accuracy"],
                        marker_color="red",
                        showlegend=showlegend,
                    ),
                    row=row_idx + 1,
                    col=col_idx + 1,
                )
                showlegend = False

        _add_baseline(fig, baseline0, model_ids=model_ids)

        fig.update_xaxes(
            zerolinecolor="grey",
            zeroline=True,
            gridcolor="grey",
            griddash="dot",
            tickmode="array",
            tickvals=[0, 2.5, 5, 7.5, 10],
        )
        fig.update_yaxes(
            tickformat=".0%",
            zerolinecolor="grey",
            zeroline=True,
            matches=None,
            showticklabels=True,
            gridcolor="grey",
            griddash="dot",
            minor=dict(showgrid=True),
        )

        fig.update_layout(
            plot_bgcolor="white",
            width=w2,
            height=h,
            font=dict(size=12),
            legend=dict(
                # title=f"modulation, {param_name}",
                title="",
                orientation="h",
                x=0.0,
                y=-0.3,
                xanchor="left",
                yanchor="top",
            ),
            margin=dict(l=20, r=rmargin, t=20, b=0),
            yaxis1=dict(range=[0.0, baseline0["fastseg_small"] * 1.1]),
            yaxis2=dict(range=[0.0, baseline0["fastseg_large"] * 1.1]),
        )

        # fig.update_traces(
        #     marker_size=8,
        #     marker_line_width=1.5,
        # )

        if variant == "fastseg":
            fig.update_layout(
                yaxis1=dict(range=[0.0, baseline0["fastseg_small"] * 1.1]),
                yaxis2=dict(range=[0.0, baseline0["fastseg_large"] * 1.1]),
            )
        elif variant == "yolov8":
            fig.update_layout(
                yaxis1=dict(range=[0.0, baseline0["yolov8_n"] * 1.1]),
                yaxis2=dict(range=[0.0, baseline0["yolov8_s"] * 1.1]),
                yaxis3=dict(range=[0.0, baseline0["yolov8_l"] * 1.1]),
            )

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        if do_save:
            fname = f"{outdir}/score_{codec}_{variant}_sionna"
            fig.write_image(f"{fname}.png", scale=scale)
            fig.write_html(f"{fname}.html")

        if do_show:
            fig.show()


def tcm_cr(do_reload: bool = False):
    print("TCM compression ratios")

    # Collect JPEG
    _, df_full_tcm = collect_df("tcm_sionna_8imgs_cr", do_reload=do_reload)
    print(df_full_tcm.columns)

    df_tcm = df_full_tcm[
        [
            "model_id",
            "codec",
            "csnr_db",
            "nbits_per_sym",
            "coderate",
            "param",
            "eq_cr",
            "score0_lvc",
        ]
    ].sort_values(
        ["model_id", "codec", "csnr_db", "nbits_per_sym", "param", "coderate"]
    )

    print(df_tcm.to_string())


def bd_default2(
    outdir: Path,
    w: int,
    h: int,
    scale: float = DEFAULT_SCALE,
    do_print: bool = True,
    do_show: bool = True,
    do_save: bool = True,
    do_reload: bool = False,
):
    result_order = {
        "lvc": 0,
        "lvc_g_256": 1,
        "lvc_g_ff": 1,
        "lvc_g_zf": 1,
        "lvc_g": 2,
    }

    if do_print:
        print("BD metrics default 2")

    _, df_full = collect_df("acc_vs_csnr", do_reload=do_reload)

    df_full, duplicates = _remove_duplicates(df_full)
    if do_print:
        print("Duplicates:", len(duplicates))

    df_full = df_full.sort_values(
        by=["probe_model_id", "model_id"], key=lambda x: x.map(MODEL_ORDER)
    )
    df_full = df_full.sort_values(by=DEFAULT_ID_COLS[2:])

    estimator = "zf"
    mode = 444
    block_dct = False
    nchunks = 256

    # msg = f"{estimator}, {mode}, {'bb' if block_dct else 'ff'}"
    df = df_full[df_full["csnr_db"] != "inf"]
    df = _filter_df(df, estimator, mode, block_dct, nchunks)[
        ["model_id", "nchunks", "cr", "csnr_db", "score0_lvc", "score0_lvc_g"]
    ]
    df_g = (
        df.copy()
        .drop("score0_lvc", axis=1)
        .rename(columns={"score0_lvc_g": "accuracy"})
    )
    df_g["scheme"] = "CV-Cast"
    df = df.drop("score0_lvc_g", axis=1).rename(columns={"score0_lvc": "accuracy"})
    df["scheme"] = "LCT"

    print(df)

    df_pareto = _pareto(df, low_cols=["csnr_db", "accuracy"])
    df_pareto_g = _pareto(df_g, low_cols=["csnr_db", "accuracy"])

    print(df_pareto)

    fig = px.line(
        df_pareto,
        x="csnr_db",
        y="accuracy",
        # color="CR",
        facet_col="model_id",
        facet_col_spacing=0.025,
        # facet_row="nchunks",
        symbol="scheme",
        labels="scheme",
        line_dash="scheme",
        line_dash_sequence=[DASH, "solid"],
        symbol_sequence=["cross-thin-open", "x-thin-open"],
        markers=True,
        range_y=[0.0, None],
    )

    if do_show:
        fig.show()


def bd_default(
    outdir: Path,
    w: int,
    h: int,
    scale: float = DEFAULT_SCALE,
    do_print: bool = True,
    do_show: bool = True,
    do_save: bool = True,
    do_reload: bool = False,
):
    result_order = {
        "lvc": 0,
        "lvc_g_256": 1,
        "lvc_g_ff": 1,
        "lvc_g_zf": 1,
        "lvc_g": 2,
    }

    if do_print:
        print("BD metrics default")

    _, df_full = collect_df("acc_vs_csnr", do_reload=do_reload)

    df_full, duplicates = _remove_duplicates(df_full)
    if do_print:
        print("Duplicates:", len(duplicates))

    df_full = df_full.sort_values(
        by=["probe_model_id", "model_id"], key=lambda x: x.map(MODEL_ORDER)
    )
    df_full = df_full.sort_values(by=DEFAULT_ID_COLS[2:])

    df_bd_default = _get_bd_default(df_full)

    df = df_bd_default[
        (df_bd_default["csnr_db"] == 10)
        & (df_bd_default["nchunks"] == 256)
        & (df_bd_default["result"] != "total")
    ]
    df = df.replace(True, "block-based")
    df = df.replace(False, "full-frame")
    df = df.replace("llse", "LLSE")
    df = df.replace("zf", "ZFE")
    df = df.rename(columns={"block_dct": "DCT"})
    df = df.rename(columns={"bdrate": "BD-Rate"})
    df = df.rename(columns={"bdacc": "BD-Accuracy"})

    df1 = df[
        (df["DCT"] == "full-frame")
        & (df["estimator"] == "LLSE")
        & (df["result"] != "lvc_g_256")
        & (df["result"] != "lvc_g_ff")
    ]
    df1 = df1.sort_values(by="estimator", ascending=False)
    df1 = df1.sort_values(by=["result"], key=lambda x: x.map(result_order))
    df1 = df1.sort_values(by=["model_id"], key=lambda x: x.map(MODEL_ORDER))

    # df1 = df1.rename(columns={"result": "reference vs. target"})
    # df1 = df1.replace("lvc", "LCT (ZFE) vs. LCT (LLSE)")
    # df1 = df1.replace("lvc_g", "LCT (LLSE) vs. CV-Cast (LLSE)")
    # df1 = df1.replace("total", "LCT (ZFE) vs. CV-Cast (LLSE)")
    # df1 = df1.replace("lvc_g_ff", "CV-Cast (ZFE) vs. CV-Cast (bb)")
    # df1 = df1.replace("lvc_g_zf", "CV-Cast (ZFE) vs. CV-Cast (LLSE)")

    df1 = df1.rename(columns={"result": "target vs. reference"})
    df1 = df1.replace("lvc", "LCT (LLSE) vs. LCT (ZFE)")
    df1 = df1.replace("lvc_g", "CV-Cast (LLSE) vs. LCT (LLSE)")
    df1 = df1.replace("total", "CV-Cast (LLSE) vs. LCT (ZFE)")
    df1 = df1.replace("lvc_g_ff", "CV-Cast (bb) vs. CV-Cast (ZFE)")
    df1 = df1.replace("lvc_g_zf", "CV-Cast (LLSE) vs. CV-Cast (ZFE)")

    fig = px.bar(
        df1,
        y="model_id",
        # y="result",
        x="BD-Rate",
        color="target vs. reference",
        # color="model_id",
        # color_discrete_map={"ZFE": "#ef553b", "LLSE": "#636efa"},
        pattern_shape="target vs. reference",
        # pattern_shape="model_id",
        # pattern_shape_map={"LLSE": "/", "ZFE": ""},
        # pattern_shape_sequence=["/", "\\"],
        pattern_shape_sequence=["/", "\\"],
        barmode="group",
        orientation="h",
    )

    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(
        title=None,
        zerolinecolor="grey",
        zeroline=True,
        # autorange="reversed",
        gridcolor="grey",
        griddash="dot",
        minor=dict(showgrid=True),
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        width=w,
        height=h * 0.8,
        legend=dict(title="", orientation="h", x=0.0, y=-0.2),
        margin=dict(l=20, r=20, t=20, b=0),
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    if do_save:
        fname = f"{outdir}/bdrate_estimators_2"
        fig.write_image(f"{fname}.png", scale=scale)
        fig.write_html(f"{fname}.html")

    if do_show:
        fig.show()

    fig = px.bar(
        df1,
        x="BD-Accuracy",
        y="model_id",
        color="target vs. reference",
        # color_discrete_map={"ZFE": "#ef553b", "LLSE": "#636efa"},
        pattern_shape="target vs. reference",
        # pattern_shape_map={"LLSE": "/", "ZFE": ""},
        pattern_shape_sequence=["/", "\\"],
        barmode="group",
        orientation="h",
    )

    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(
        title=None,
        zerolinecolor="grey",
        zeroline=True,
        gridcolor="grey",
        griddash="dot",
        minor=dict(showgrid=True),
    )
    fig.update_layout(
        showlegend=True,
        plot_bgcolor="white",
        width=w,
        height=h,
        legend=dict(title="", orientation="h", x=0.0, y=-0.1),
        margin=dict(l=20, r=20, t=20, b=0),
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    if do_save:
        fname = f"{outdir}/bdacc_estimators_2"
        fig.write_image(f"{fname}.png", scale=scale)
        fig.write_html(f"{fname}.html")

    if do_show:
        fig.show()

    df2 = df[
        (df["estimator"] == "ZFE")
        & (df["DCT"] == "block-based")
        & (df["result"] != "lvc_g_256")
        & (df["result"] != "lvc_g_zf")
    ]
    df2 = df2.sort_values(by="DCT", ascending=False)
    df2 = df2.sort_values(by=["result"], key=lambda x: x.map(result_order))
    df2 = df2.sort_values(by=["model_id"], key=lambda x: x.map(MODEL_ORDER))

    # df2.loc[:, "lvc"] = -df2["lvc"]
    # df2.loc[:, "lvc_g_ff"] = -df2["lvc_g_ff"]

    new_rows = []

    for _, row in df2.iterrows():
        row = row.to_dict()
        if row["result"] in ["lvc", "lvc_g_ff"]:
            row["BD-Rate"] = -row["BD-Rate"]
            row["BD-Accuracy"] = -row["BD-Accuracy"]
        new_rows.append(row)

    df2 = pd.DataFrame(new_rows)

    # df2 = df2.rename(columns={"result": "reference vs. target"})
    df2 = df2.replace("lvc", "LCT (ff) vs. LCT (bb)")  # <-- invert
    # df2 = df2.replace("lvc_g", "LCT (bb) vs. CV-Cast (bb)") # <-- keep
    df2 = df2.replace("total", "LCT (ff) vs. CV-Cast (bb)")
    df2 = df2.replace("lvc_g_ff", "CV-Cast (ff) vs. CV-Cast (bb)") # <-- invert
    df2 = df2.replace("lvc_g_zf", "CV-Cast (ff) vs. CV-Cast (LLSE)")

    df2 = df2.rename(columns={"result": "target vs. reference"})
    # df2 = df2.replace("lvc", "LCT (bb) vs. LCT (ff)")
    df2 = df2.replace("lvc_g", "CV-Cast (bb) vs. LCT (bb)")
    # df2 = df2.replace("total", "CV-Cast (bb) vs. LCT (ff)")
    # df2 = df2.replace("lvc_g_ff", "CV-Cast (bb) vs. CV-Cast (ff)")
    # df2 = df2.replace("lvc_g_zf", "CV-Cast (LLSE) vs. CV-Cast (ff)")

    fig = px.bar(
        df2,
        y="model_id",
        x="BD-Rate",
        color="target vs. reference",
        # color_discrete_map={"full-frame": "#ef553b", "block-based": "#636efa"},
        pattern_shape="target vs. reference",
        # pattern_shape_map={"full-frame": "", "block-based": "/"},
        pattern_shape_sequence=["/", "\\"],
        barmode="group",
        orientation="h",
    )

    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(
        title=None,
        zerolinecolor="grey",
        zeroline=True,
        # autorange="reversed",
        gridcolor="grey",
        griddash="dot",
        minor=dict(showgrid=True),
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        width=w,
        height=h * 0.8,
        legend=dict(title="", orientation="h", x=0.0, y=-0.1),
        margin=dict(l=20, r=20, t=20, b=0),
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    if do_save:
        fname = f"{outdir}/bdrate_dcts_2"
        fig.write_image(f"{fname}.png", scale=scale)
        fig.write_html(f"{fname}.html")

    if do_show:
        fig.show()

    fig = px.bar(
        df2,
        x="BD-Accuracy",
        y="model_id",
        color="target vs. reference",
        # color_discrete_map={"full-frame": "#ef553b", "block-based": "#636efa"},
        pattern_shape="target vs. reference",
        # pattern_shape_map={"full-frame": "", "block-based": "/"},
        pattern_shape_sequence=["/", "\\"],
        barmode="group",
        orientation="h",
    )

    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(
        title=None,
        zerolinecolor="grey",
        zeroline=True,
        gridcolor="grey",
        griddash="dot",
        minor=dict(showgrid=True),
    )
    fig.update_layout(
        showlegend=True,
        plot_bgcolor="white",
        width=w,
        height=h,
        legend=dict(title="", orientation="h", x=0.0, y=-0.1),
        margin=dict(l=20, r=20, t=20, b=0, pad=5),
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    if do_save:
        fname = f"{outdir}/bdacc_dcts_2"
        fig.write_image(f"{fname}.png", scale=scale)
        fig.write_html(f"{fname}.html")

    if do_show:
        fig.show()


def bd_table_64_vs_1024(
    outdir: Path,
    do_reload: bool = False,
):
    print("BD metrics chunks comparison")

    _, df_full = collect_df("acc_vs_csnr", do_reload=do_reload)
    df_full, duplicates = _remove_duplicates(df_full)
    print("Duplicates full:", len(duplicates))

    _, df_full_precise = collect_df("acc_vs_csnr_precise", do_reload=do_reload)
    df_full_precise, duplicates = _remove_duplicates(df_full_precise)
    print("Duplicates precise:", len(duplicates))

    df_full = df_full.sort_values(
        by=["probe_model_id", "model_id"], key=lambda x: x.map(MODEL_ORDER)
    )
    df_full = df_full.sort_values(by=DEFAULT_ID_COLS[2:])

    df_full_precise = df_full_precise.sort_values(
        by=["probe_model_id", "model_id"], key=lambda x: x.map(MODEL_ORDER)
    )
    df_full_precise = df_full_precise.sort_values(by=DEFAULT_ID_COLS[2:])

    df_bd = _get_bd(df_full)
    df_bd_default = _get_bd_default(df_full)
    df_bd_precise = _get_bd(df_full_precise)

    df = (
        df_bd[
            (df_bd["csnr_db"] == 10)
            & (df_bd["estimator"] == "zf")
            & (df_bd["block_dct"] == False)
        ]
        .drop(["estimator", "mode", "csnr_db", "block_dct"], axis=1)
        .round(2)
    )
    print(df)

    print("BD metrics default chunks comparison")

    df = (
        df_bd_default[
            (df_bd_default["csnr_db"] == 10)
            & (df_bd_default["nchunks"] != 256)
            & (df_bd_default["estimator"] == "zf")
            & (df_bd_default["block_dct"] == False)
            & (df_bd_default["result"] != "lvc_g_ff")
            & (df_bd_default["result"] != "lvc_g_zf")
            & (df_bd_default["result"] != "total")
        ]
        .drop(["estimator", "mode", "csnr_db", "block_dct"], axis=1)
        .round(2)
    )
    print(df)

    print("BD metrics chunks comparison (precise)")

    df = (
        df_bd_precise[
            (df_bd_precise["csnr_db"] == 10)
            & (df_bd_precise["estimator"] == "zf")
            & (df_bd_precise["block_dct"] == False)
        ]
        .drop(["estimator", "mode", "csnr_db", "block_dct"], axis=1)
        .round({"bdrate": 2, "bdacc": 2})
    )
    print(df)


def bd_table_sionna_metrics(
    outdir: Path,
    do_reload: bool = False,
):
    print("Sionna BD metrics")

    probe_results_list, df_sionna = collect_df(
        "acc_vs_csnr_sionna", do_reload=do_reload
    )
    df_sionna, duplicates = _remove_duplicates(df_sionna)
    print("Duplicates:", len(duplicates))

    df_sionna = df_sionna.sort_values(
        by=["probe_model_id", "model_id"], key=lambda x: x.map(MODEL_ORDER)
    )
    df_sionna = df_sionna.sort_values(by=DEFAULT_ID_COLS[2:])

    df_bd_sionna = _get_bd(df_sionna)

    df = (
        df_bd_sionna[
            # (df_bd_sionna["csnr_db"] == 10)
            (df_bd_sionna["estimator"] == "zf")
            & (df_bd_sionna["block_dct"] == False)
            & (df_bd_sionna["nchunks"] == 256)
        ]
        .drop(["estimator", "mode", "block_dct"], axis=1)
        .round(2)
    )
    print(df)


def maps_fss(
    outdir: Path,
    w: int,
    h: int,
    scale: float = DEFAULT_SCALE,
    do_print: bool = True,
    do_show: bool = True,
    do_save: bool = True,
    do_reload: bool = False,
):
    print("FastSeg small")
    probe_data = torch.load(f"{PROBE_DIR_OLD}/probe_results.pt")["fastseg_small"]
    print(list(probe_data.keys()))

    mode = 444
    grad_key = "grads_norm_420" if mode == 420 else "grads_norm"

    dct_var_y_64 = probe_data["dct_var"][64][0].unsqueeze(dim=0).numpy()
    dct_var_y_256 = probe_data["dct_var"][256][0].unsqueeze(dim=0).numpy()
    dct_var_y_1024 = probe_data["dct_var"][1024][0].unsqueeze(dim=0).numpy()

    gnorm_y_64 = probe_data[grad_key][64][0].unsqueeze(dim=0).numpy()
    gnorm_y_256 = probe_data[grad_key][256][0].unsqueeze(dim=0).numpy()
    gnorm_y_1024 = probe_data[grad_key][1024][0].unsqueeze(dim=0).numpy()

    gnorm_sq_y_64 = probe_data[grad_key][64][0].square().unsqueeze(dim=0).numpy()
    gnorm_sq_y_256 = probe_data[grad_key][256][0].square().unsqueeze(dim=0).numpy()
    gnorm_sq_y_1024 = probe_data[grad_key][1024][0].square().unsqueeze(dim=0).numpy()

    prod_y_64 = gnorm_sq_y_64 * dct_var_y_64
    prod_y_256 = gnorm_sq_y_256 * dct_var_y_256
    prod_y_1024 = gnorm_sq_y_1024 * dct_var_y_1024

    img_64 = np.concatenate([dct_var_y_64, gnorm_y_64, prod_y_64])
    img_256 = np.concatenate([dct_var_y_256, gnorm_y_256, prod_y_256])
    img_1024 = np.concatenate([dct_var_y_1024, gnorm_y_1024, prod_y_1024])

    _heatmaps(
        img_64,
        w,
        h,
        scale,
        1,
        3,
        None,
        show=True,
        save=f"{outdir}/maps_y_fastseg_small_64.png",
    )
    _heatmaps(
        img_256,
        w,
        h,
        scale,
        1,
        3,
        None,
        show=True,
        save=f"{outdir}/maps_y_fastseg_small_256.png",
    )
    # _heatmaps(img_1024, w, h, scale, 1, 3, None, show=True, save=None)


def maps_y8s(
    outdir: Path,
    w: int,
    h: int,
    scale: float = DEFAULT_SCALE,
    do_print: bool = True,
    do_show: bool = True,
    do_save: bool = True,
    do_reload: bool = False,
):
    print("YOLOv8s")
    probe_data = torch.load(f"{PROBE_DIR_OLD}/probe_results.pt")["yolov8_s"]

    mode = 444
    grad_key = "grads_norm_420" if mode == 420 else "grads_norm"

    dct_var_y_64 = probe_data["dct_var"][64][0].unsqueeze(dim=0).numpy()
    dct_var_y_256 = probe_data["dct_var"][256][0].unsqueeze(dim=0).numpy()
    dct_var_y_1024 = probe_data["dct_var"][1024][0].unsqueeze(dim=0).numpy()

    gnorm_y_64 = probe_data[grad_key][64][0].unsqueeze(dim=0).numpy()
    gnorm_y_256 = probe_data[grad_key][256][0].unsqueeze(dim=0).numpy()
    gnorm_y_1024 = probe_data[grad_key][1024][0].unsqueeze(dim=0).numpy()

    gnorm_sq_y_64 = probe_data[grad_key][64][0].square().unsqueeze(dim=0).numpy()
    gnorm_sq_y_256 = probe_data[grad_key][256][0].square().unsqueeze(dim=0).numpy()
    gnorm_sq_y_1024 = probe_data[grad_key][1024][0].square().unsqueeze(dim=0).numpy()

    prod_y_64 = gnorm_sq_y_64 * dct_var_y_64
    prod_y_256 = gnorm_sq_y_256 * dct_var_y_256
    prod_y_1024 = gnorm_sq_y_1024 * dct_var_y_1024

    img_64 = np.concatenate([dct_var_y_64, gnorm_y_64, prod_y_64])
    img_256 = np.concatenate([dct_var_y_256, gnorm_y_256, prod_y_256])
    img_1024 = np.concatenate([dct_var_y_1024, gnorm_y_1024, prod_y_1024])

    # _heatmaps(img_64, w, h, scale, 1, 3, None, show=True, save=None)
    _heatmaps(
        img_256, w, h, scale, 1, 3, None, show=True, save=f"{outdir}/maps_y_yolov8s.png"
    )
    # _heatmaps(img_1024, w, h, scale, 1, 3, None, show=True, save=None)


def grace_quant_table(
    outdir: Path,
    w: int,
    h: int,
    scale: float = DEFAULT_SCALE,
    do_show: bool = True,
):
    q_search_dir = Path("experiments_tupu/runs/run27_keep")
    fname = q_search_dir / "q_search_fastseg_small_8imgs.pt"
    df = pd.DataFrame(torch.load(fname))

    res = df.query(
        " and ".join(
            [
                "codec == 'grace'",
                "model == 'fastseg_small'",
                "mode == 444",
                "dist == 'dist_abs'",
                "norm == 'abs'",
                "sub == 'submean'",
                "dctsz == 'ff'",
                "sc == 1",
                "nbits_per_sym == 4",
                "target_cr == 0.5",
            ]
        )
    )

    B = res.iloc[0]["param"]

    probe_dir = Path("experiments_tupu/runs/run24_keep")
    probe_fname = (
        probe_dir
        / "probe_result_full_fastseg_small_444_dist_abs_submean_ff_sc1_normabs.pt"
    )
    probe_data = torch.load(probe_fname)
    grad_norm = probe_data["grads_norm"][64]

    qt, _ = get_quant_table_approx(grad_norm, B)

    label1 = f"FSS Norm gradient"
    label2 = f"FSS GRACE quant tables approx (B = {B})"

    _heatmaps(
        grad_norm.detach().cpu().numpy(),
        w,
        h,
        scale,
        1,
        3,
        label=label1,
        show=do_show,
        save=None,
    )
    _heatmaps(
        qt.detach().cpu().numpy(),
        w,
        h,
        scale,
        1,
        3,
        label=label2,
        show=do_show,
        save=None,
    )

    probe_fname = (
        probe_dir
        / "probe_result_full_fastseg_small_444_dist_abs_submean_ff_sc255_normabs.pt"
    )
    probe_data = torch.load(probe_fname)
    grad_norm = probe_data["grads_norm"][64]

    dct = probe_data["dct_yuv"]

    norm_abs = lambda tensor: tensor.abs().mean()
    dct_block_norm = block_process(
        dct,
        (int(dct.shape[1] / 8), int(dct.shape[2] / 8)),
        norm_abs,
    )

    B = 3e-5

    q = torch.zeros_like(grad_norm)
    d = torch.zeros_like(grad_norm)
    bounded = torch.zeros_like(grad_norm)
    max_loss_increase = torch.zeros_like(grad_norm)

    for i, (g_norm, dct_norm) in enumerate(zip(grad_norm, dct_block_norm)):
        q[i], d[i], bounded[i], max_loss_increase[i] = get_quant_table(
            g_norm, dct_norm, B, do_print=False
        )

    _heatmaps(
        q.detach().cpu().numpy(),
        w,
        h,
        scale,
        1,
        3,
        label=None,
        show=do_show,
        save=Path("experiments/plots/grace_q_table.png"),
    )
