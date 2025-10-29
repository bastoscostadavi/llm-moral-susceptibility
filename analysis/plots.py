# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==5.5.0",
#     "anthropic==0.71.0",
#     "matplotlib==3.10.7",
#     "openai==2.6.1",
#     "polars==1.34.0",
#     "vl-convert-python==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.16.4"
app = marimo.App(width="medium")


app._unparsable_cell(
    r"""
    pwdimport altair as alt
    from matplotlib import pyplot as plt
    import polars as pl
    from pathlib import Path
    import marimo as mo
    import vl_convert
    """,
    name="_"
)


@app.cell
def _(Path):
    def _get_results_directory():
        cwd = Path.cwd()
        if (r := cwd / "results") in cwd.iterdir() and r.is_dir():
            return r
        elif cwd.name == "results":
            return cwd
        elif cwd.name == "analysis":
            return cwd.parent / "results"
        else:
            raise Exception(
                "Can't find the results. Run this script either from the 'root', 'analysis' or 'results' directories."
            )


    RESULTS_DIR = _get_results_directory()
    return (RESULTS_DIR,)


@app.cell
def _(RESULTS_DIR, pl):
    table = dict(
        map(
            lambda p: (p.name.split(".")[0], pl.read_csv(p)),
            RESULTS_DIR.glob("moral*csv"),
        )
    )
    return (table,)


@app.cell
def _(table):
    table
    return


@app.cell
def _(pl):
    def create_val_err_label(val, err, *, round=2, sep=" ± "):
        return (
            pl.col(val).round(round).cast(pl.String)
            + pl.lit(sep)
            + pl.col(err).round(round).cast(pl.String)
        )
    return (create_val_err_label,)


@app.cell
def _(RESULTS_DIR, alt, create_val_err_label, table):
    mm_data = alt.Chart(
        table["moral_metrics"].with_columns(
            s_and_e_label=create_val_err_label("susceptibility", "s_uncertainty"),
            r_and_e_label=create_val_err_label("robustness", "r_uncertainty"),
        )
    )

    y_model = alt.Y(
        "model:N",
        title="Model",
        sort=alt.SortField("susceptibility", order="descending"),
    )
    x_susc = alt.X("susceptibility:Q", title="Susceptibility")

    s_point_plot = (
        mm_data.mark_point(color="black", filled=True, size=30)
        .encode(y_model, x_susc, tooltip=["model", "susceptibility"])
        .properties(title="Moral Susceptibility by Model", width=250, height=300)
    )

    s_error_bars = mm_data.mark_errorbar().encode(
        y_model, x_susc, alt.XError("s_uncertainty")
    )

    s_label = s_point_plot.mark_text(
        align="center", baseline="bottom", size=8, dy=-5, dx=0
    ).encode(text="s_and_e_label")

    s_plot = (
        (s_point_plot + s_error_bars + s_label)
        .configure_axis(grid=False)
        .configure_view(
            strokeWidth=0  # Removes the chart border
        )
    )

    s_plot.save(RESULTS_DIR / "susceptibility_point_plot.pdf")
    s_plot.save(RESULTS_DIR / "susceptibility_point_plot.png")
    s_plot
    return (mm_data,)


@app.cell
def _(RESULTS_DIR, alt, mm_data):
    x_robus = alt.X("robustness:Q", title="Robustness")
    y_model2 = alt.Y(
        "model:N",
        title="Model",
        sort=alt.SortField("robustness", order="descending"),
    )

    r_point_plot = (
        mm_data.mark_point(color="black", filled=True, size=30)
        .encode(y_model2, x_robus, tooltip=["model", "robustness"])
        .properties(title="Moral Robustness by Model", width=250, height=300)
    )

    r_error_bars = mm_data.mark_errorbar().encode(
        y_model2, x_robus, alt.XError("r_uncertainty")
    )

    r_label = r_point_plot.mark_text(
        align="center", baseline="bottom", size=8, dy=-5, dx=20
    ).encode(text="r_and_e_label")

    r_plot = (
        (r_point_plot + r_error_bars + r_label)
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
    )

    r_plot.save(RESULTS_DIR / "robustness_point_plot.pdf")
    r_plot.save(RESULTS_DIR / "robustness_point_plot.png")
    r_plot
    return


@app.cell
def _(pl):
    def zscore(val):
        val_col = pl.col(val)
        return (val_col - val_col.mean()) / val_col.std()


    def zscore_error(val, err):
        return (pl.col(err) / pl.col(val).std()).abs()
    return zscore, zscore_error


@app.cell
def _(RESULTS_DIR, alt, create_val_err_label, table, zscore, zscore_error):
    mm_z_data = alt.Chart(
        table["moral_metrics"]
        .with_columns(
            s_zscore=zscore("susceptibility"),
            s_zscore_error=zscore_error("susceptibility", "s_uncertainty"),
            r_zscore=zscore("robustness"),
            r_zscore_error=zscore_error("robustness", "r_uncertainty"),
        )
        .with_columns(
            sz_label=create_val_err_label("s_zscore", "s_zscore_error"),
            rz_label=create_val_err_label("r_zscore", "r_zscore_error"),
        )
    )

    sz_point_plot = (
        mm_z_data.mark_point(color="black", filled=True)
        .encode(
            alt.X("s_zscore:Q", title="Susceptibility Z-Score"),
            alt.Y(
                "model:N",
                title="Model",
                sort=alt.SortField("s_zscore", order="descending"),
            ),
        )
        .properties(
            title="Moral Susceptibility Z-Score by Model", width=250, height=300
        )
    )

    sz_error_plot = mm_z_data.mark_errorbar().encode(
        alt.Y(
            "model:N",
            title="Model",
            sort=alt.SortField("s_zscore", order="descending"),
        ),
        alt.X("s_zscore:Q", title="Susceptibility Z-Score"),
        alt.XError("s_zscore_error:Q"),
    )

    sz_label = sz_point_plot.mark_text(
        align="center", baseline="bottom", size=8, dy=-5, dx=0
    ).encode(text="sz_label")

    sz_plot = (
        (sz_point_plot + sz_error_plot + sz_label)
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
    )
    sz_plot.save(RESULTS_DIR / "susceptibility_zscore_point_plot.pdf")
    sz_plot.save(RESULTS_DIR / "susceptibility_zscore_point_plot.png")

    sz_plot
    return (mm_z_data,)


@app.cell
def _(RESULTS_DIR, alt, mm_z_data):
    rz_point_plot = (
        mm_z_data.mark_point(color="black", filled=True)
        .encode(
            alt.X("r_zscore:Q", title="Robustness Z-Score"),
            alt.Y(
                "model:N",
                title="Model",
                sort=alt.SortField("r_zscore", order="descending"),
            ),
        )
        .properties(
            title="Moral Robustness Z-Score by Model", width=250, height=300
        )
    )

    rz_error_plot = mm_z_data.mark_errorbar().encode(
        alt.Y(
            "model:N",
            title="Model",
            sort=alt.SortField("r_zscore", order="descending"),
        ),
        alt.X("r_zscore:Q", title="Robustness Z-Score"),
        alt.XError("r_zscore_error:Q"),
    )

    rz_label = rz_point_plot.mark_text(
        align="center", baseline="bottom", size=8, dy=-5, dx=0
    ).encode(text="rz_label")

    rz_plot = (
        (rz_point_plot + rz_error_plot + rz_label)
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
    )
    rz_plot.save(RESULTS_DIR / "robustness_zscore_point_plot.pdf")
    rz_plot.save(RESULTS_DIR / "robustness_zscore_point_plot.png")

    rz_plot
    return


@app.cell
def _(alt, create_val_err_label, table, zscore, zscore_error):
    mmbf_data = alt.Chart(
        table["moral_metrics_by_foundation"]
        .with_columns(
             s_zscore=zscore("susceptibility").over("foundation"),
             s_zscore_error=zscore_error("susceptibility","s_uncertainty").over("foundation"),
             r_zscore=zscore("robustness").over("foundation"),
             r_zscore_error=zscore_error("robustness","r_uncertainty").over("foundation")
         )
         .with_columns(
             s_label=create_val_err_label("susceptibility", "s_uncertainty"),
             sz_label=create_val_err_label("s_zscore", "s_zscore_error"),
             r_label=create_val_err_label("robustness", "r_uncertainty"),
             rz_label=create_val_err_label("r_zscore", "r_zscore_error"),
         )
    )


    return


if __name__ == "__main__":
    app.run()
