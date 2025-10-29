# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==5.5.0",
#     "anthropic==0.71.0",
#     "matplotlib==3.10.7",
#     "openai==2.6.1",
#     "plotnine==0.15.0",
#     "polars==1.34.0",
#     "pyarrow==22.0.0",
#     "pygwalker==0.4.9.15",
#     "pyobsplot==0.5.4",
#     "seaborn==0.13.2",
#     "vl-convert-python==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.16.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    from matplotlib import pyplot as plt
    import polars as pl
    from pathlib import Path
    import marimo as mo
    import vl_convert
    from pyobsplot import Plot, d3, js
    import pyarrow
    import seaborn as sns
    import seaborn.objects as so
    import plotnine as pn
    return Path, alt, pl


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
def _(pl):
    def zscore(val):
        val_col = pl.col(val)
        return (val_col - val_col.mean()) / val_col.std()


    def zscore_error(val, err):
        return (pl.col(err) / pl.col(val).std()).abs()
    return zscore, zscore_error


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
def _(create_val_err_label, zscore, zscore_error):
    def extracolumns(df, group=None):
        if group:
            return df.with_columns(
                s_zscore=zscore("susceptibility").over(group),
                s_zscore_error=zscore_error(
                    "susceptibility", "susceptibility_uncertainty"
                ).over(group),
                r_zscore=zscore("robustness").over(group),
                r_zscore_error=zscore_error(
                    "robustness", "robustness_uncertainty"
                ).over(group),
            ).with_columns(
                s_label=create_val_err_label(
                    "susceptibility", "susceptibility_uncertainty"
                ),
                r_label=create_val_err_label(
                    "robustness", "robustness_uncertainty"
                ),
                sz_label=create_val_err_label("s_zscore", "s_zscore_error"),
                rz_label=create_val_err_label("r_zscore", "r_zscore_error"),
            )
        return df.with_columns(
            s_zscore=zscore("susceptibility"),
            s_zscore_error=zscore_error(
                "susceptibility", "susceptibility_uncertainty"
            ),
            r_zscore=zscore("robustness"),
            r_zscore_error=zscore_error("robustness", "robustness_uncertainty"),
        ).with_columns(
            s_label=create_val_err_label(
                "susceptibility", "susceptibility_uncertainty"
            ),
            r_label=create_val_err_label("robustness", "robustness_uncertainty"),
            sz_label=create_val_err_label("s_zscore", "s_zscore_error"),
            rz_label=create_val_err_label("r_zscore", "r_zscore_error"),
        )
    return (extracolumns,)


@app.cell
def _(extracolumns, pl):
    def load_results(results_dir):
        table = {}
        for f in results_dir.glob("moral*csv"):
            if "foundation" not in f.stem:
                table[f.stem] = extracolumns(
                    pl.read_csv(f)
                    .with_columns(foundation=pl.lit("All Foundations"))
                    .select(
                        "model",
                        "foundation",
                        "robustness",
                        "robustness_uncertainty",
                        "susceptibility",
                        "susceptibility_uncertainty",
                    )
                )
            else:
                table[f.stem] = extracolumns(pl.read_csv(f), group="foundation")
        return table
    return (load_results,)


@app.cell
def _(RESULTS_DIR, load_results):
    table = load_results(RESULTS_DIR)

    table
    return (table,)


@app.cell
def _(pl, table):
    df = pl.concat(table.values())
    df
    return (df,)


@app.cell
def _():
    MODEL_COLORS = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    FOUNDATION_COLORS = ["blue", "green", "yellow", "red", "lime"]

    FOUNDATIONS_ORDER = [
        "All Foundations",
        "Harm/Care",
        "Fairness/Reciprocity",
        "In-group/Loyalty",
        "Authority/Respect",
        "Purity/Sanctity",
    ]
    return (FOUNDATIONS_ORDER,)


@app.cell
def _(FOUNDATIONS_ORDER, RESULTS_DIR, alt, df, pl):
    _p_data = alt.Chart(
        df.with_columns(
            f_color=pl.when(pl.col("foundation") == "All Foundations")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        ),
        height=100,
        width=150,
    )

    _p_bars = (
        _p_data
        # .transform_filter(alt.datum.foundat)
        .mark_bar(size=15).encode(
            alt.X("susceptibility:Q",title="Susceptibility"),
            alt.Y("foundation:N", title=None, sort=FOUNDATIONS_ORDER),
            alt.Color("model:N", title=""),
            opacity=alt.when(alt.datum.foundation == "All Foundations")
            .then(alt.value(0.6))
            .otherwise(alt.value(0.3)),
        )
    )

    _p_error = _p_data.mark_errorbar().encode(
        alt.X("susceptibility:Q",title="Susceptibility"),
        alt.XError("susceptibility_uncertainty:Q"),
        alt.Y("foundation:N", title=None, sort=FOUNDATIONS_ORDER),
    )

    _p_label = _p_bars.mark_text(
        align="right",
        baseline="middle",
        dx=-8.5,
        dy=0,
        fontSize=8,
        opacity=1.0,
        color="#000000",
    ).encode(
        alt.Y("foundation:N"),
        alt.X("susceptibility:Q",title="Susceptibility"),
        alt.Text("s_label:N"),
        color=alt.value("black"),
        opacity=alt.value(1.0),
    )

    _p = (
        (_p_bars + _p_error + _p_label)
        .facet("model:N", columns=5, title="")
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
        # .properties(
        #     width=100,
        #     height=50,
        # )
    )

    _p.save(RESULTS_DIR / "susceptibility_bars.png")
    _p.save(RESULTS_DIR / "susceptibility_bars.pdf")

    _p
    return


@app.cell
def _(FOUNDATIONS_ORDER, RESULTS_DIR, alt, df, pl):
    _p_data = alt.Chart(
        df.with_columns(
            f_color=pl.when(pl.col("foundation") == "All Foundations")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        ),
        height=100,
        width=150,
    )

    _p_bars = (
        _p_data
        # .transform_filter(alt.datum.foundat)
        .mark_bar(size=15).encode(
            alt.X("s_zscore:Q",title="Susceptibility Z-Score"),
            alt.Y("foundation:N", title=None, sort=FOUNDATIONS_ORDER),
            alt.Color("model:N", title=""),
            opacity=alt.when(alt.datum.foundation == "All Foundations")
            .then(alt.value(0.6))
            .otherwise(alt.value(0.3)),
        )
    )

    _p_error = _p_data.mark_errorbar().encode(
        alt.X("s_zscore:Q",title="Susceptibility Z-Score"),
        alt.XError("s_zscore_error:Q"),
        alt.Y("foundation:N", title=None, sort=FOUNDATIONS_ORDER),
    )

    _p_label = _p_bars.mark_text(
        align="left",
        baseline="middle",
        dx=40,
        dy=0,
        fontSize=8,
        opacity=1.0,
        color="#000000",
    ).encode(
        alt.Y("foundation:N"),
        alt.X("s_zscore:Q",title="Susceptibility Z-Score"),
        alt.Text("sz_label:N"),
        color=alt.value("black"),
        opacity=alt.value(1.0),
    )

    _p = (
        (_p_bars + _p_error + _p_label)
        .facet("model:N", columns=5, title="")
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
        # .properties(
        #     width=100,
        #     height=50,
        # )
    )

    _p.save(RESULTS_DIR / "susceptibility_zscore_bars.png")
    _p.save(RESULTS_DIR / "susceptibility_zscore_bars.pdf")

    _p
    return


@app.cell
def _(FOUNDATIONS_ORDER, RESULTS_DIR, alt, df, pl):
    _p_data = alt.Chart(
        df.with_columns(
            f_color=pl.when(pl.col("foundation") == "All Foundations")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        ),
        height=100,
        width=150,
    )

    _p_bars = (
        _p_data
        # .transform_filter(alt.datum.foundat)
        .mark_bar(size=15).encode(
            alt.X("robustness:Q",title="Robustness"),
            alt.Y("foundation:N", title=None, sort=FOUNDATIONS_ORDER),
            alt.Color("model:N", title=""),
            opacity=alt.when(alt.datum.foundation == "All Foundations")
            .then(alt.value(0.6))
            .otherwise(alt.value(0.3)),
        )
    )

    _p_error = _p_data.mark_errorbar().encode(
        alt.X("robustness:Q",title="Robustness"),
        alt.XError("robustness_uncertainty:Q"),
        alt.Y("foundation:N", title=None, sort=FOUNDATIONS_ORDER),
    )

    _p_label = _p_bars.mark_text(
        align="left",
        baseline="middle",
        dx=8.5,
        dy=-5,
        fontSize=8,
        opacity=1.0,
        color="#000000",
    ).encode(
        alt.Y("foundation:N"),
        alt.X("robustness:Q",title="Robustness"),
        alt.Text("r_label:N"),
        color=alt.value("black"),
        opacity=alt.value(1.0),
    )

    _p = (
        (_p_bars + _p_error + _p_label)
        .facet("model:N", columns=5, title="")
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
        # .properties(
        #     width=100,
        #     height=50,
        # )
    )

    _p.save(RESULTS_DIR / "robustness_bars.png")
    _p.save(RESULTS_DIR / "robustness_bars.pdf")

    _p
    return


@app.cell
def _(FOUNDATIONS_ORDER, RESULTS_DIR, alt, df, pl):
    _p_data = alt.Chart(
        df.with_columns(
            f_color=pl.when(pl.col("foundation") == "All Foundations")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        ),
        height=100,
        width=150,
    )

    _p_bars = (
        _p_data
        # .transform_filter(alt.datum.foundat)
        .mark_bar(size=15).encode(
            alt.X("r_zscore:Q",title="Robustness Z-Score"),
            alt.Y("foundation:N", title=None, sort=FOUNDATIONS_ORDER),
            alt.Color("model:N", title=""),
            opacity=alt.when(alt.datum.foundation == "All Foundations")
            .then(alt.value(0.6))
            .otherwise(alt.value(0.3)),
        )
    )

    _p_error = _p_data.mark_errorbar().encode(
        alt.X("r_zscore:Q",title="Robustness Z-Score"),
        alt.XError("r_zscore_error:Q"),
        alt.Y("foundation:N", title=None, sort=FOUNDATIONS_ORDER),
    )

    _p_label = _p_bars.mark_text(
        align="left",
        baseline="middle",
        dx=45,
        dy=0,
        fontSize=8,
        opacity=1.0,
        color="#000000",
    ).encode(
        alt.Y("foundation:N"),
        alt.X("r_zscore:Q",title="Robustness Z-Score"),
        alt.Text("rz_label:N"),
        color=alt.value("black"),
        opacity=alt.value(1.0),
    )

    _p = (
        (_p_bars + _p_error + _p_label)
        .facet("model:N", columns=5, title="")
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
        # .properties(
        #     width=100,
        #     height=50,
        # )
    )

    _p.save(RESULTS_DIR / "robustness_zscore_bars.png")
    _p.save(RESULTS_DIR / "robustness_zscore_bars.pdf")

    _p
    return


app._unparsable_cell(
    r"""
    Create a scatter plot for @table[\"moral_metrics\"] with column model in the Y-axis, column \"susceptibility\" in the X-axis, column \"susceptibility_uncertainty\" as the error in the X-Axis, with column \"s_label\" as annotations, removing the grid the right and top spines, using the altair library
    """,
    name="_"
)


@app.cell
def _(alt):
    def model_scatter(
        df,
        metric,
        metric_uncertainty,
        metric_label,
        *,
        metric_title=None,
        dx=0,
        dy=-5,
        size=50,
        height=150,
        width=200,
    ):
        scatter_data = alt.Chart(df)

        y_axis = alt.Y(
            "model:N",
            title="Model",
            sort=alt.SortField(metric, order="descending"),
        )
        x_axis = alt.X(
            metric + ":Q",
            title=(metric_title if metric_title else metric.capitalize()),
        )

        points = scatter_data.mark_point(
            color="black", filled=True, size=size, opacity=1.0
        ).encode(y_axis, x_axis)

        error_bars = scatter_data.mark_errorbar().encode(
            y_axis, x_axis, alt.XError(metric_uncertainty + ":Q")
        )

        labels = scatter_data.mark_text(
            align="center", baseline="bottom", dx=dx, dy=dy, fontSize=7
        ).encode(y_axis, x_axis, text=metric_label + ":N")

        scatter_plot = (
            (points + error_bars + labels)
            .configure_axis(grid=False)
            .configure_view(strokeWidth=0)
            .properties(
                title="Susceptibility by Model with Uncertainty",
                width=width,
                height=height,
            )
        )

        return scatter_plot
    return (model_scatter,)


@app.cell
def _(model_scatter, table):
    model_scatter(
        table["moral_metrics"],
        "susceptibility",
        "susceptibility_uncertainty",
        "s_label",
        size=20,
    )
    return


@app.cell
def _(model_scatter, table):
    model_scatter(
        table["moral_metrics"],
        "robustness",
        "robustness_uncertainty",
        "r_label",
        dx=15,
        size=20,
    )
    return


@app.cell
def _(alt, table):
    _foundation_data = alt.Chart(table["moral_metrics_per_foundation"])

    _y_axis = alt.Y(
        "model:N",
        title="Model",
        sort=alt.SortField("susceptibility", order="descending"),
    )
    _x_axis = alt.X("susceptibility:Q", title="Susceptibility")

    _points = _foundation_data.mark_point(
        color="black", filled=True, size=50, opacity=1.0
    ).encode(
        _y_axis, _x_axis, alt.Color("foundation:N"), alt.Column("foundation:N")
    )

    _error_bars = _foundation_data.mark_errorbar().encode(
        _y_axis, _x_axis, alt.XError("susceptibility_uncertainty:Q")
    )

    _labels = _foundation_data.mark_text(
        align="center", baseline="bottom", dx=0, dy=-5, fontSize=8
    ).encode(_y_axis, _x_axis, text="s_label:N")

    _foundation_plot = (
        (_points + _error_bars + _labels)
        # .facet(facet="foundation:N", columns=3)
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
        .properties(
            title="Susceptibility by Model and Foundation",
            # width=200,
            # height=150,
        )
    )

    _foundation_plot
    return


@app.cell
def _():
    # mm_data = alt.Chart(table["moral_metrics"])

    # y_model = alt.Y(
    #     "model:N",
    #     title="Model",
    #     sort=alt.SortField("susceptibility", order="descending"),
    # )
    # x_susc = alt.X("susceptibility:Q", title="Susceptibility")

    # s_point_plot = (
    #     mm_data.mark_point(color="black", filled=True, size=36)
    #     .encode(y_model, x_susc)
    #     .properties(title="Moral Susceptibility by Model", width=250, height=200)
    # )

    # s_error_bars = mm_data.mark_errorbar().encode(
    #     y_model, x_susc, alt.XError("susceptibility_uncertainty")
    # )

    # s_label = s_point_plot.mark_text(
    #     align="center", baseline="bottom", size=8, dy=-5, dx=0
    # ).encode(text="s_label")

    # s_plot = (
    #     (s_point_plot + s_error_bars + s_label)
    #     .configure_axis(grid=False)
    #     .configure_view(
    #         strokeWidth=0  # Removes the chart border
    #     )
    # )

    # # s_plot.save(RESULTS_DIR / "susceptibility_point_plot.pdf")
    # # s_plot.save(RESULTS_DIR / "susceptibility_point_plot.png")
    # s_plot
    return


@app.cell
def _():
    # x_robus = alt.X("robustness:Q", title="Robustness")
    # y_model2 = alt.Y(
    #     "model:N",
    #     title="Model",
    #     sort=alt.SortField("robustness", order="descending"),
    # )

    # r_point_plot = (
    #     mm_data.mark_point(color="black", filled=True, size=30)
    #     .encode(y_model2, x_robus, tooltip=["model", "robustness"])
    #     .properties(title="Moral Robustness by Model", width=250, height=300)
    # )

    # r_error_bars = mm_data.mark_errorbar().encode(
    #     y_model2, x_robus, alt.XError("r_uncertainty")
    # )

    # r_label = r_point_plot.mark_text(
    #     align="center", baseline="bottom", size=8, dy=-5, dx=20
    # ).encode(text="r_and_e_label")

    # r_plot = (
    #     (r_point_plot + r_error_bars + r_label)
    #     .configure_axis(grid=False)
    #     .configure_view(strokeWidth=0)
    # )

    # r_plot.save(RESULTS_DIR / "robustness_point_plot.pdf")
    # r_plot.save(RESULTS_DIR / "robustness_point_plot.png")
    # r_plot
    return


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
    return mm_z_data, sz_point_plot


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
def _(alt, create_val_err_label, sz_point_plot, table, zscore, zscore_error):
    mmbf_data = alt.Chart(
        table["moral_metrics_by_foundation"]
        .with_columns(
            s_zscore=zscore("susceptibility").over("foundation"),
            s_zscore_error=zscore_error("susceptibility", "s_uncertainty").over(
                "foundation"
            ),
            r_zscore=zscore("robustness").over("foundation"),
            r_zscore_error=zscore_error("robustness", "r_uncertainty").over(
                "foundation"
            ),
        )
        .with_columns(
            s_label=create_val_err_label("susceptibility", "s_uncertainty"),
            sz_label=create_val_err_label("s_zscore", "s_zscore_error"),
            r_label=create_val_err_label("robustness", "r_uncertainty"),
            rz_label=create_val_err_label("r_zscore", "r_zscore_error"),
        )
    )

    sbf_point_plot = (
        mmbf_data.mark_point(color="black", filled=True)
        .encode(
            alt.X("susceptibility:Q", title="Susceptibility by Foundation"),
            alt.Y(
                "model:N",
                title="Model",
                sort=alt.SortField("s_zscore", order="descending"),
            ),
            alt.Color("foundation:N"),
        )
        .properties(
            title="Moral Susceptibility Z-Score by Model", width=250, height=300
        )
    )

    # sbf_error_plot = mmbf_data.mark_errorbar().encode(
    #     alt.Y(
    #         "model:N",
    #         title="Model",
    #         sort=alt.SortField("s_zscore", order="descending"),
    #     ),
    #     alt.X("s_zscore:Q", title="Susceptibility Z-Score"),
    #     alt.XError("s_zscore_error:Q"),
    # )

    # sz_label = sz_point_plot.mark_text(
    #     align="center", baseline="bottom", size=8, dy=-5, dx=0
    # ).encode(text="sz_label")

    sbf_plot = (
        (sz_point_plot)  # + sz_error_plot + sz_label)
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
    )
    # sz_plot.save(RESULTS_DIR / "susceptibility_zscore_point_plot.pdf")
    # sz_plot.save(RESULTS_DIR / "susceptibility_zscore_point_plot.png")

    sbf_plot
    return


@app.cell
def _(alt, create_val_err_label, table, zscore, zscore_error):
    (
        alt.Chart(
            table["moral_metrics_per_foundation"]
            .with_columns(
                s_zscore=zscore("susceptibility").over("foundation"),
                s_zscore_error=zscore_error(
                    "susceptibility", "susceptibility_uncertainty"
                ).over("foundation"),
                r_zscore=zscore("robustness").over("foundation"),
                r_zscore_error=zscore_error(
                    "robustness", "robustness_uncertainty"
                ).over("foundation"),
            )
            .with_columns(
                s_label=create_val_err_label(
                    "susceptibility", "susceptibility_uncertainty"
                ),
                sz_label=create_val_err_label("s_zscore", "s_zscore_error"),
                r_label=create_val_err_label(
                    "robustness", "robustness_uncertainty"
                ),
                rz_label=create_val_err_label("r_zscore", "r_zscore_error"),
            )
        )
        .mark_circle(color="black", filled=True)
        .encode(
            alt.X("susceptibility:Q", title=""),
            alt.Y(
                "model:N",
                title="",
                sort=alt.SortField("susceptibility", order="descending"),
            ),
            alt.Color("foundation:N"),
            # alt.Column("foundation:N")
        )
        .facet(facet="foundation:N", columns=3)
        .properties(
            title="Moral Susceptibility Z-Score by Model", width=100, height=75
        )
    )
    return


if __name__ == "__main__":
    app.run()
