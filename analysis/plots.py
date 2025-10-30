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
    _foo = [
        "gpt-4.1-nano",
        "gemini-2.5-flash-lite",
        "gpt-4.1",
        "grok-4-fast",
        "gpt-4.1-mini",
        "claude-sonnet-4-5",
        "gpt-4o-mini",
        "gpt-4o",
        "claude-haiku-4-5",
    ]

    _foo.sort()

    _foo
    return


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

    MODELS_ORDER = [
        "claude-haiku-4-5",
        "gpt-4.1-nano",
        "gpt-4o-mini",
        "gpt-4.1",
        "grok-4",
        "claude-sonnet-4-5",
        "gpt-4o",
        "gpt-4.1-mini",
        "gemini-2.5-flash-lite",
        "grok-4-fast",
    ]
    return FOUNDATIONS_ORDER, MODELS_ORDER


@app.cell
def _(FOUNDATIONS_ORDER, MODELS_ORDER, RESULTS_DIR, alt, df, pl):
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
            alt.X("susceptibility:Q", title="Susceptibility"),
            alt.Y("foundation:N", title=None, sort=FOUNDATIONS_ORDER),
            alt.Color("model:N", title="", sort=MODELS_ORDER, legend=None),
            opacity=alt.when(alt.datum.foundation == "All Foundations")
            .then(alt.value(0.6))
            .otherwise(alt.value(0.3)),
        )
    )

    _p_error = _p_data.mark_errorbar().encode(
        alt.X("susceptibility:Q", title="Susceptibility"),
        alt.XError("susceptibility_uncertainty:Q"),
        alt.Y("foundation:N", title=None, sort=FOUNDATIONS_ORDER),
    )

    _p_label = _p_bars.mark_text(
        align="left",
        baseline="middle",
        dx=8.5,
        dy=0,
        fontSize=8,
        opacity=1.0,
        color="#000000",
    ).encode(
        alt.Y("foundation:N"),
        alt.X("susceptibility:Q", title="Susceptibility"),
        alt.Text("s_label:N"),
        color=alt.value("black"),
        opacity=alt.value(1.0),
    )

    _p = (
        (_p_bars + _p_error + _p_label)
        .facet(facet=alt.Facet("model:N",title="Model",sort=MODELS_ORDER), columns=5, title="")
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
        .resolve_scale(x="independent")
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
def _(alt):
    alt.Facet
    return


@app.cell
def _(FOUNDATIONS_ORDER, MODELS_ORDER, RESULTS_DIR, alt, df, pl):
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
            alt.X("s_zscore:Q", title="Susceptibility Z-Score"),
            alt.Y("foundation:N", title=None, sort=FOUNDATIONS_ORDER),
            alt.Color("model:N", title="", legend=None),
            opacity=alt.when(alt.datum.foundation == "All Foundations")
            .then(alt.value(0.6))
            .otherwise(alt.value(0.3)),
        )
    )

    _p_error = _p_data.mark_errorbar().encode(
        alt.X("s_zscore:Q", title="Susceptibility Z-Score"),
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
        alt.X("s_zscore:Q", title="Susceptibility Z-Score"),
        alt.Text("sz_label:N"),
        color=alt.value("black"),
        opacity=alt.value(1.0),
    )

    _p = (
        (_p_bars + _p_error + _p_label)
        .facet(facet=alt.Facet("model:N",title="Model",sort=MODELS_ORDER), columns=5, title="")
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
        .resolve_scale(x="independent")
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
def _(FOUNDATIONS_ORDER, MODELS_ORDER, RESULTS_DIR, alt, df, pl):
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
            alt.X("robustness:Q", title="Robustness"),
            alt.Y("foundation:N", title=None, sort=FOUNDATIONS_ORDER),
            alt.Color("model:N", title="", legend=None),
            opacity=alt.when(alt.datum.foundation == "All Foundations")
            .then(alt.value(0.6))
            .otherwise(alt.value(0.3)),
        )
    )

    _p_error = _p_data.mark_errorbar().encode(
        alt.X("robustness:Q", title="Robustness"),
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
        alt.X("robustness:Q", title="Robustness"),
        alt.Text("r_label:N"),
        color=alt.value("black"),
        opacity=alt.value(1.0),
    )

    _p = (
        (_p_bars + _p_error + _p_label)
        .facet(facet=alt.Facet("model:N",title="Model",sort=MODELS_ORDER), columns=5, title="")
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
        .resolve_scale(x="independent")
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
def _(FOUNDATIONS_ORDER, MODELS_ORDER, RESULTS_DIR, alt, df, pl):
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
            alt.X("r_zscore:Q", title="Robustness Z-Score"),
            alt.Y("foundation:N", title=None, sort=FOUNDATIONS_ORDER),
            alt.Color("model:N", title="", legend=None),
            opacity=alt.when(alt.datum.foundation == "All Foundations")
            .then(alt.value(0.6))
            .otherwise(alt.value(0.3)),
        )
    )

    _p_error = _p_data.mark_errorbar().encode(
        alt.X("r_zscore:Q", title="Robustness Z-Score"),
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
        alt.X("r_zscore:Q", title="Robustness Z-Score"),
        alt.Text("rz_label:N"),
        color=alt.value("black"),
        opacity=alt.value(1.0),
    )

    _p = (
        (_p_bars + _p_error + _p_label)
        .facet(facet=alt.Facet("model:N",title="Model",sort=MODELS_ORDER), columns=5, title="")
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


@app.cell
def _(RESULTS_DIR, alt, table):
    _moral_metrics = table["moral_metrics"]
    model_order = sorted(_moral_metrics.get_column("model").to_list())

    _p_data = alt.Chart(_moral_metrics, height=200, width=200)

    _p_bars_s = _p_data.mark_bar(size=15, opacity=0.6).encode(
        alt.Y("model:N", sort=model_order, title=None),
        alt.X("susceptibility:Q", title="Susceptibility"),
        alt.Color("model:N", legend=None, sort=model_order),
    )

    _p_error_s = _p_data.mark_errorbar().encode(
        alt.Y("model:N", sort=model_order),
        alt.X("susceptibility:Q", title="Susceptibility"),
        alt.XError("susceptibility_uncertainty:Q"),
    )

    _p_label_s = _p_bars_s.mark_text(
        align="left",
        baseline="middle",
        dx=12,
        dy=0,
        fontSize=9,
        opacity=1.0,
        color="#000000",
    ).encode(alt.Text("s_label"), color=alt.value("black"), opacity=alt.value(1.0))

    _p_bars_r = _p_data.mark_bar(size=15, opacity=0.6).encode(
        alt.Y("model:N", sort=model_order, title=None),
        alt.X("robustness:Q", title="Robustness"),
        alt.Color("model:N", legend=None, sort=model_order),
    )

    _p_error_r = _p_data.mark_errorbar().encode(
        alt.Y("model:N", sort=model_order),
        alt.X("robustness:Q", title="Robustness"),
        alt.XError("robustness_uncertainty:Q"),
    )

    _p_label_r = _p_bars_r.mark_text(
        align="left",
        baseline="middle",
        dx=12,
        dy=0,
        fontSize=9,
        opacity=1.0,
        color="#000000",
    ).encode(alt.Text("r_label"), color=alt.value("black"), opacity=alt.value(1.0))

    _p = (
        (
            (_p_bars_r + _p_error_r + _p_label_r)
            | (_p_bars_s + _p_error_s + _p_label_s)
        )
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
        .resolve_scale(y="independent", x="independent")
    )

    _p.save(RESULTS_DIR / "moral_metrics_overall_bars.png")
    _p.save(RESULTS_DIR / "moral_metrics_overall_bars.pdf")

    _p
    return


if __name__ == "__main__":
    app.run()
