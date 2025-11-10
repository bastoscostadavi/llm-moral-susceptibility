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
    MODEL_COLORS = {
        "claude-haiku-4-5": "#F9C784",
        "claude-sonnet-4-5": "#E67E22",
        "gpt-4.1-nano": "#D9F0D3",
        "gpt-4o-mini": "#A6DBA0",
        "gpt-4.1": "#52B788",
        "grok-4": "#E07BB6",
        "gpt-4o": "#2F855A",
        "gpt-4.1-mini": "#74C69D",
        "gemini-2.5-flash": "#F2D16B",
        "gemini-2.5-flash-lite": "#F9E69F",
        "grok-4-fast": "#C24796",
        # Pale red gradient reserved for GPT-5 family (lightest to darkest)
        "gpt-5-nano": "#F8C8C8",
        "gpt-5-mini": "#F29B9B",
        "gpt-5": "#E36B6B",
        "deepseek-v3": "#A48DEB",
        "deepseek-v3.1": "#5B4B8A",
        "llama-4-maverick": "#4A90E2",
        "llama-4-scout": "#63B3FF",
    }

    FOUNDATION_COLORS = ["blue", "green", "yellow", "red", "lime"]

    FOUNDATIONS_ORDER = [
        "All Foundations",
        "Harm/Care",
        "Fairness/Reciprocity",
        "In-group/Loyalty",
        "Authority/Respect",
        "Purity/Sanctity",
    ]

    MODELS_ORDER = sorted(MODEL_COLORS)
    return FOUNDATIONS_ORDER, MODEL_COLORS, MODELS_ORDER


@app.cell
def _(FOUNDATIONS_ORDER, MODEL_COLORS, MODELS_ORDER, RESULTS_DIR, alt, df, pl):
    _chart_df = df

    _p_data = alt.Chart(
        _chart_df.with_columns(
            f_color=pl.when(pl.col("foundation") == "All Foundations")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        ),
        height=100,
        width=150,
    )

    _color_domain = [model for model in MODELS_ORDER if model in MODEL_COLORS]
    _color_range = [MODEL_COLORS[model] for model in _color_domain]
    _color_scale = alt.Scale(domain=_color_domain, range=_color_range)

    _foundation_y = alt.Y(
        "foundation:N",
        title=None,
        sort=FOUNDATIONS_ORDER,
        scale=alt.Scale(paddingInner=0.0, paddingOuter=0.05),
    )

    _p_bars = (
        _p_data
        # .transform_filter(alt.datum.foundat)
        .mark_bar(size=10).encode(
            alt.X(
                "susceptibility:Q",
                title="Susceptibility",
                scale=alt.Scale(domain=[0, 1]),
            ),
            _foundation_y,
            alt.Color(
                "model:N",
                title="",
                sort=MODELS_ORDER,
                legend=None,
                scale=_color_scale,
            ),
            opacity=alt.when(alt.datum.foundation == "All Foundations")
            .then(alt.value(0.6))
            .otherwise(alt.value(0.3)),
        )
    )

    _p_error = _p_data.mark_errorbar().encode(
        alt.X(
            "susceptibility:Q",
            title="Susceptibility",
            scale=alt.Scale(domain=[0, 1]),
        ),
        alt.XError("susceptibility_uncertainty:Q"),
        _foundation_y,
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
        _foundation_y,
        alt.X(
            "susceptibility:Q",
            title="Susceptibility",
            scale=alt.Scale(domain=[0, 1]),
        ),
        alt.Text("s_label:N"),
        color=alt.value("black"),
        opacity=alt.value(1.0),
    )

    _p = (
        (_p_bars + _p_error + _p_label)
        .facet(
            facet=alt.Facet("model:N", title="Model", sort=MODELS_ORDER),
            columns=3,
            title="",
        )
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
        .resolve_scale(x="shared")
        # .properties(
        #     width=100,
        #     height=50,
        # )
    )

    _p.save(RESULTS_DIR / "susceptibility_bars.pdf")

    _p
    return


@app.cell
def _(FOUNDATIONS_ORDER, MODEL_COLORS, MODELS_ORDER, RESULTS_DIR, alt, df, pl):
    _chart_df = df

    _p_data = alt.Chart(
        _chart_df.with_columns(
            f_color=pl.when(pl.col("foundation") == "All Foundations")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        ),
        height=100,
        width=150,
    )

    _color_domain = [model for model in MODELS_ORDER if model in MODEL_COLORS]
    _color_range = [MODEL_COLORS[model] for model in _color_domain]
    _color_scale = alt.Scale(domain=_color_domain, range=_color_range)

    _foundation_y = alt.Y(
        "foundation:N",
        title=None,
        sort=FOUNDATIONS_ORDER,
        scale=alt.Scale(paddingInner=0.0, paddingOuter=0.05),
    )

    _p_bars = (
        _p_data
        # .transform_filter(alt.datum.foundat)
        .mark_bar(size=10).encode(
            alt.X(
                "robustness:Q",
                title="Robustness",
                scale=alt.Scale(domain=[0, 1]),
            ),
            _foundation_y,
            alt.Color(
                "model:N", title="", legend=None, sort=MODELS_ORDER, scale=_color_scale
            ),
            opacity=alt.when(alt.datum.foundation == "All Foundations")
            .then(alt.value(0.6))
            .otherwise(alt.value(0.3)),
        )
    )

    _p_error = _p_data.mark_errorbar().encode(
        alt.X(
            "robustness:Q",
            title="Robustness",
            scale=alt.Scale(domain=[0, 1]),
        ),
        alt.XError("robustness_uncertainty:Q"),
        _foundation_y,
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
        _foundation_y,
        alt.X(
            "robustness:Q",
            title="Robustness",
            scale=alt.Scale(domain=[0, 1]),
        ),
        alt.Text("r_label:N"),
        color=alt.value("black"),
        opacity=alt.value(1.0),
    )

    _p = (
        (_p_bars + _p_error + _p_label)
        .facet(
            facet=alt.Facet("model:N", title="Model", sort=MODELS_ORDER),
            columns=3,
            title="",
        )
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
        .resolve_scale(x="shared")
        # .properties(
        #     width=100,
        #     height=50,
        # )
    )

    _p.save(RESULTS_DIR / "robustness_bars.pdf")

    _p
    return


@app.cell
def _(MODEL_COLORS, MODELS_ORDER, RESULTS_DIR, alt, table):
    _moral_metrics = table["moral_metrics"]
    model_order = sorted(_moral_metrics.get_column("model").to_list())

    _color_domain = [model for model in MODELS_ORDER if model in MODEL_COLORS]
    _color_range = [MODEL_COLORS[model] for model in _color_domain]

    _extra_models = [model for model in model_order if model not in _color_domain]
    if _extra_models:
        _color_domain.extend(_extra_models)
        _color_range.extend("#7f7f7f" for _ in _extra_models)

    _color_scale = alt.Scale(domain=_color_domain, range=_color_range)

    _p_data = alt.Chart(_moral_metrics, height=200, width=200)

    _p_bars_s = _p_data.mark_bar(size=10, opacity=0.6).encode(
        alt.Y("model:N", sort=model_order, title=None),
        alt.X(
            "susceptibility:Q",
            title="Susceptibility",
            scale=alt.Scale(domain=[0, 1]),
        ),
        alt.Color("model:N", legend=None, sort=model_order, scale=_color_scale),
    )

    _p_error_s = _p_data.mark_errorbar().encode(
        alt.Y("model:N", sort=model_order),
        alt.X(
            "susceptibility:Q",
            title="Susceptibility",
            scale=alt.Scale(domain=[0, 1]),
        ),
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

    _p_bars_r = _p_data.mark_bar(size=10, opacity=0.6).encode(
        alt.Y("model:N", sort=model_order, title=None),
        alt.X(
            "robustness:Q",
            title="Robustness",
            scale=alt.Scale(domain=[0, 1]),
        ),
        alt.Color("model:N", legend=None, sort=model_order, scale=_color_scale),
    )

    _p_error_r = _p_data.mark_errorbar().encode(
        alt.Y("model:N", sort=model_order),
        alt.X(
            "robustness:Q",
            title="Robustness",
            scale=alt.Scale(domain=[0, 1]),
        ),
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

    _robust_base = _p_bars_r + _p_error_r + _p_label_r
    _suscept_base = _p_bars_s + _p_error_s + _p_label_s

    _robust_chart = _robust_base.configure_axis(grid=False).configure_view(strokeWidth=0)
    _suscept_chart = _suscept_base.configure_axis(grid=False).configure_view(strokeWidth=0)

    _combined_chart = (
        (_robust_base | _suscept_base)
        .resolve_scale(y="independent", x="independent")
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
    )

    _robust_chart.save(RESULTS_DIR / "moral_metrics_robustness_bars.pdf")

    _suscept_chart.save(RESULTS_DIR / "moral_metrics_susceptibility_bars.pdf")

    _combined_chart.save(RESULTS_DIR / "moral_metrics_overall_bars.pdf")

    _combined_chart
    return


if __name__ == "__main__":
    app.run()
