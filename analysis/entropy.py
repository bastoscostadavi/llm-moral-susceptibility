# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==5.5.0",
#     "anthropic==0.72.0",
#     "matplotlib==3.10.7",
#     "plotnine==0.15.1",
#     "polars==1.35.1",
#     "pyarrow==22.0.0",
#     "pyobsplot==0.5.4",
#     "seaborn==0.13.2",
#     "vega-datasets==0.9.0",
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
    import re
    import marimo as mo
    import vl_convert
    import pyarrow
    return Path, alt, pl


@app.cell
def _(Path):
    def _get_data_directory():
        cwd = Path.cwd()
        if (r := cwd / "data") in cwd.iterdir() and r.is_dir():
            return r
        elif cwd.name == "data":
            return cwd
        elif cwd.name == "analysis" or cwd.name == "results":
            return cwd.parent / "data"
        else:
            raise Exception(
                "Can't find the results. Run this script either from the 'root', 'analysis' or 'results' directories."
            )


    DATA_DIR = _get_data_directory()
    return (DATA_DIR,)


@app.cell
def _(pl):
    def load_results(data_dir):
        persona_tables = []
        self_tables = []
        for f in data_dir.glob("*csv"):
            df = pl.read_csv(
                f,
                infer_schema_length=10000,
                schema_overrides={"response": pl.Utf8},
            ).with_columns(model=pl.lit(f.stem))
            if "persona_id" in df.columns:
                persona_tables.append(
                    df.select(
                        "model",
                        "persona_id",
                        "question_id",
                        "run_index",
                        "rating",
                    )
                )
            else:
                self_tables.append(
                    df.select(
                        "model",
                        "question_id",
                        "run_index",
                        "rating",
                    )
                )
        return dict(persona=pl.concat(persona_tables), self=pl.concat(self_tables))
    return (load_results,)


@app.cell
def _(DATA_DIR, load_results):
    table = load_results(DATA_DIR)

    table
    return (table,)


@app.cell
def _(pl):
    def entropy_columns(df):
        return (
            df.filter(pl.col("rating") >= 0)
            .group_by("model", "question_id", "rating")
            .agg(pl.col("rating").count().alias("n"))
            .with_columns(
                N=pl.col("n").sum().over("model", "question_id"),
                p=(pl.col("n") / pl.col("n").sum().over("model", "question_id")),
            )
            .with_columns(ln_p=pl.col("p").log())
            .with_columns(p_ln_p=pl.col("p") * pl.col("ln_p"))
            .select("model", "question_id", "rating", "p", "ln_p", "p_ln_p")
            .sort("model", "question_id", "rating")
        )
    return (entropy_columns,)


@app.cell
def _(alt):
    def plot_question_rating_ridgelines(df, *, step=10, overlap=1):
        return (
            alt.Chart(df, height=step)
            .mark_area(
                interpolate="monotone",
                fillOpacity=0.8,
                stroke="lightgray",
                strokeWidth=0.5,
            )
            .encode(
                alt.X("rating:N").title("Rating"),
                alt.Y("p:Q").axis(None).scale(range=[step, -step * overlap]),
                alt.Fill("model:N"),
            )
            .facet(
                row=(
                    alt.Row("question_id:N")
                    .title(None)
                    .header(labelAngle=0, labelAlign="left")
                ),
                column=alt.Column("model:N").title(None),
            )
            .properties(
                title="Ratings distribution over questions by model",
                bounds="flush",
            )
            .configure_facet(spacing=0)
            .configure_view(stroke=None)
            .configure_title(anchor="end")
        )
    return (plot_question_rating_ridgelines,)


@app.cell
def _(entropy_columns, plot_question_rating_ridgelines, table):
    plot_question_rating_ridgelines(entropy_columns(table["self"]))
    return


@app.cell
def _(entropy_columns, plot_question_rating_ridgelines, table):
    plot_question_rating_ridgelines(entropy_columns(table["persona"]))
    return


@app.cell
def _(alt, entropy_columns, pl, table):
    import math

    (
        alt.Chart(
            data=entropy_columns(table["persona"])
            .group_by("model")
            .agg(entropy=(-pl.col("p") * pl.col("ln_p")).sum())
        )
        .mark_bar()
        .encode(
            alt.X("entropy:Q"),
            alt.Y("model:N", sort=alt.SortField("entropy", order="descending")),
        )
    )
    return


@app.cell
def _(entropy_columns, table):
    entropy_columns(table["self"])
    return


@app.cell
def _(entropy_columns, pl, table):
    (
        entropy_columns(table["persona"]).join(
            entropy_columns(table["self"]).select(
                "model", "question_id", "rating", pl.col("ln_p").alias("self_ln_p")
            ),
            on=["model", "question_id", "rating"],
        )
    )
    return


@app.cell
def _(entropy_columns, pl, table):
    entropy_columns(table["self"]).select(
        "model", "question_id", "rating", pl.col("ln_p").alias("self_ln_p")
    )
    return


@app.cell
def _(entropy_columns, table):
    self_df = entropy_columns(table["self"])
    persona_df = entropy_columns(table["persona"])
    return (self_df,)


@app.cell
def _(pl, self_df):
    self_df.with_columns(model=pl.col("model").str.split("_"))
    return


if __name__ == "__main__":
    app.run()
