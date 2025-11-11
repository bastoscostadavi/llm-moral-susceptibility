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
def _(pl, table):
    def create_empty_count_table():
        return pl.DataFrame(
            [
                (m, q, r, 0)
                for m in table["persona"]
                .select("model")
                .unique()
                .to_series()
                .to_list()
                for q in range(1, 31)
                for r in range(6)
            ],
            ["model", "question_id", "rating", "count"],
            orient="row",
        )


    def entropy_columns(df, *, count_smoothing=1.0):
        a = count_smoothing
        ndf = (
            create_empty_count_table()
            .update(
                df.filter(pl.col("rating") >= 0)
                .group_by("model", "question_id", "rating")
                .agg(pl.col("rating").count().alias("count")),
                on=["model", "question_id", "rating"],
                how="left",
            )
            .with_columns(
                p=(
                    (pl.col("count") + pl.lit(a))
                    / (
                        pl.col("count").sum().over("model", "question_id")
                        + pl.lit(a * 6)
                    )
                ),  # Laplace's smoothing
            )
            .with_columns(ln_invp=-pl.col("p").log())
            .with_columns(p_ln_invp=pl.col("p") * pl.col("ln_invp"))
        )

        return ndf.select("model", *(c for c in ndf.columns if c != "model")).sort(
            "model", "question_id", "rating"
        )


    def remove_self_from_model_name(df):
        return (
            df.with_columns(
                pl.col("model")
                .str.extract("([a-zA-Z0-9-.]+)_", 1)
                .alias("clean_model"),
            )
            .drop("model")
            .rename({"clean_model": "model"})
            .select("model", *(c for c in df.columns if c != "model"))
        )
    return entropy_columns, remove_self_from_model_name


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
def _(persona_df, plot_question_rating_ridgelines):
    plot_question_rating_ridgelines(persona_df)
    return


@app.cell
def _(plot_question_rating_ridgelines, self_df):
    plot_question_rating_ridgelines(self_df)
    return


@app.cell
def _(alt, both_df, persona_df, pl, self_df):
    import math

    (
        alt.Chart(
            title="Personas",
            data=persona_df.group_by("model").agg(
                entropy=(pl.col("p") * pl.col("ln_invp")).sum()
            ),
        )
        .mark_bar(size=15)
        .encode(
            alt.X("entropy:Q"),
            alt.Y("model:N", sort=alt.SortField("entropy", order="descending")),
        )
        | alt.Chart(
            title="Self",
            data=self_df.group_by("model").agg(
                entropy=(pl.col("p") * pl.col("ln_invp")).sum()
            ),
        )
        .mark_bar(size=15)
        .encode(
            alt.X("entropy:Q"),
            alt.Y("model:N", sort=alt.SortField("entropy", order="descending")),
        )
        | alt.Chart(
            title="Both - KL-Divergence",
            data=both_df.group_by("model").agg(
                rel_entropy=pl.col.rel_entropy.sum()
            ),
        )
        .mark_bar(size=15)
        .encode(
            alt.X("rel_entropy:Q"),
            alt.Y(
                "model:N", sort=alt.SortField("rel_entropy", order="descending")
            ),
        )
    )
    return


@app.cell
def _(entropy_columns, remove_self_from_model_name, table):
    _a = 1 / 6
    self_df = entropy_columns(
        remove_self_from_model_name(table["self"]), count_smoothing=_a
    )
    persona_df = entropy_columns(table["persona"], count_smoothing=_a)

    {"self_df": self_df, "persona_df": persona_df}
    return persona_df, self_df


@app.cell
def _(persona_df, pl, self_df):
    both_df = persona_df.join(
        self_df.select(
            pl.all().name.replace("^(p|ln_invp|p_ln_invp|count)$", "self_$1")
        ),
        on=["model", "question_id", "rating"],
        how="left",
    ).with_columns(
        rel_entropy=(-pl.col.p * (pl.col.ln_invp - pl.col.self_ln_invp))
    )

    both_df
    return (both_df,)


if __name__ == "__main__":
    app.run()
