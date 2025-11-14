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


app._unparsable_cell(
    r"""
     import altair as alt
    import polars as pl
    from pathlib import Path
    import re
    import marimo as mo
    import vl_convert
    import pyarrow
    """,
    name="_"
)


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
    def create_empty_count_table(model_list, *, num_questions=30, num_ratings=6):
        return pl.DataFrame(
            [
                (m, q, r, 0)
                for m in model_list
                for q in range(1, num_questions + 1)
                for r in range(num_ratings)
            ],
            ["model", "question_id", "rating", "count"],
            orient="row",
        )


    def complete_count_table(df):
        return create_empty_count_table(
            df.select("model").unique().to_series().to_list()
        ).update(
            df.filter(pl.col("rating") >= 0)
            .group_by("model", "question_id", "rating")
            .agg(pl.col("rating").count().alias("count")),
            on=["model", "question_id", "rating"],
            how="left",
        )


    def laplace_smoothing(c, a, N, *, window_cols=[]):
        return (pl.col(c) + pl.lit(a)) / (
            pl.col(c).sum().over(*window_cols) + pl.lit(N * a)
        )


    def compute_empirical_probability(df, f, *fargs, **fkwargs):
        return df.with_columns(p=f(*fargs, **fkwargs))


    def entropy_columns(df):
        return df.with_columns(ln_invp=-pl.col("p").log()).with_columns(
            p_ln_invp=pl.col("p") * pl.col("ln_invp")
        )


    # def entropy_columns(df, *, count_smoothing=1.0):
    #     a = count_smoothing
    #     ndf = (
    #         create_empty_count_table(
    #             df.select("model").unique().to_series().to_list()
    #         )
    #         .update(
    #             df.filter(pl.col("rating") >= 0)
    #             .group_by("model", "question_id", "rating")
    #             .agg(pl.col("rating").count().alias("count")),
    #             on=["model", "question_id", "rating"],
    #             how="left",
    #         )
    #         .with_columns(
    #             p=laplace_smoothing(
    #                 "count", a, 6, window_cols=["model", "question_id"]
    #             )
    #         )
    #         .with_columns(ln_invp=-pl.col("p").log())
    #         .with_columns(p_ln_invp=pl.col("p") * pl.col("ln_invp"))
    #     )

    #     return ndf.select("model", *(c for c in ndf.columns if c != "model")).sort(
    #         "model", "question_id", "rating"
    #     )


    def remove_self_from_model_name(df):
        return df.with_columns(
            model=pl.col("model").str.extract("([a-zA-Z0-9-.]+)_", 1)
        )
    return (
        complete_count_table,
        compute_empirical_probability,
        entropy_columns,
        laplace_smoothing,
        remove_self_from_model_name,
    )


@app.cell
def _():
    MODEL_COLORS = {
        "Average across models": "#4D4D4D",
        "claude-haiku-4-5": "#F9C784",
        "claude-sonnet-4-5": "#E67E22",
        "gpt-4.1-nano": "#D9F0D3",
        "gpt-4o-mini": "#A6DBA0",
        "gpt-4.1": "#52B788",
        "grok-4": "#BDA0E3",
        "gpt-4o": "#2F855A",
        "gpt-4.1-mini": "#74C69D",
        "gemini-2.5-flash": "#F2D16B",
        "gemini-2.5-flash-lite": "#F9E69F",
        "grok-4-fast": "#7E57C2",
        # Pale red gradient reserved for GPT-5 family (lightest to darkest)
        "gpt-5-nano": "#F8C8C8",
        "gpt-5-mini": "#F29B9B",
        "gpt-5": "#E36B6B",
        "deepseek-chat-v3.1": "#5B4B8A",
        "llama-4-maverick": "#4A90E2",
        "llama-4-scout": "#4A90E2",
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
    return MODELS_ORDER, MODEL_COLORS


@app.cell
def _(MODELS_ORDER, MODEL_COLORS, alt):
    def plot_question_rating_ridgelines(df, *, step=10, overlap=1):
        _color_domain = [model for model in MODELS_ORDER if model in MODEL_COLORS]
        _color_range = [MODEL_COLORS[model] for model in _color_domain]
        _color_scale = alt.Scale(domain=_color_domain, range=_color_range)

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
                alt.Fill("model:N", scale=_color_scale, legend=None),
            )
            .facet(
                row=(
                    alt.Row("question_id:N")
                    .title(None)
                    .header(labelAngle=0, labelAlign="left")
                ),
                column=alt.Column("model:N", sort=MODELS_ORDER).title(None),
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
def _(DATA_DIR, persona_df, plot_question_rating_ridgelines):
    personas_ridgeline = plot_question_rating_ridgelines(persona_df)
    personas_ridgeline.save(
        DATA_DIR.parent / "results" / "ratings_distribution_personas.pdf"
    )
    personas_ridgeline
    return


@app.cell
def _(DATA_DIR, plot_question_rating_ridgelines, self_df):
    self_ridgeline = plot_question_rating_ridgelines(self_df)
    self_ridgeline.save(
        DATA_DIR.parent / "results" / "ratings_distribution_self.pdf"
    )
    self_ridgeline
    return


@app.cell
def _(mo):
    a = mo.ui.slider(steps=[0.1, 1 / 6, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5])
    return (a,)


@app.cell
def _(a):
    a
    return


@app.cell
def _(
    DATA_DIR,
    MODELS_ORDER,
    MODEL_COLORS,
    alt,
    both_df,
    persona_df,
    pl,
    self_df,
):
    import math

    _color_domain = [model for model in MODELS_ORDER if model in MODEL_COLORS]
    _color_range = [MODEL_COLORS[model] for model in _color_domain]
    _color_scale = alt.Scale(domain=_color_domain, range=_color_range)

    _p_ent = (
        alt.Chart(
            title="Model Entropy - Pesonas",
            data=persona_df.group_by("model").agg(
                entropy=(pl.col("p") * pl.col("ln_invp")).sum(),
                mean_entropy_per_question=(pl.col.p * pl.col.ln_invp).mean(),
                std_entropy_per_question=(pl.col.p * pl.col.ln_invp).std(),
            ),
        )
        .mark_bar(size=15, opacity=0.6)
        .encode(
            alt.X("entropy:Q", title="Entropy"),
            alt.Y(
                "model:N", sort=MODELS_ORDER, title=None
            ),  # alt.SortField("entropy", order="descending")),
            alt.Color(
                "model:N",
                title="",
                sort=MODELS_ORDER,
                legend=None,
                scale=_color_scale,
            ),
        )
    )

    _s_ent = (
        alt.Chart(
            title="Model Entropy - Without Personas",
            data=self_df.group_by("model").agg(
                entropy=(pl.col("p") * pl.col("ln_invp")).sum(),
                mean_entropy_per_question=(pl.col.p * pl.col.ln_invp).mean(),
                std_entropy_per_question=(pl.col.p * pl.col.ln_invp).std(),
            ),
        )
        .mark_bar(size=15, opacity=0.6)
        .encode(
            alt.X("entropy:Q", title="Entropy"),
            alt.Y(
                "model:N", sort=MODELS_ORDER, title=None
            ),  # alt.SortField("entropy", order="descending")),
            alt.Color(
                "model:N",
                title="",
                sort=MODELS_ORDER,
                legend=None,
                scale=_color_scale,
            ),
        )
    )

    _b_ent = (
        alt.Chart(
            title="Model Entropy - Relative",
            data=both_df.group_by("model").agg(
                rel_entropy=pl.col.rel_entropy.sum(),
                mean_entropy_per_question=(pl.col.rel_entropy).mean(),
                std_entropy_per_question=(pl.col.rel_entropy).std(),
            ),
        )
        .mark_bar(size=15, opacity=0.6)
        .encode(
            alt.X("rel_entropy:Q", title="Entropy"),
            # alt.X("mean_entropy_per_question:Q"),
            alt.Y(
                "model:N",
                sort=MODELS_ORDER,  # alt.SortField("rel_entropy", order="descending")
                title=None,
            ),
            alt.Color(
                "model:N",
                title="",
                sort=MODELS_ORDER,
                legend=None,
                scale=_color_scale,
            ),
        )
    )

    _p_ent.save(DATA_DIR.parent / "results" / "entropy_personas.pdf")
    _s_ent.save(DATA_DIR.parent / "results" / "entropy_self.pdf")
    _b_ent.save(DATA_DIR.parent / "results" / "entropy_relative.pdf")

    _combined = (
        (_p_ent | _s_ent | _b_ent)
        .resolve_scale(x="shared")
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
    )

    _combined.save(DATA_DIR.parent / "results" / "entropy_combined_plot.pdf")
    _combined
    return


@app.cell
def _(
    a,
    complete_count_table,
    compute_empirical_probability,
    entropy_columns,
    laplace_smoothing,
    pl,
    remove_self_from_model_name,
    table,
):
    _a = 0.1

    self_df = entropy_columns(
        compute_empirical_probability(
            complete_count_table(remove_self_from_model_name(table["self"])),
            laplace_smoothing,
            "count",
            a.value,
            6,
            window_cols=["model", "question_id"],
        )
    )

    persona_df = entropy_columns(
        compute_empirical_probability(
            complete_count_table(table["persona"]),
            laplace_smoothing,
            "count",
            a.value,
            6,
            window_cols=["model", "question_id"],
        )
    )

    both_df = persona_df.join(
        self_df, on=["model", "question_id", "rating"], how="left", suffix="_self"
    ).with_columns(
        rel_entropy=(pl.col.p_self * (pl.col.ln_invp - pl.col.ln_invp_self)),
        sym_entropy=(
            pl.col.p_self * (pl.col.ln_invp - pl.col.ln_invp_self)
            + pl.col.p * (pl.col.ln_invp_self - pl.col.ln_invp)
        )
        / pl.lit(2),
    )

    {"self_df": self_df, "persona_df": persona_df, "both_df": both_df}
    return both_df, persona_df, self_df


if __name__ == "__main__":
    app.run()
