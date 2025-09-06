import os
import json
import click
from loguru import logger
import pandas as pd
from typing import Optional, Any

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None

from pipeline.data_loader import DataLoader
from pipeline.data_cleaner import DataCleaner
from pipeline.imputer import DataImputer
from pipeline.analyzer import Analyzer
from pipeline.reporter import Reporter

# optional pretty ASCII tables
try:
    from tabulate import tabulate
except Exception:
    tabulate = None

# configure logging
CONFIG_FILE = ".rtgs_config.json"
os.makedirs("logs", exist_ok=True)
logger.add("logs/pipeline.json", serialize=True, rotation="1 MB")

def save_last_path(file_path: str) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_file": file_path}, f)

def load_last_path() -> Optional[str]:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("last_file")
    return None

# transformers availability flag and pipeline reference
_try_transformers = False
_hf_pipeline = None
try:
    from transformers import pipeline as hf_pipeline  # type: ignore
    _try_transformers = True
    _hf_pipeline = hf_pipeline
except Exception:
    _try_transformers = False
    _hf_pipeline = None

class TransformerAgent:
    """
    Minimal local transformer agent wrapper using Hugging Face pipeline.
    Requires `transformers` and a backend (torch). Use env HF_MODEL to change model.
    """
    def __init__(self, model: Optional[str] = None, device: Optional[int] = None, pipeline_kwargs: Optional[dict] = None):
        if not _try_transformers or _hf_pipeline is None:
            raise ImportError("transformers not installed. Install with: pip install transformers[torch]")
        self.model = model or os.getenv("HF_MODEL", "gpt2")
        env_dev = os.getenv("HF_DEVICE")
        if device is None and env_dev is not None:
            try:
                device = int(env_dev)
            except Exception:
                device = -1
        self.device = -1 if device is None else device
        self.pipeline_kwargs = pipeline_kwargs or {}
        device_arg = self.device if self.device is not None else -1
        self._pipe = _hf_pipeline("text-generation", model=self.model, device=device_arg, **self.pipeline_kwargs)

    def generate(self, prompt: str, max_length: int = 256, do_sample: bool = False, temperature: float = 0.0) -> str:
        try:
            out = self._pipe(prompt, max_length=max_length, do_sample=do_sample, temperature=temperature, return_full_text=False)
            if isinstance(out, list) and out:
                first = out[0]
                if isinstance(first, dict):
                    return first.get("generated_text") or first.get("text") or json.dumps(first)
                return str(first)
            return str(out)
        except Exception as e:
            return f"[generator-error] {e}"

    def generate_json(self, prompt: str, max_length: int = 512) -> Optional[Any]:
        resp = self.generate(prompt, max_length=max_length)
        try:
            return json.loads(resp)
        except Exception:
            # attempt to extract first JSON substring
            start_idxs = [i for i in (resp.find("{"), resp.find("[")) if i >= 0]
            start = min(start_idxs) if start_idxs else -1
            if start >= 0:
                sub = resp[start:]
                try:
                    return json.loads(sub)
                except Exception:
                    pass
        return None

@click.group()
def cli():
    """RTGS-Style AI Analyst CLI"""
    pass

@cli.command()
@click.argument("file_path", required=False)
def load(file_path):
    """Load dataset and remember its path"""
    if not file_path:
        file_path = load_last_path()
        if not file_path:
            click.echo(" No dataset path found. Use: cli.py load <path>")
            return
    if not os.path.exists(file_path):
        click.echo(f" File not found: {file_path}")
        return
    save_last_path(file_path)
    loader = DataLoader(file_path)
    df = loader.load()
    click.echo(f" Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

@cli.command()
def clean():
    """Clean dataset (uses last loaded path)"""
    file_path = load_last_path()
    if not file_path:
        click.echo(" Please load a dataset first using `load <path>`")
        return
    loader = DataLoader(file_path)
    df = loader.load()
    cleaner = DataCleaner(df)
    df = cleaner.clean()
    click.echo(" Dataset cleaned successfully")

@cli.command()
def impute():
    """Impute missing values (uses last loaded path)"""
    file_path = load_last_path()
    if not file_path:
        click.echo(" Please load a dataset first using `load <path>`")
        return
    loader = DataLoader(file_path)
    df = loader.load()
    cleaner = DataCleaner(df)
    df = cleaner.clean()
    imputer = DataImputer(df)
    df = imputer.impute()
    click.echo(" Missing values imputed successfully")

@cli.command()
def analyze():
    """Run full analysis and generate report (uses last loaded path)
    Prints ASCII tables and simple ASCII bar charts to console.
    """
    file_path = load_last_path()
    if not file_path:
        click.echo(" Please load a dataset first using `load <path>`")
        return

    loader = DataLoader(file_path)
    df = loader.load()
    cleaner = DataCleaner(df)
    df = cleaner.clean()
    imputer = DataImputer(df)
    df = imputer.impute()

    analyzer = Analyzer(df)
    results = analyzer.run_analysis()

    reporter = Reporter(results)
    try:
        reporter.generate()
    except Exception as e:
        logger.exception("Reporter.generate failed: %s", e)
        click.echo(" Report generation failed; see logs for details.")
        return

    # Prepare summary DataFrame
    try:
        if isinstance(results, pd.DataFrame):
            summary_df = results
        elif isinstance(results, dict):
            summary_df = pd.DataFrame({k: (v if not hasattr(v, "items") else list(v.values())) for k, v in results.items()})
        else:
            summary_df = pd.DataFrame(results)
    except Exception:
        summary_df = pd.DataFrame({"info": [str(results)]})

    # helper: simple ASCII bar
    def _bar(pct: float, width: int = 40) -> str:
        try:
            pctf = float(pct)
        except Exception:
            pctf = 0.0
        n = int(round((pctf / 100.0) * width))
        n = max(0, min(width, n))
        return "[" + "#" * n + " " * (width - n) + f"] {pctf:.1f}%"

    # Print a concise tabular summary using tabulate if available
    click.echo("\n--- Analysis summary (first rows) ---")
    try:
        to_print = summary_df.head(50)
        if tabulate:
            click.echo(tabulate(to_print, headers="keys", tablefmt="grid", showindex=True))
        else:
            click.echo(to_print.to_string(index=True))
    except Exception:
        click.echo("Could not render summary table; falling back to raw repr.")
        click.echo(repr(summary_df.head(20)))
    click.echo("--- end summary ---\n")

    # Print dataset-level ASCII diagnostics: top missing columns and simple hist stats
    try:
        miss = (df.isna().mean() * 100).sort_values(ascending=False).head(12)
        if not miss.empty:
            click.echo("Top missing columns (ASCII bars):")
            for col, pct in miss.items():
                click.echo(f"{col[:30]:30} {_bar(pct, width=36)}")
            click.echo("")
        # show basic numeric summary table
        num = df.select_dtypes(include=["number"]).describe().transpose()
        if not num.empty:
            click.echo("Numeric columns (summary):")
            # round and select common columns; handle different pandas versions
            cols = [c for c in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] if c in num.columns]
            if tabulate:
                click.echo(tabulate(num[cols].round(3), headers="keys", tablefmt="github"))
            else:
                click.echo(num[cols].round(3).to_string())
            click.echo("")
    except Exception:
        # non-fatal, continue
        pass

    click.echo(" Insights and report generated!")

@cli.command()
def schema():
    """Interactive schema proposal: ask quick questions and print/save a schema table."""
    file_path = load_last_path()
    if not file_path:
        click.echo(" Please load a dataset first using `load <path>`")
        return
    loader = DataLoader(file_path)
    df = loader.load()
    nrows = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        missing_pct = float(s.isna().mean())
        unique_count = int(s.nunique(dropna=True))
        inferred_type = (
            "datetime" if pd.api.types.is_datetime64_any_dtype(s) else
            "bool" if pd.api.types.is_bool_dtype(s) else
            "int" if pd.api.types.is_integer_dtype(s) else
            "float" if pd.api.types.is_float_dtype(s) else
            "category" if pd.api.types.is_categorical_dtype(s) else
            "object"
        )
        sample_values = s.dropna().unique()[:3].tolist()
        rows.append({
            "column": col,
            "inferred_type": inferred_type,
            "missing_pct": round(missing_pct, 4),
            "unique_count": unique_count,
            "sample_values": sample_values,
        })
    schema_df = pd.DataFrame(rows).sort_values(["missing_pct", "unique_count"], ascending=[True, False])
    click.echo("\nInferred schema (preview):")
    with pd.option_context("display.max_rows", 50, "display.max_columns", 8, "display.width", 120):
        click.echo(schema_df.head(200).to_string(index=False))
    pk_candidates = schema_df[schema_df["unique_count"] == nrows]["column"].tolist()
    if not pk_candidates:
        pk_candidates = schema_df[schema_df["unique_count"] >= max(1, int(nrows * 0.995))]["column"].tolist()
    if pk_candidates:
        click.echo("\nPrimary key candidates: " + ", ".join(pk_candidates[:10]))
        pk = click.prompt("Choose primary key from candidates or type 'none'", default="none")
    else:
        click.echo("\nNo obvious primary-key candidates found.")
        pk = "none"
    drop_threshold = click.prompt("Drop columns with missing ratio > (0-1)", default=0.9, type=float)
    click.echo("\nIf you want to override types, enter comma-separated column names (leave blank to accept inferred types).")
    cat_override = click.prompt("Treat these columns as categorical", default="", show_default=False)
    num_override = click.prompt("Treat these columns as numeric", default="", show_default=False)
    cat_set = {c.strip() for c in cat_override.split(",") if c.strip()}
    num_set = {c.strip() for c in num_override.split(",") if c.strip()}
    def user_type(row):
        c = row["column"]
        if c in cat_set:
            return "category"
        if c in num_set:
            return "numeric"
        if row["missing_pct"] > drop_threshold:
            return "drop"
        return row["inferred_type"]
    schema_df["user_type"] = schema_df.apply(user_type, axis=1)
    schema_df["primary_key"] = schema_df["column"].apply(lambda c: c == pk)
    click.echo("\nFinal proposed schema:")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 8, "display.width", 140):
        click.echo(schema_df[["column", "inferred_type", "user_type", "missing_pct", "unique_count", "primary_key", "sample_values"]].to_string(index=False))
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    schema_csv = os.path.join(out_dir, "proposed_schema.csv")
    schema_json = os.path.join(out_dir, "proposed_schema.json")
    schema_df.to_csv(schema_csv, index=False)
    schema_df.to_json(schema_json, orient="records", indent=2)
    click.echo(f"\nSchema saved: {schema_csv}  and  {schema_json}")
    click.echo("If okay, run `impute` or adjust overrides and re-run `cli.py schema`.")

@cli.command()
@click.argument("query", required=False)
@click.option("--json", "as_json", is_flag=True, default=False, help="Return JSON structured output.")
@click.option("--model", "-m", default=None, help="Local HF model name (overrides HF_MODEL env var).")
@click.option("--max-length", "-L", default=256, type=int, help="Max generation length.")
def llm(query, as_json, model, max_length):
    """Ask the local transformer LLM and perform actions (tables/plots/stats) on the last-loaded dataset."""
    if not _try_transformers:
        click.echo("Transformers not available. Install with: pip install transformers[torch]")
        return

    if not query:
        query = click.edit("# Write your query above. Save & close to send.\n")
        if not query:
            click.echo("No query provided.")
            return

    last_path = load_last_path()
    if not last_path or not os.path.exists(last_path):
        click.echo(" No dataset loaded. Run: main.py load <path> first.")
        return

    # load small preview and metadata for context
    try:
        df = pd.read_csv(last_path)
    except Exception as e:
        click.echo(f"Failed to read dataset: {e}")
        return

    cols = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in cols}
    missing = (df.isna().mean() * 100).round(2).sort_values(ascending=False).head(12).to_dict()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    preview = df.head(5)[cols[:12]].to_csv(index=False)  # limit width

    # build structured prompt asking model to return JSON actions
    prompt = f"""
You are a data-assistant. The user query is below. You MUST return JSON only (no other text).
Context:
- Columns: {cols}
- Dtypes: {dtypes}
- Top missing (pct): {missing}
- Numeric columns: {numeric_cols}
- Preview (first 5 rows CSV):
{preview}

User query:
{query}

Return JSON with:
{{
  "answer": "<short textual answer/summary>",
  "actions": [
    {{
      "type": "table" | "plot" | "stat",
      "columns": ["col1", "col2", ...],
      "n": <rows for table, optional>,
      // for plot:
      "plot_type": "hist" | "line" | "scatter" | "bar" | "box",
      // for stat, optional list of stats ["mean","median","count"...]
    }}
  ]
}}

Examples:
- If user asks 'show distribution of age and salary' -> return one action with type 'plot', plot_type 'hist', columns ['age','salary'].
- If user asks 'top 10 rows of hospital and test_name' -> return type 'table', columns ['hospital','test_name'], n=10.
Keep JSON concise and valid.
"""
    try:
        agent = TransformerAgent(model=model)
    except Exception as e:
        click.echo(f"Failed to initialize TransformerAgent: {e}")
        return

    parsed = agent.generate_json(prompt, max_length=max_length)
    if parsed is None:
        # fallback to plain text answer
        out = agent.generate(prompt, max_length=max_length)
        click.echo(out)
        return

    # Ensure outputs dir exists
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    results_out = {"answer": parsed.get("answer", ""), "tables": [], "plots": [], "stats": {}}

    actions = parsed.get("actions", [])
    if not isinstance(actions, list):
        actions = []

    for i, act in enumerate(actions):
        try:
            atype = act.get("type")
            cols_req = act.get("columns", [])
            # validate columns
            cols_ok = [c for c in cols_req if c in df.columns]
            if not cols_ok and cols_req:
                results_out.setdefault("warnings", []).append(f"Action {i}: none of requested columns found: {cols_req}")
                continue

            if atype == "table":
                n = int(act.get("n", 5))
                tbl = df[cols_ok].head(n)
                csv_text = tbl.to_csv(index=False)
                results_out["tables"].append({"columns": cols_ok, "n": n, "csv": csv_text})
            elif atype == "stat":
                stats_list = act.get("stats", ["count", "mean", "std", "min", "25%", "50%", "75%", "max"])
                # compute describe and pick requested stats when possible
                desc = df[cols_ok].describe().transpose().to_dict(orient="index")
                # filter to stats_list where available
                filtered = {}
                for c, vals in desc.items():
                    filtered[c] = {k: vals.get(k) for k in stats_list if k in vals}
                results_out["stats"].update(filtered)
            elif atype == "plot":
                plot_type = act.get("plot_type", "hist")
                filename = os.path.join(out_dir, f"llm_plot_{i}.png")
                try:
                    plt.clf()
                    if plot_type == "hist":
                        # if multiple numeric columns, draw histogram per column
                        for c in cols_ok:
                            if c in numeric_cols:
                                if sns:
                                    sns.histplot(df[c].dropna(), kde=False, label=c)
                                else:
                                    plt.hist(df[c].dropna(), alpha=0.6, bins=30, label=c)
                        plt.legend()
                        plt.title("Histogram: " + ", ".join(cols_ok))
                    elif plot_type == "line":
                        if len(cols_ok) >= 2:
                            x, y = cols_ok[0], cols_ok[1]
                            plt.plot(df[x], df[y], marker=".", linestyle="-")
                            plt.xlabel(x); plt.ylabel(y)
                        else:
                            plt.plot(df[cols_ok[0]])
                    elif plot_type == "scatter":
                        if len(cols_ok) >= 2:
                            x, y = cols_ok[0], cols_ok[1]
                            if sns:
                                sns.scatterplot(x=df[x], y=df[y])
                            else:
                                plt.scatter(df[x], df[y], alpha=0.6)
                            plt.xlabel(x); plt.ylabel(y)
                        else:
                            raise ValueError("scatter requires two columns")
                    elif plot_type == "box":
                        if sns:
                            sns.boxplot(data=df[cols_ok].dropna())
                        else:
                            df[cols_ok].plot.box()
                    elif plot_type == "bar":
                        # aggregate first column counts or mean of second if provided
                        if len(cols_ok) == 1:
                            vc = df[cols_ok[0]].value_counts().head(20)
                            vc.plot.bar()
                        elif len(cols_ok) >= 2:
                            agg = df.groupby(cols_ok[0])[cols_ok[1]].mean().nlargest(20)
                            agg.plot.bar()
                    else:
                        # unknown -> try simple plot of first column
                        plt.plot(df[cols_ok[0]].dropna())
                    plt.tight_layout()
                    plt.savefig(filename)
                    results_out["plots"].append({"file": filename, "plot_type": plot_type, "columns": cols_ok})
                    plt.clf()
                except Exception as e:
                    results_out.setdefault("warnings", []).append(f"plot {i} failed: {e}")
            else:
                results_out.setdefault("warnings", []).append(f"Unknown action type: {atype}")
        except Exception as e:
            results_out.setdefault("warnings", []).append(f"Action {i} error: {e}")

    # output: JSON with answer, tables (csv strings) and saved plot paths
    if as_json:
        click.echo(json.dumps(results_out, indent=2, default=str))
    else:
        # human-friendly print
        if results_out.get("answer"):
            click.echo("\nAnswer:\n" + results_out["answer"] + "\n")
        if results_out.get("tables"):
            click.echo("Tables:")
            for t in results_out["tables"]:
                click.echo(f"Columns: {t['columns']} (n={t['n']})")
                click.echo(t["csv"])
        if results_out.get("plots"):
            click.echo("Saved plots:")
            for p in results_out["plots"]:
                click.echo(f" - {p['file']}  ({p['plot_type']} on {p['columns']})")
        if results_out.get("stats"):
            click.echo("Stats:")
            click.echo(json.dumps(results_out["stats"], indent=2, default=str))
        if results_out.get("warnings"):
            click.echo("\nWarnings:")
            for w in results_out["warnings"]:
                click.echo(" - " + str(w))

if __name__ == "__main__":
    cli()