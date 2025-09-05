import os
import json
import click
from loguru import logger
import pandas as pd
from typing import Optional, Any

from pipeline.data_loader import DataLoader
from pipeline.data_cleaner import DataCleaner
from pipeline.imputer import DataImputer
from pipeline.analyzer import Analyzer
from pipeline.reporter import Reporter

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
    """Run full analysis and generate report (uses last loaded path)"""
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
    try:
        if isinstance(results, pd.DataFrame):
            summary_df = results
        elif isinstance(results, dict):
            summary_df = pd.DataFrame({k: (v if not hasattr(v, "items") else list(v.values())) for k, v in results.items()})
        else:
            summary_df = pd.DataFrame(results)
        with pd.option_context("display.max_rows", 20, "display.max_columns", 20, "display.width", 120):
            click.echo("\n--- Analysis summary (first rows) ---")
            click.echo(summary_df.head(50).to_string(index=True))
            click.echo("--- end summary ---\n")
    except Exception:
        click.echo(" Report generated. Summary could not be rendered to table; check report files in outputs/ or logs/")
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
    """Ask the local transformer LLM (requires transformers)."""
    if not _try_transformers:
        click.echo("Transformers not available. Install with: pip install transformers[torch]")
        return
    if not query:
        query = click.edit("# Write your query above. Save & close to send.\n")
        if not query:
            click.echo("No query provided.")
            return
    ctx = ""
    last_path = load_last_path()
    if last_path and os.path.exists(last_path):
        try:
            preview_df = pd.read_csv(last_path, nrows=5)
            preview_csv = preview_df.to_csv(index=False)
            miss = (preview_df.isna().mean() * 100).round(2).sort_values(ascending=False)
            miss_txt = "\n".join([f"{c}: {p}%" for c, p in miss.items()])
            ctx = f"Dataset preview (first 5 rows CSV):\n{preview_csv}\nTop missing on preview:\n{miss_txt}\n\n"
        except Exception:
            ctx = ""
    if as_json:
        prompt = (
            "You are an assistant that returns ONLY valid JSON. "
            "JSON must include 'answer' (string) and may include 'table' (array/object).\n\n"
            f"{ctx}\nUser request:\n{query}\n\nReturn valid JSON only."
        )
    else:
        prompt = f"{ctx}\nUser request:\n{query}\n\nAnswer concisely."
    try:
        agent = TransformerAgent(model=model)
    except Exception as e:
        click.echo(f"Failed to initialize TransformerAgent: {e}")
        return
    if as_json:
        parsed = agent.generate_json(prompt, max_length=max_length)
        if parsed is None:
            out = agent.generate(prompt, max_length=max_length)
            click.echo(out)
        else:
            click.echo(json.dumps(parsed, indent=2))
    else:
        out = agent.generate(prompt, max_length=max_length)
        click.echo(out)

if __name__ == "__main__":
    cli()
