"""Read-only portfolio view over registered models and generated artifacts."""

import html
import json
import os
from pathlib import Path

from fastapi.responses import HTMLResponse


def _registry(models_dir: Path = Path("models")) -> dict:
    path = models_dir / "registry.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _metric(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def portfolio_page() -> HTMLResponse:
    registry = _registry()
    mlflow_url = html.escape(os.getenv("MLFLOW_PUBLIC_URL", "http://localhost:5000"), quote=True)
    cards = []
    for domain in ("cpu", "gpu"):
        champion = registry.get(domain)
        if not champion:
            cards.append(f"<article class='card'><h2>{domain.upper()}</h2><p>No champion registered yet.</p></article>")
            continue
        metrics = champion["metrics"]
        cards.append(f"""
        <article class='card'>
          <p class='eyebrow'>{domain.upper()} CHAMPION</p>
          <h2>{html.escape(champion['model_name'])}</h2>
            <div class='metrics'>
            <div><span>Held-out R²</span><strong>{_metric(metrics.get('R2_Score'))}</strong></div>
            <div><span>RMSE</span><strong>{_metric(metrics.get('RMSE'), 3)}</strong></div>
            <div><span>MAE</span><strong>{_metric(metrics.get('MAE'), 3)}</strong></div>
            <div><span>MAPE</span><strong>{_metric(metrics.get('MAPE_Percent'), 2)}%</strong></div>
          </div>
          <p class='detail'>Feature contract: {len(champion.get('feature_names', []))} transformed features<br>
          Updated: {html.escape(champion.get('updated_at', 'unknown'))}</p>
        </article>""")

    return HTMLResponse(f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Hardware Benchmarking MLOps</title><style>
:root {{ color-scheme: dark; --bg:#0b1020; --panel:#141b31; --line:#2a365a; --ink:#f4f7ff; --muted:#a9b6d5; --accent:#70e1c8; }}
* {{ box-sizing:border-box }} body {{ margin:0; font:16px/1.5 Inter,ui-sans-serif,system-ui,sans-serif; background:radial-gradient(circle at top right,#203565,var(--bg) 45%); color:var(--ink) }}
main {{ max-width:1080px; margin:auto; padding:64px 24px }} h1 {{ font-size:clamp(2.2rem,6vw,4.8rem); line-height:1; margin:.2rem 0 1rem }} h2 {{ margin:.2rem 0 1.2rem }} .eyebrow {{ color:var(--accent); font-weight:700; font-size:.77rem; letter-spacing:.12em }} .lead {{ max-width:720px; color:var(--muted); font-size:1.1rem }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:18px; margin:42px 0 }} .card,.flow {{ background:color-mix(in srgb,var(--panel) 92%,transparent); border:1px solid var(--line); border-radius:18px; padding:24px }} .metrics {{ display:grid; grid-template-columns:repeat(2,1fr); gap:10px }} .metrics span,.detail {{ color:var(--muted); font-size:.82rem }} .metrics strong {{ display:block; font-size:1.45rem }}
.flow {{ display:flex; flex-wrap:wrap; gap:9px; align-items:center }} .step {{ background:#202b4b; border-radius:999px; padding:7px 12px; font-size:.84rem }} .arrow {{ color:var(--accent) }} a {{ color:var(--accent) }} footer {{ margin-top:40px; color:var(--muted) }}
</style></head><body><main><p class='eyebrow'>CLASSICAL ML · PIPES & FILTERS · MLFLOW</p><h1>Hardware benchmarking,<br>made inspectable.</h1>
<p class='lead'>A reproducible MLOps workflow for CPU and GPU performance inference. Every candidate passes leakage-safe validation before a champion model is registered.</p>
<section class='grid'>{''.join(cards)}</section>
<section class='flow'><strong>Automated flow</strong><span class='arrow'>→</span>{''.join(f"<span class='step'>{step}</span>" for step in ['Validate','Explore','Clean','Impute + Feature Store','Cross-validate 11 models','Hold-out test','Promote'])}</section>
<footer>Explore experiment history in <a href='{mlflow_url}' target='_blank' rel='noreferrer'>MLflow</a> · API schema at <a href='/docs'>/docs</a> · Raw registry at <a href='/metrics'>/metrics</a></footer>
</main></body></html>""")
