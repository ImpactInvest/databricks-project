"""Draws docs/images/architecture.png. Run: python docs/images/make_architecture.py"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, EDGE = "#1f2933", "#52606d", "#9aa5b1"
FILL = {"data": "#e8f1fb", "agent": "#fdf3e1", "serve": "#e7f6ec", "eval": "#f1ecfa"}

fig, ax = plt.subplots(figsize=(14, 8.2), dpi=160)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8.2)
ax.axis("off")


def box(x, y, w, h, title, sub="", kind="data"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=FILL[kind], ec=EDGE, lw=1.2
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2 + (0.14 if sub else 0),
        title,
        ha="center",
        va="center",
        fontsize=10.5,
        color=INK,
        weight="bold",
    )
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=8.3, color=MUTED)


def arrow(x1, y1, x2, y2, label="", curve=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.2,
            color=MUTED,
            connectionstyle=f"arc3,rad={curve}",
        )
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", fontsize=8, color=MUTED)


# Unity Catalog frame
ax.add_patch(
    FancyBboxPatch(
        (2.3, 5.25),
        11.4,
        2.65,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        fc="none",
        ec="#7b8794",
        lw=1,
        ls="--",
    )
)
ax.text(2.45, 7.7, "Unity Catalog: workspace.genai_agent", fontsize=9, color=MUTED, style="italic")

box(0.2, 6.1, 1.8, 1.0, "samples", "bakehouse", "data")
box(2.6, 6.1, 2.5, 1.0, "01 PySpark ingest", "clean, derive, drop PII", "data")
box(5.5, 6.55, 3.1, 0.95, "silver_* Delta tables", "sales, franchises, customers, suppliers", "data")
box(5.5, 5.45, 3.1, 0.9, "reviews_chunked", "Delta, Change Data Feed on", "data")
box(9.1, 5.45, 2.7, 0.9, "AI Search index", "Delta Sync, gte-large-en", "data")
box(9.1, 6.55, 2.7, 0.95, "Comments", "table + column metadata", "data")
box(12.1, 5.9, 1.45, 1.2, "Genie", "Agent", "eval")

arrow(2.0, 6.6, 2.6, 6.6)
arrow(5.1, 6.8, 5.5, 7.0)
arrow(5.1, 6.4, 5.5, 5.9)
arrow(8.6, 5.9, 9.1, 5.9, "sync")
arrow(8.6, 7.0, 9.1, 7.0)
arrow(11.8, 7.0, 12.1, 6.7)

# Agent
box(
    2.6,
    2.9,
    3.3,
    1.5,
    "query_sales_data",
    "Text2SQL -> sql_guard -> run_sql\nread-only, allowlist, LIMIT, timeout",
    "agent",
)
box(6.4, 2.9, 3.0, 1.5, "Tool-calling agent", "MLflow ResponsesAgent\nmax 6 tool calls", "agent")
box(9.9, 2.9, 3.0, 1.5, "search_reviews", "top-k + franchise filter\nRETRIEVER span, review ids", "agent")
box(6.4, 1.3, 3.0, 0.9, "Foundation Model API", "LLM endpoint (pay per token)", "agent")

arrow(6.4, 3.65, 5.9, 3.65)
arrow(9.4, 3.65, 9.9, 3.65)
arrow(7.9, 2.9, 7.9, 2.2)
arrow(3.6, 4.4, 5.5, 6.85, curve=-0.2)
ax.text(2.2, 4.62, "SQL on warehouse\n(Spark in notebooks)", fontsize=8, color=MUTED, ha="center")
arrow(11.4, 4.4, 10.45, 5.45)

# Serving and evaluation
box(0.2, 0.25, 3.2, 1.25, "MLflow 3", "tracing, golden-set eval,\nLLM judges, DSPy, UC registry", "eval")
box(10.3, 0.25, 1.9, 1.0, "Model Serving", "CPU endpoint", "serve")
box(12.4, 0.25, 1.4, 1.0, "Databricks", "App (UI)", "serve")
arrow(9.4, 2.9, 10.6, 1.25, "", curve=0.2)
ax.text(11.35, 2.15, "log, register\nand deploy", fontsize=8, color=MUTED, ha="left")
arrow(12.2, 0.75, 12.4, 0.75)
arrow(6.4, 3.2, 3.4, 1.2, "traces", curve=0.1)

ax.text(
    7.0, 8.05, "Databricks GenAI agent with evaluation", ha="center", fontsize=13, color=INK, weight="bold"
)
out = Path(__file__).with_name("architecture.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
