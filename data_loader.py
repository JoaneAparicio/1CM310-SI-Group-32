"""
data_loader.py — SDG Shopfloor Simulation
==========================================
Loads work orders and item routings from data.xlsx and returns clean structures
ready for the simulation engine.
"""

import pandas as pd
from pathlib import Path


def load_data(data_file: Path) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    """
    Load and validate the simulation input data.

    Parameters
    ----------
    data_file : Path
        Path to data.xlsx (must contain sheets 'Item_Routing' and 'WOs').

    Returns
    -------
    df_wo : pd.DataFrame
        Work orders with columns: wo_num, item_name, start_date, due_date, ord_qty.
    routing_dict : dict[str, list[dict]]
        Maps item_name → sorted list of operation dicts
        (keys: item_name, operation, setup_days, run_days,
               outside_process_fix_lt, wc, leadtime, planned_qtime).
    """
    print("Loading data…")

    # ── Routings ──────────────────────────────────────────────────────────────
    df_routing = pd.read_excel(data_file, sheet_name="Item_Routing", header=0)
    df_routing.columns = [
        "item_name", "operation", "setup_days", "run_days",
        "outside_process_fix_lt", "wc", "leadtime", "planned_qtime",
    ]
    df_routing = df_routing.dropna(subset=["item_name"])
    df_routing["setup_days"]    = pd.to_numeric(df_routing["setup_days"],    errors="coerce").fillna(0)
    df_routing["run_days"]      = pd.to_numeric(df_routing["run_days"],      errors="coerce").fillna(0)
    df_routing["planned_qtime"] = pd.to_numeric(df_routing["planned_qtime"], errors="coerce").fillna(0)
    df_routing["operation"]     = pd.to_numeric(df_routing["operation"],     errors="coerce")
    df_routing = df_routing.dropna(subset=["operation"]).copy()

    # Build routing dict: item_name → list of op dicts sorted by operation number
    routing_dict: dict[str, list[dict]] = {}
    for item, grp in df_routing.groupby("item_name"):
        routing_dict[item] = grp.sort_values("operation").to_dict("records")

    # ── Work orders ───────────────────────────────────────────────────────────
    df_wo = pd.read_excel(data_file, sheet_name="WOs", header=0)
    df_wo.columns = ["wo_num", "item_name", "start_date", "due_date", "ord_qty"]
    df_wo = df_wo.dropna(subset=["wo_num"])
    df_wo["ord_qty"]    = pd.to_numeric(df_wo["ord_qty"], errors="coerce").fillna(1).astype(int)
    df_wo["start_date"] = pd.to_datetime(df_wo["start_date"])
    df_wo["due_date"]   = pd.to_datetime(df_wo["due_date"])

    # ── Summary ───────────────────────────────────────────────────────────────
    items_no_routing = df_wo[~df_wo["item_name"].isin(routing_dict)]["item_name"].unique()
    print(f"Items with routing : {len(routing_dict)}")
    print(f"Work orders        : {len(df_wo)}")
    print(f"WOs without routing: {len(items_no_routing)} items → skipped")

    return df_wo, routing_dict
