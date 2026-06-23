"""Assemble Table 2 (main results matrix) from existing phase summaries.

Outputs:
  results/cea/table2_matrix.json
  results/cea/table2_main.tex      — condensed table for main paper (agricultural datasets)
  results/cea/table2_appendix.tex — full matrix incl. LoveDA OOD + extra methods
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("results/cea")
OUT_JSON = ROOT / "table2_matrix.json"
OUT_MAIN = ROOT / "table2_main.tex"
OUT_APPX = ROOT / "table2_appendix.tex"

ARCH = "segformer_b0_imagenet"
METHOD_LABEL = {
    "bicubic": "Bicubic",
    "semantic_inr": "Ours",
    "inr_no_semantic": "INR w/o sem.",
    "tiny_rescnn": "Tiny ResCNN",
    "uniform_sharp": "Unif. sharp",
}

DEGRAD_LABEL = {
    "all": "all",
    "structure": "struct.",
    "composite": "comp.",
    "struct+comp": "s+c",
    "structure,composite": "s+c",
}

DATASET_LABEL = {
    "WeedsGalore": "WeedsGalore",
    "CWFID": "CWFID",
    "CoFly-WeedDB": "CoFly",
    "LoveDA-Rural (OOD)": "LoveDA-R",
    "LoveDA-Urban (OOD)": "LoveDA-U",
    "WeedsGalore (baselines)": "WG (base.)",
}


def _degrad(d: str) -> str:
    return DEGRAD_LABEL.get(d, d.replace("structure", "struct.").replace("composite", "comp."))


def _dataset(d: str) -> str:
    return DATASET_LABEL.get(d, d)


def _load(name: str) -> dict:
    return json.loads((ROOT / name / "summary.json").read_text(encoding="utf-8"))


def _f(x, nd=3):
    return f"{float(x):.{nd}f}"


def _fd(x, nd=3, sign=True):
    v = float(x)
    if not sign:
        return _f(v, nd)
    return f"{v:+.{nd}f}"


def _row(dataset, degrad, method, psnr=None, ssim=None, bf=None, miou=None,
         dpsnr=None, dssim=None, dbf=None, dmiou=None, n=None):
    return {
        "dataset": dataset, "degradation": degrad, "method": method,
        "psnr": psnr, "ssim": ssim, "boundary_f": bf, "frozen_miou": miou,
        "delta_psnr": dpsnr, "delta_ssim": dssim, "delta_boundary_f": dbf,
        "delta_miou": dmiou, "n": n,
    }


def weedsgalore_rows() -> list[dict]:
    s = _load("phase2_main")
    means = s["means"][ARCH]
    delta = s["delta_vs_bicubic"][ARCH]["semantic_inr"]
    rows = []
    # aggregate (absolute)
    rows.append(_row("WeedsGalore", "all", "Bicubic",
                     psnr=means["bicubic"]["psnr"], ssim=means["bicubic"]["ssim"],
                     bf=means["bicubic"]["boundary_f"], miou=means["bicubic"]["deep_frozen_miou"],
                     n=delta["deep_frozen_miou"]["n"]))
    rows.append(_row("WeedsGalore", "all", "Ours",
                     psnr=means["semantic_inr"]["psnr"], ssim=means["semantic_inr"]["ssim"],
                     bf=means["semantic_inr"]["boundary_f"], miou=means["semantic_inr"]["deep_frozen_miou"],
                     dpsnr=delta["psnr"]["mean_delta"], dssim=delta["ssim"]["mean_delta"],
                     dbf=delta["boundary_f"]["mean_delta"], dmiou=delta["deep_frozen_miou"]["mean_delta"],
                     n=delta["deep_frozen_miou"]["n"]))
    for plan in ("structure", "composite", "veil"):
        pe = delta["deep_miou_by_plan"][plan]
        rows.append(_row("WeedsGalore", plan, "",
                         dmiou=pe["mean_delta"], n=pe["n"]))
    return rows


def loveda_rows(tag: str, label: str) -> list[dict]:
    s = _load(tag)
    means = s["means"][ARCH]
    delta = s["delta_vs_bicubic"][ARCH]["semantic_inr"]
    rows = [
        _row(label, "struct+comp", "Bicubic",
             psnr=means["bicubic"]["psnr"], ssim=means["bicubic"]["ssim"],
             bf=means["bicubic"]["boundary_f"], miou=means["bicubic"]["deep_frozen_miou"],
             n=delta["deep_frozen_miou"]["n"]),
        _row(label, "struct+comp", "Ours",
             psnr=means["semantic_inr"]["psnr"], ssim=means["semantic_inr"]["ssim"],
             bf=means["semantic_inr"]["boundary_f"], miou=means["semantic_inr"]["deep_frozen_miou"],
             dpsnr=delta["psnr"]["mean_delta"], dssim=delta["ssim"]["mean_delta"],
             dbf=delta["boundary_f"]["mean_delta"], dmiou=delta["deep_frozen_miou"]["mean_delta"],
             n=delta["deep_frozen_miou"]["n"]),
    ]
    for plan in ("structure", "composite"):
        if plan in delta.get("deep_miou_by_plan", {}):
            pe = delta["deep_miou_by_plan"][plan]
            rows.append(_row(label, plan, "", dmiou=pe["mean_delta"], n=pe["n"]))
    return rows


def replication_rows(tag: str, label: str) -> list[dict]:
    s = _load(tag)
    means = s["means"]
    delta = s["delta_vs_bicubic"]["semantic_inr"]
    key = f"{ARCH}:all"
    pe = delta[key]
    rows = [
        _row(label, "struct+comp", "Bicubic",
             psnr=means["bicubic"]["psnr"], ssim=means["bicubic"]["ssim"],
             miou=means["bicubic"]["miou_segformer_b0_imagenet"]),
        _row(label, "struct+comp", "Ours",
             psnr=means["semantic_inr"]["psnr"], ssim=means["semantic_inr"]["ssim"],
             miou=means["semantic_inr"]["miou_segformer_b0_imagenet"],
             dmiou=pe["mean_delta"]),
    ]
    for plan in ("structure", "composite"):
        pk = f"{ARCH}:{plan}"
        if pk in delta:
            rows.append(_row(label, plan, "", dmiou=delta[pk]["mean_delta"]))
    return rows


def appendix_method_rows(dataset_tag: str, label: str, s: dict) -> list[dict]:
    """phase3-style summary with nested arch -> method."""
    rows = []
    if "means" in s and ARCH in s["means"]:
        means = s["means"][ARCH]
        for method, m in means.items():
            rows.append(_row(label, s.get("plans", "struct+comp"), METHOD_LABEL.get(method, method),
                             psnr=m.get("psnr"), ssim=m.get("ssim"),
                             bf=m.get("boundary_f"), miou=m.get("deep_frozen_miou")))
        delta_block = s.get("delta_vs_bicubic", {}).get(ARCH, {})
        for method, d in delta_block.items():
            if method == "bicubic":
                continue
            for plan, pe in d.get("deep_miou_by_plan", {}).items():
                rows.append(_row(label, plan, METHOD_LABEL.get(method, method),
                                 dmiou=pe["mean_delta"], n=pe.get("n")))
    return rows


def _format_row(r: dict, use_multirow: bool, group_len: int | None, show_dataset: bool) -> str:
    psnr = _f(r["psnr"]) if r.get("psnr") is not None else "---"
    ssim = _f(r["ssim"]) if r.get("ssim") is not None else "---"
    bf = _f(r["boundary_f"]) if r.get("boundary_f") is not None else "---"
    miou = _f(r["frozen_miou"]) if r.get("frozen_miou") is not None else "---"
    dmiou = _fd(r["delta_miou"]) if r.get("delta_miou") is not None else "---"
    ds = _dataset(r["dataset"])
    deg = _degrad(r["degradation"])
    method = r["method"] if r["method"] else "$\\Delta$ vs bic."
    if show_dataset and use_multirow and group_len and group_len > 1:
        ds_cell = f"\\multirow{{{group_len}}}{{*}}{{{ds}}}"
    elif show_dataset:
        ds_cell = ds
    else:
        ds_cell = ""
    return f"{ds_cell} & {deg} & {method} & {psnr} & {ssim} & {bf} & {miou} & {dmiou}"


def _grouped_rows(rows: list[dict]) -> list[tuple[dict, bool, int | None, bool]]:
    """Yield (row, use_multirow, group_len, show_dataset) for consecutive dataset groups."""
    out: list[tuple[dict, bool, int | None, bool]] = []
    i = 0
    while i < len(rows):
        ds = rows[i]["dataset"]
        j = i + 1
        while j < len(rows) and rows[j]["dataset"] == ds:
            j += 1
        glen = j - i
        for k in range(i, j):
            out.append((rows[k], glen > 1, glen if k == i else None, k == i))
        i = j
    return out


def main() -> None:
    main_rows = (
        weedsgalore_rows()
        + replication_rows("second_dataset", "CWFID")
        + replication_rows("dataset_cofly", "CoFly-WeedDB")
    )
    appx_rows = main_rows + loveda_rows("phase5_loveda_rural", "LoveDA-Rural (OOD)") + loveda_rows(
        "phase5_loveda_urban", "LoveDA-Urban (OOD)"
    )
    appx_rows += appendix_method_rows("phase3_baselines", "WeedsGalore (baselines)", _load("phase3_baselines"))

    OUT_JSON.write_text(json.dumps({"main": main_rows, "appendix": appx_rows}, indent=2), encoding="utf-8")

    header = "Data & Reg. & Method & PSNR & SSIM & BF & mIoU & $\\Delta$mIoU \\\\"

    def build_table(rows: list[dict], caption: str, label: str, compact: bool) -> str:
        grouped = _grouped_rows(rows)
        body = [_format_row(r, use_mr, glen, show_ds) for r, use_mr, glen, show_ds in grouped]
        lines = [
            "\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        ]
        if not compact:
            lines.append("\\scriptsize")
        lines.append("\\tighttable{%")
        lines.append("\\begin{tabular}{@{}lll rrrr r@{}}")
        lines.append("\\toprule")
        lines.append(header)
        lines.append("\\midrule")
        lines.extend(line + " \\\\" for line in body)
        lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""]
        return "\n".join(lines)

    OUT_MAIN.write_text(build_table(
        main_rows,
        "Q2: Main agricultural results (frozen SegFormer-B0). Aggregate rows report absolute PSNR/SSIM/BF/mIoU; "
        "regime sub-rows report $\\Delta$mIoU vs bicubic only (WeedsGalore: struct./comp./veil). BF = Boundary-F.",
        "tab:mainresults", True,
    ), encoding="utf-8")

    OUT_APPX.write_text(build_table(
        appx_rows,
        "Full results matrix (LoveDA OOD scope and WeedsGalore baselines; frozen SegFormer-B0). "
        "Regime sub-rows: $\\Delta$mIoU only.",
        "tab:fullmatrix", False,
    ), encoding="utf-8")

    print(f"wrote {OUT_JSON} ({len(main_rows)} main, {len(appx_rows)} appendix rows)")


if __name__ == "__main__":
    main()
