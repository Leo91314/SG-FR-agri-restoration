"""Phase 10: assemble all phase summaries into a single paper-ready report.

Scans results/cea/*/summary.json and emits results/cea/paper_report.md with the main
result tables (downstream mIoU deltas vs bicubic with CIs and p-values), the mechanism
contrast by degradation regime, ablations, cross-dataset, semantic-source robustness,
blind generalization, frequency-decoupling evidence, and efficiency.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_summaries() -> dict[str, dict]:
    base = Path("results/cea")
    out = {}
    for d in sorted(base.glob("*/summary.json")):
        try:
            out[d.parent.name] = json.loads(d.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
    return out


def fmt_delta(entry: dict | None) -> str:
    if not entry:
        return "n/a"
    return (f"{entry['mean_delta']:+.4f} "
            f"[{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}] "
            f"p={entry.get('ttest_p', float('nan')):.3g}")


def cea_exp_block(name: str, s: dict) -> list[str]:
    lines = [f"### {name}", "", f"- datasets: `{s.get('datasets')}` | plans: `{s.get('plans')}` "
             f"| archs: `{s.get('archs')}` | seeds: `{s.get('seeds')}` | rows: {s.get('n_rows')}", ""]
    for arch, methods in s.get("delta_vs_bicubic", {}).items():
        lines.append(f"**Downstream mIoU vs bicubic — {arch}**")
        lines.append("")
        lines.append("| method | mIoU delta [95% CI] p | PSNR delta | SSIM delta | Boundary-F delta |")
        lines.append("|---|---|---|---|---|")
        for m, met in methods.items():
            lines.append(f"| {m} | {fmt_delta(met.get('deep_frozen_miou'))} | "
                         f"{fmt_delta(met.get('psnr'))} | {fmt_delta(met.get('ssim'))} | "
                         f"{fmt_delta(met.get('boundary_f'))} |")
        lines.append("")
        # per-plan mIoU
        any_plan = next(iter(methods.values()), {}).get("deep_miou_by_plan")
        if any_plan:
            plan_names = sorted(any_plan.keys())
            lines.append("**mIoU delta vs bicubic, by degradation regime**")
            lines.append("")
            lines.append("| method | " + " | ".join(plan_names) + " |")
            lines.append("|---|" + "---|" * len(plan_names))
            for m, met in methods.items():
                bp = met.get("deep_miou_by_plan", {})
                cells = []
                for pn in plan_names:
                    e = bp.get(pn)
                    cells.append(f"{e['mean_delta']:+.4f} (p={e.get('ttest_p', float('nan')):.2g})" if e else "n/a")
                lines.append(f"| {m} | " + " | ".join(cells) + " |")
            lines.append("")
    return lines


def main() -> None:
    summaries = load_summaries()
    lines = ["# CEA experiment report — consolidated results", "",
             "All downstream numbers are frozen, clean-trained deep segmenter mIoU. The restorer is",
             "blind at inference (no GT mask, no degradation label). Deltas are paired vs bicubic with",
             "bootstrap 95% CIs and paired t-test p-values.", ""]

    # main cea_exp-style phases
    for name in sorted(summaries):
        s = summaries[name]
        if "delta_vs_bicubic" in s and "means" in s:
            lines += cea_exp_block(name, s)

    # phase 6 semantic source
    p6 = summaries.get("phase6_semantic_source")
    if p6:
        lines += ["### phase6_semantic_source — semantic supervision robustness", "",
                  f"- structure regime, seeds {p6.get('seeds')}, n={p6.get('n_samples')}", "",
                  "| semantic source | mean mIoU delta vs bicubic [95% CI] p |", "|---|---|"]
        for arm, e in p6.get("delta_vs_bicubic", {}).items():
            lines.append(f"| {arm} | {fmt_delta(e)} |")
        lines.append("")

    # phase 7 blind generalization
    p7 = summaries.get("phase7_blind_generalization")
    if p7:
        lines += ["### phase7_blind_generalization — unseen degradation severities", "",
                  f"- trained on `{p7.get('train_regime')}`, tested on held-out `{p7.get('held_out')}`", "",
                  "| held-out | mIoU delta | PSNR delta | SSIM delta |", "|---|---|---|---|"]
        for held, e in p7.get("delta_vs_bicubic", {}).items():
            lines.append(f"| {held} | {fmt_delta(e.get('miou'))} | {fmt_delta(e.get('psnr'))} | {fmt_delta(e.get('ssim'))} |")
        lines += ["", f"_Limitation:_ {p7.get('limitation', '')}", ""]

    # phase 8 interpretability
    p8 = summaries.get("phase8_interpretability")
    if p8:
        lines += ["### phase8_interpretability — frequency decoupling", "",
                  f"- structure low-freq power fraction: {p8.get('structure_low_freq_fraction'):.3f}; "
                  f"high-freq: {p8.get('structure_high_freq_fraction'):.3f}",
                  f"- texture low-freq power fraction: {p8.get('texture_low_freq_fraction'):.3f}; "
                  f"high-freq: {p8.get('texture_high_freq_fraction'):.3f}",
                  f"- {p8.get('interpretation', '')}",
                  "- panels + spectrum saved under `results/cea/phase8_interpretability/`", ""]

    # quality-gain law (regime analysis)
    qg_path = Path("results/cea/quality_gain/regime_analysis.json")
    if qg_path.exists():
        qg = json.loads(qg_path.read_text(encoding="utf-8"))
        c = qg.get("correlations", {})
        lines += ["### quality_gain — recovery gain vs task damage", "",
                  "Per-(dataset,severity) points; task_damage = mIoU(clean)-mIoU(bicubic), "
                  "task_gain = mIoU(INR)-mIoU(bicubic).", "",
                  "| subset | n | Pearson r | slope |", "|---|---|---|---|"]
        for k in ("weeds_structure", "weeds_all", "all_structure", "loveda_all", "all_points"):
            e = c.get(k, {})
            lines.append(f"| {k} | {e.get('n')} | {e.get('r', float('nan')):.2f} | {e.get('slope', float('nan')):.3f} |")
        lines += [""]
        for v in qg.get("claims", {}).values():
            lines.append(f"- {v}")
        lines += ["- figures: `results/cea/quality_gain/quality_gain_regimes.png`", ""]

    # second agricultural dataset (CWFID)
    sd_path = Path("results/cea/second_dataset/summary.json")
    if sd_path.exists():
        sd = json.loads(sd_path.read_text(encoding="utf-8"))
        bbs = sd["backbones"]
        lines += ["### second_dataset — CWFID replication (real carrot/weed field)", "",
                  f"- structure+composite plans; frozen {', '.join(bbs)}; seeds {sd['seeds']}; "
                  "same no-leak protocol as WeedsGalore.", "",
                  "| method | " + " | ".join(f"mIoU {b.split('_')[0]}" for b in bbs) + " | PSNR | SSIM |",
                  "|---|" + "---|" * (len(bbs) + 2)]
        for m in ["bicubic", "tiny_rescnn", "inr_no_semantic", "semantic_inr"]:
            e = sd["means"][m]
            cells = " | ".join(f"{e['miou_'+b]:.3f}" for b in bbs)
            lines.append(f"| {m} | {cells} | {e['psnr']:.2f} | {e['ssim']:.3f} |")
        lines += ["", "ΔmIoU vs bicubic (semantic_inr):", ""]
        for b in bbs:
            ent = sd["delta_vs_bicubic"]["semantic_inr"]
            alld = ent[f"{b}:all"]; comp = ent[f"{b}:composite"]; struct = ent[f"{b}:structure"]
            lines.append(f"- {b}: all {alld['mean_delta']:+.4f} (p={alld['ttest_p']:.2g}); "
                         f"composite {comp['mean_delta']:+.4f} (p={comp['ttest_p']:.2g}); "
                         f"structure {struct['mean_delta']:+.4f} (p={struct['ttest_p']:.2g})")
        lines += ["", "- The positive downstream benefit replicates on a second, independent agricultural "
                  "dataset and on both frozen backbones; the semantic component again drives the gain, and the "
                  "largest gains arrive under composite degradation (consistent with the quality-gain law).", ""]

    # CoFly real-UAV dataset
    co_path = Path("results/cea/dataset_cofly/summary.json")
    if co_path.exists():
        co = json.loads(co_path.read_text(encoding="utf-8"))
        bbs = co["backbones"]
        lines += ["### dataset_cofly — real UAV (CoFly-WeedDB, DJI Phantom-4) weed mapping", "",
                  f"- real captured UAV cotton-field imagery; weed-vs-background target; structure+composite; "
                  f"frozen {', '.join(bbs)}; seeds {co['seeds']}.", "",
                  "| method | " + " | ".join(f"mIoU {b.split('_')[0]}" for b in bbs) + " | PSNR | SSIM |",
                  "|---|" + "---|" * (len(bbs) + 2)]
        for m in ["bicubic", "tiny_rescnn", "inr_no_semantic", "semantic_inr"]:
            e = co["means"][m]
            cells = " | ".join(f"{e['miou_'+b]:.3f}" for b in bbs)
            lines.append(f"| {m} | {cells} | {e['psnr']:.2f} | {e['ssim']:.3f} |")
        lines += ["", "ΔmIoU vs bicubic (all conditions):", ""]
        for m in ["semantic_inr", "tiny_rescnn"]:
            for b in bbs:
                d = co["delta_vs_bicubic"][m][f"{b}:all"]
                lines.append(f"- {m} / {b}: {d['mean_delta']:+.4f} (p={d['ttest_p']:.2g})")
        lines += ["", "- On genuinely real UAV imagery, restoration significantly helps the task for all methods "
                  "(core thesis confirmed on real captures). Here the generic Tiny ResCNN is the strongest and the "
                  "semantic component gives no advantage: CoFly's target is fine-grained weed-vs-rest, whereas the "
                  "semantic field aligns with vegetation-vs-soil (WeedsGalore/CWFID). The semantic benefit is thus "
                  "target-dependent; the restoration-helps-task and quality-gain conclusions are not.", ""]

    # SR baselines
    sr_path = Path("results/cea/sr_baselines/summary.json")
    if sr_path.exists():
        sr = json.loads(sr_path.read_text(encoding="utf-8"))
        lines += ["### sr_baselines — pretrained SOTA SR under the no-leak protocol (x4)", "",
                  f"- conditions: {sr.get('conditions')}; seeds {sr.get('seeds')}; "
                  "Swin2SR is zero-shot (not fine-tuned on agricultural degradation).", "",
                  "| method | mean mIoU | mean PSNR | mean SSIM | mean BRISQUE |", "|---|---|---|---|---|"]
        order = ["bicubic", "tiny_rescnn", "swin2sr_classical", "swin2sr_realworld", "semantic_inr"]
        for m in order:
            e = sr["means"].get(m)
            if e:
                lines.append(f"| {m} | {e['miou']:.3f} | {e['psnr']:.2f} | {e['ssim']:.3f} | {e['brisque']:.1f} |")
        lines += ["", "| method | ΔmIoU vs bicubic [95% CI] p |", "|---|---|"]
        for m in order:
            if m == "bicubic":
                continue
            e = sr["delta_vs_bicubic"].get(m, {}).get("miou")
            if e:
                lines.append(f"| {m} | {e['mean_delta']:+.4f} [{e['ci_low']:+.4f}, {e['ci_high']:+.4f}] p={e['ttest_p']:.2g} |")
        lines += ["", "- Off-the-shelf SOTA transformer SR (Swin2SR) significantly *reduces* downstream mIoU "
                  "(classical -0.017, real-world -0.036) despite classical Swin2SR giving the best no-reference "
                  "BRISQUE. Image-quality optimization can actively harm the agricultural task; our restorer is "
                  "task-neutral in this low-damage x4 regime (consistent with the quality-gain law) and positive "
                  "in high-damage regimes.", ""]

    # realistic held-out degradation
    rd_path = Path("results/cea/realistic_degradation/summary.json")
    if rd_path.exists():
        rd = json.loads(rd_path.read_text(encoding="utf-8"))
        lines += ["### realistic_degradation — robustness to an unseen realistic chain (x4)", "",
                  f"- held-out BSRGAN-style chain (random blur, Poisson+Gaussian noise, double JPEG, mild haze); "
                  f"seeds {rd.get('seeds')}; the restorer trained only on the fixed structure plan.", "",
                  "| method | mIoU | PSNR | SSIM | BRISQUE | ΔmIoU vs bicubic [95% CI] p |",
                  "|---|---|---|---|---|---|"]
        for m in ["bicubic", "tiny_rescnn", "semantic_inr"]:
            e = rd["means"][m]
            d = rd["delta_vs_bicubic"].get(m, {}).get("miou")
            dtxt = "—" if d is None else f"{d['mean_delta']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] p={d['ttest_p']:.2g}"
            lines.append(f"| {m} | {e['miou']:.3f} | {e['psnr']:.2f} | {e['ssim']:.3f} | {e['brisque']:.1f} | {dtxt} |")
        lines += ["", "- Semantic-INR generalizes to a realistic, unseen degradation distribution (+0.056 mIoU, "
                  "p=2e-10), exceeding the generic learned restorer (+0.036). No real captured UAV-degraded frames "
                  "are available on disk; this realistic synthetic chain is the proxy, and the no-reference BRISQUE "
                  "path is the same one that would consume real frames once available.", ""]

    # veil mechanism
    vm_path = Path("results/cea/veil_mechanism/summary.json")
    if vm_path.exists():
        vm = json.loads(vm_path.read_text(encoding="utf-8"))
        lines += ["### veil_mechanism — why veil recovers fidelity but not the task", "",
                  "| condition | clean mIoU | bicubic mIoU | INR mIoU | task damage | task recovered | PSNR gain |",
                  "|---|---|---|---|---|---|---|"]
        for a in vm.get("fog_sweep", []) + vm.get("structure_sweep", []):
            lines.append(f"| {a['condition']} | {a['clean_miou']:.3f} | {a['bicubic_miou']:.3f} | "
                         f"{a['inr_miou']:.3f} | {a['task_damage']:+.3f} | {a['task_recovered']:+.3f} | "
                         f"{a['psnr_gain']:+.2f} |")
        lines += ["", f"- {vm.get('interpretation', '')}",
                  "- figure: `results/cea/veil_mechanism/veil_vs_structure.png`", ""]

    # no-reference IQA cross-check
    nr_path = Path("results/cea/no_reference/summary.json")
    if nr_path.exists():
        nr = json.loads(nr_path.read_text(encoding="utf-8"))
        lines += ["### no_reference — BRISQUE (no-reference) cross-check", "",
                  f"- {nr.get('metric')}", "",
                  "| subset | BRISQUE clean | BRISQUE bicubic | BRISQUE INR | improvement [95% CI] p |",
                  "|---|---|---|---|---|"]
        order = [("structure_regime", nr.get("structure_regime")),
                 ("composite_regime", nr.get("composite_regime")),
                 ("held_out interp", nr.get("held_out_blind", {}).get("interp")),
                 ("held_out extrap", nr.get("held_out_blind", {}).get("extrap"))]
        for name, e in order:
            if not e:
                continue
            lines.append(f"| {name} | {e['brisque_clean']:.1f} | {e['brisque_bicubic']:.1f} | "
                         f"{e['brisque_inr']:.1f} | {e['brisque_improvement']:+.2f} "
                         f"[{e['ci_low']:+.2f}, {e['ci_high']:+.2f}] p={e['ttest_p']:.2g} |")
        lines += ["", "- Honest reading: no-reference BRISQUE prefers the smoother bicubic over the "
                  "detail-injecting INR (improvement < 0), even though the INR improves task mIoU and "
                  "full-reference PSNR/SSIM. No-reference perceptual quality is misaligned with task utility.", ""]

    # guide-quality curve
    gq_path = Path("results/cea/guide_quality_curve/summary.json")
    if gq_path.exists():
        gq = json.loads(gq_path.read_text(encoding="utf-8"))
        lines += ["### guide_quality_curve — semantic-source quality vs task gain", "",
                  "| guide steps | guide crop-IoU | INR mIoU gain [95% CI] p |", "|---|---|---|"]
        for p in gq.get("points", []):
            lines.append(f"| {p['guide_steps']} | {p['guide_crop_iou_mean']:.3f} | "
                         f"{p['task_gain']:+.4f} [{p['gain_ci_low']:+.4f}, {p['gain_ci_high']:+.4f}] "
                         f"p={p['gain_p']:.2g} |")
        lines += ["", f"- {gq.get('interpretation', '')}",
                  "- figure: `results/cea/guide_quality_curve/guide_quality_curve.png`", ""]

    # phase 9 efficiency
    p9 = summaries.get("phase9_efficiency")
    if p9:
        lines += ["### phase9_efficiency", "", f"- device: `{p9.get('device')}`, output {p9.get('output_shape')}", "",
                  "| method | params (M) | inference s/img |", "|---|---|---|"]
        for m, e in p9.get("methods", {}).items():
            lines.append(f"| {m} | {e.get('params_million')} | {e.get('inference_s_per_image')} |")
        if p9.get("flops"):
            lines += ["", f"- FLOPs: {json.dumps(p9['flops'])}"]
        lines.append("")

    # Table 2 main-results matrix (paper Q2 + appendix)
    t2_path = Path("results/cea/table2_matrix.json")
    if t2_path.exists():
        t2 = json.loads(t2_path.read_text(encoding="utf-8"))
        lines += ["### table2_matrix — main agricultural results (frozen SegFormer-B0)", "",
                  "Source: `scripts/cea_table2_matrix.py` → `results/cea/table2_main.tex` (main) / "
                  "`table2_appendix.tex` (appendix). Aggregate rows: absolute PSNR/SSIM/BF/mIoU; "
                  "regime sub-rows: ΔmIoU vs bicubic only.", "",
                  "| dataset | regime | method | PSNR | SSIM | BF | mIoU | ΔmIoU |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in t2.get("main", []):
            def _v(k):
                v = r.get(k)
                return "—" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
            lines.append(
                f"| {r['dataset']} | {r['degradation']} | {r['method'] or 'Δ vs bic.'} | "
                f"{_v('psnr')} | {_v('ssim')} | {_v('boundary_f')} | {_v('frozen_miou')} | {_v('delta_miou')} |"
            )
        lines += ["", f"- appendix rows: {len(t2.get('appendix', []))} (LoveDA OOD + WeedsGalore baselines)", ""]

    report = "\n".join(lines) + "\n"
    Path("results/cea/paper_report.md").write_text(report, encoding="utf-8")
    print(f"wrote results/cea/paper_report.md ({len(report)} chars) from phases: {sorted(summaries)}")


if __name__ == "__main__":
    main()
