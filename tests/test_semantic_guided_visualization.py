import json
from pathlib import Path

import numpy as np

from cea_plus.degradation import build_degradation_plan
from cea_plus.synthesis import make_synthetic_agri_sample


def test_guided_frequency_case_export_writes_mechanism_artifacts_without_oracle_inputs(tmp_path: Path):
    from cea_plus.semantic_guided_visualization import export_guided_frequency_case

    class GradientGuide:
        def predict_probability(self, image):
            h, w = image.shape[:2]
            x = np.linspace(0.1, 0.9, w, dtype=np.float32)
            return np.tile(x[None, :], (h, 1))

    sample = make_synthetic_agri_sample(seed=5400, size=32).with_name("case_0")
    summary = export_guided_frequency_case(
        output_dir=tmp_path / "case_export",
        sample=sample,
        degradation_config=build_degradation_plan(scales=(2,), modes=("fog",))[0],
        guide_segmenter=GradientGuide(),
        seed=61,
        prefix="case_0",
        base_strength=0.36,
        inner_boost=0.2,
        outer_boost=-0.2,
        probability_calibration="image_minmax",
        uniform_mix_weight=0.5,
    )

    payload = json.loads(summary.summary_json.read_text(encoding="utf-8"))

    assert summary.restored_png.exists()
    assert summary.probability_png.exists()
    assert summary.confidence_png.exists()
    assert summary.strength_png.exists()
    assert summary.spectrum_json.exists()
    assert payload["restoration_inputs"] == ["low_res", "output_shape", "guide_prediction"]
    assert payload["gt_mask_usage"] == "visualization_only"
    assert payload["degradation"] == "x2_fog"
    assert payload["spectrum"]["restored_high_frequency_energy"] >= 0.0
    assert payload["spectrum"]["bicubic_high_frequency_energy"] >= 0.0
