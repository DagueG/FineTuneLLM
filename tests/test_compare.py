from chsa_triage.eval.compare import compare, format_comparison_table


def test_compare_and_format():
    sft = {"priority_accuracy": 0.389, "safety_flagged": 2, "unparsed": 3,
           "avg_red_flag_recall": 0.042,
           "per_level_accuracy": {"urgence_vitale": 0.5, "urgent": 0.667, "non_urgent": 0.0}}
    dpo = {"priority_accuracy": 0.5, "safety_flagged": 1, "unparsed": 1,
           "avg_red_flag_recall": 0.1,
           "per_level_accuracy": {"urgence_vitale": 0.667, "urgent": 0.667, "non_urgent": 0.167}}
    comp = compare({"sft-lora": sft, "dpo-lora": dpo})
    assert comp["models"] == ["sft-lora", "dpo-lora"]
    assert comp["metrics"]["priority_accuracy"]["dpo-lora"] == 0.5
    assert comp["per_level"]["non_urgent"]["sft-lora"] == 0.0

    table = format_comparison_table(comp)
    assert "Exactitude priorité" in table
    assert "sft-lora" in table and "dpo-lora" in table
    assert "urgence_vitale" in table


def test_compare_handles_missing_metrics():
    comp = compare({"a": {}, "b": {"priority_accuracy": 0.4}})
    # ne lève pas, remplit avec None
    assert comp["metrics"]["priority_accuracy"]["a"] is None
    assert "-" in format_comparison_table(comp)
