from pipeline.bed_report import three_stage_table, format_table


def test_three_stage_table_calls_each_stage():
    seen = []
    def bed_fn(stage):
        seen.append(stage)
        return {"current": 1.0, "+ema": 0.7, "+ema+calib": 0.4}[stage]
    results = three_stage_table(bed_fn)
    assert seen == ["current", "+ema", "+ema+calib"]
    assert results["+ema+calib"] == 0.4


def test_format_table_mentions_all_stages():
    s = format_table({"current": 1.0, "+ema": 0.7, "+ema+calib": 0.4})
    assert "current" in s and "+ema+calib" in s
    assert "0.4" in s
