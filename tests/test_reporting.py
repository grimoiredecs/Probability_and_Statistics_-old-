from pathlib import Path

from scripts.run_pipeline import export_report_index


def test_report_index_links_both_domain_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    index = export_report_index(
        "outputs/reports/cpu-run/cpu/report.md",
        "outputs/reports/gpu-run/gpu/report.md",
    )

    assert index == Path("outputs/reports/latest.md")
    contents = (tmp_path / index).read_text(encoding="utf-8")
    assert "CPU model report" in contents
    assert "GPU model report" in contents
    assert "cpu-run/cpu/report.md" in contents
