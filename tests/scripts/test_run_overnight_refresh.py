from scripts.run_overnight_refresh import _sanitize_for_console


def test_sanitize_for_console_replaces_unencodable_chars_for_cp1252():
    text = "bad\ufffdchar"
    assert _sanitize_for_console(text, encoding="cp1252") == "bad?char"


def test_sanitize_for_console_preserves_utf8_text():
    text = "ok \u2713"
    assert _sanitize_for_console(text, encoding="utf-8") == text
