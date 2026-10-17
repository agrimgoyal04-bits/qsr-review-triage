import io

import pandas as pd
import pytest

from src.tabular import MAX_FILE_ROWS, SpreadsheetError, read_reviews_table


def xlsx_bytes(df, sheet="Sheet1", extra_sheet=None):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name=sheet)
        if extra_sheet is not None:
            extra_sheet.to_excel(w, index=False, sheet_name="Other")
    return buf.getvalue()


def test_xlsx_basic_with_brand():
    df = pd.DataFrame({"text": ["Cold fries", "Great!"], "brand": ["McDonald's", "Subway"]})
    out, blanks = read_reviews_table(xlsx_bytes(df), "reviews.xlsx")
    assert out["text"].tolist() == ["Cold fries", "Great!"] and out["brand"].tolist() == ["McDonald's", "Subway"]
    assert blanks == 0


def test_xlsx_reads_first_sheet_only():
    first = pd.DataFrame({"text": ["first sheet"]})
    second = pd.DataFrame({"text": ["second sheet"]})
    out, _ = read_reviews_table(xlsx_bytes(first, extra_sheet=second), "x.xlsx")
    assert out["text"].tolist() == ["first sheet"]


def test_column_names_case_and_spaces_and_aliases():
    out, _ = read_reviews_table(xlsx_bytes(pd.DataFrame({"  Review ": ["a b"], "BRAND": ["KFC"]})), "x.xlsx")
    assert out.columns.tolist() == ["text", "brand"] and out["brand"][0] == "KFC"


def test_blank_rows_dropped_and_counted():
    df = pd.DataFrame({"text": ["ok", None, "   ", "fine"]})
    out, blanks = read_reviews_table(xlsx_bytes(df), "x.xlsx")
    assert out["text"].tolist() == ["ok", "fine"] and blanks == 2


def test_numbers_in_text_column_become_strings():
    out, _ = read_reviews_table(xlsx_bytes(pd.DataFrame({"text": [12345, "word"]})), "x.xlsx")
    assert all(isinstance(t, str) for t in out["text"])


def test_csv_utf8_bom_and_latin1():
    csv = "text,brand\nCafé was cold,Starbucks\n"
    out, _ = read_reviews_table(("\ufeff" + csv).encode("utf-8"), "a.csv")
    assert out["text"][0] == "Café was cold"
    out2, _ = read_reviews_table(csv.encode("latin-1"), "b.csv")
    assert out2["text"][0] == "Café was cold"


def test_missing_text_column_gives_helpful_message():
    with pytest.raises(SpreadsheetError, match="No review-text column"):
        read_reviews_table(xlsx_bytes(pd.DataFrame({"comments_x": ["a"], "id": [1]})), "x.xlsx")


def test_empty_text_column_rejected():
    with pytest.raises(SpreadsheetError, match="empty"):
        read_reviews_table(xlsx_bytes(pd.DataFrame({"text": [None, " "]})), "x.xlsx")


@pytest.mark.parametrize("name", ["old.xls", "notes.txt", "data.pdf", ""])
def test_unsupported_extension(name):
    with pytest.raises(SpreadsheetError, match="Unsupported"):
        read_reviews_table(b"whatever", name)


def test_corrupt_xlsx_rejected_cleanly():
    with pytest.raises(SpreadsheetError, match="Could not read"):
        read_reviews_table(b"this is not a zip file", "broken.xlsx")


def test_file_like_object_accepted():
    out, _ = read_reviews_table(io.BytesIO(b"text\nhello\n"), "a.csv")
    assert out["text"].tolist() == ["hello"]


def test_row_limit(monkeypatch):
    monkeypatch.setattr("src.tabular.MAX_FILE_ROWS", 3)
    with pytest.raises(SpreadsheetError, match="limit"):
        read_reviews_table(b"text\na\nb\nc\nd\n", "a.csv")


def test_sample_csv_in_repo_loads():
    from src.config import DATA_DIR

    out, blanks = read_reviews_table((DATA_DIR / "sample_reviews.csv").read_bytes(), "sample_reviews.csv")
    assert len(out) == 8 and "brand" in out.columns and blanks == 0
