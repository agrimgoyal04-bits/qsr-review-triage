"""Read an uploaded spreadsheet (.csv or .xlsx) into a clean DataFrame with a 'text' column.

Rules (kept deliberately forgiving for non-technical users):
  * Column names are matched ignoring case and surrounding spaces: "Text", " TEXT " -> text.
    "review" / "review_text" / "comment" are also accepted as the text column.
  * Optional "brand" column.
  * .xlsx: the FIRST sheet is read.
  * .csv: UTF-8 (with or without BOM), falls back to Latin-1 for files saved by older Excel.
  * Empty / whitespace-only texts are dropped (and counted, so the UI can tell the user).
"""
import io
from typing import BinaryIO, Union

import pandas as pd

TEXT_ALIASES = ("text", "review", "review_text", "comment", "comments", "feedback")
MAX_FILE_ROWS = 100_000  # refuse absurdly large uploads early; the UI also caps rows to classify


class SpreadsheetError(ValueError):
    """Raised with a user-friendly message."""


def _read_raw(file: Union[BinaryIO, bytes], filename: str) -> pd.DataFrame:
    data = file if isinstance(file, (bytes, bytearray)) else file.read()
    name = (filename or "").lower()
    try:
        if name.endswith(".xlsx"):
            return pd.read_excel(io.BytesIO(data), sheet_name=0, engine="openpyxl", dtype=str)
        if name.endswith(".csv"):
            try:
                return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", dtype=str)
            except UnicodeDecodeError:
                return pd.read_csv(io.BytesIO(data), encoding="latin-1", dtype=str)
    except SpreadsheetError:
        raise
    except Exception as e:  # noqa: BLE001  (corrupt file, wrong format, empty file...)
        raise SpreadsheetError(
            f"Could not read '{filename}' ({type(e).__name__}). Is it a valid .csv or .xlsx file "
            f"(not password-protected, not an old .xls)?"
        ) from e
    raise SpreadsheetError("Unsupported file type. Please upload a .csv or .xlsx file (old .xls is not supported).")


def read_reviews_table(file: Union[BinaryIO, bytes], filename: str) -> "tuple[pd.DataFrame, int]":
    """Returns (dataframe with columns 'text' and maybe 'brand', number_of_blank_rows_dropped)."""
    df = _read_raw(file, filename)
    if len(df) > MAX_FILE_ROWS:
        raise SpreadsheetError(f"File has {len(df):,} rows; the limit is {MAX_FILE_ROWS:,}. Please split it.")
    df.columns = [str(c).strip().lower() for c in df.columns]

    text_col = next((c for c in TEXT_ALIASES if c in df.columns), None)
    if text_col is None:
        raise SpreadsheetError(
            f"No review-text column found. Columns in your file: {list(df.columns)}. "
            f"Rename the column that holds the review to 'text'."
        )
    out = pd.DataFrame({"text": df[text_col]})
    if "brand" in df.columns:
        out["brand"] = df["brand"]

    out["text"] = out["text"].fillna("").astype(str).str.strip()
    before = len(out)
    out = out[out["text"] != ""].reset_index(drop=True)
    if out.empty:
        raise SpreadsheetError("The text column is empty - there are no reviews to classify.")
    return out, before - len(out)
