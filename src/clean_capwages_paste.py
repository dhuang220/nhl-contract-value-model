import csv
import pandas as pd

MONTHS = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
    "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}


def _parse_date(day_month: str, offseason_year: int) -> str:
    """Turn CapWages' '21-Sep' into a full ISO date, e.g. '2026-09-21'.

    CapWages' signings tracker only ever spans one league year (July 1 onward),
    so pairing every date with offseason_year is safe for a single pasted batch.
    """
    day, month_abbrev = day_month.split("-")
    return f"{offseason_year}-{MONTHS[month_abbrev]}-{int(day):02d}"


def _parse_cap_hit(raw: str) -> int:
    """Turn '$3,700,000 ' into 3700000."""
    return int(raw.replace("$", "").replace(",", "").strip())


def _reformat_name(last_first: str) -> str:
    """Turn 'Steel, Sam' into 'Sam Steel', matching the NHL API's naming."""
    last, first = last_first.split(", ", 1)
    return f"{first} {last}"


def _looks_like_a_signing_row(row: list[str]) -> bool:
    """True for a real data row, regardless of where it sits in the raw paste.

    Copy-pasting the page's virtualized table is unreliable about *position*
    (nav/sidebar/widget text can end up interleaved, and the table can get
    captured more than once) but real rows always have this exact shape:
    9 fields, 'Last, First' in the name field, and a numeric age.
    """
    if len(row) != 9:
        return False
    if ", " not in row[0]:
        return False
    return row[1].isdigit()


def clean_capwages_signings(raw_csv_path: str, offseason_year: int) -> pd.DataFrame:
    """Extract the real signings table out of a raw CapWages page copy-paste.

    Filters every row in the raw file down to ones that look like a real
    signing (see _looks_like_a_signing_row) rather than assuming the real
    table sits at a fixed position - the page's virtualized table can get
    captured more than once, or interleaved with nav/sidebar text, when the
    whole page is selected and copied. Duplicate rows (from a repeated
    capture) are dropped. contract_type is left blank - CapWages doesn't
    expose UFA/RFA status; that's computed separately from age + NHL debut
    year (see features.infer_contract_type), which is why `age` is kept in
    the output even though our template doesn't otherwise need it.
    """
    with open(raw_csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    table_rows = [row for row in rows if _looks_like_a_signing_row(row)]

    df = pd.DataFrame(table_rows, columns=[
        "player_raw", "age", "position", "team_signed", "date_raw",
        "contract_structure", "term_years", "cap_hit_raw", "total_value_raw",
    ])

    cleaned = pd.DataFrame({
        "player_name": df["player_raw"].apply(_reformat_name),
        "age": df["age"].astype(int),
        "signing_date": df["date_raw"].apply(lambda d: _parse_date(d, offseason_year)),
        "offseason_year": offseason_year,
        "contract_type": "",  # computed separately, see features.infer_contract_type
        "term_years": df["term_years"],
        "cap_hit": df["cap_hit_raw"].apply(_parse_cap_hit),
        "team_signed": df["team_signed"],
        "position": df["position"],
        "contract_structure": df["contract_structure"],
    })
    return cleaned.drop_duplicates(subset=["player_name", "signing_date", "cap_hit"])
