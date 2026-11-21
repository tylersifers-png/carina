"""CSV export of parquet files and datasets: exact values, splitting, safe output."""

import hashlib
import json
from datetime import date, datetime, timezone
from decimal import Decimal

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import pytest

from carina import cli, export

UTC = timezone.utc


def sample_table():
    return pa.table(
        {
            "id": pa.array([1, 2**62, None], pa.int64()),
            "price": pa.array([0.1 + 0.2, 1e-9, None], pa.float64()),
            "amount": pa.array(
                [Decimal("1.2300"), Decimal("-0.0001"), None], pa.decimal128(12, 4)
            ),
            "day": pa.array([date(2026, 1, 5), date(1999, 12, 31), None], pa.date32()),
            "moment": pa.array(
                [datetime(2026, 1, 5, 1, 2, 3, 123456, tzinfo=UTC), None, None],
                pa.timestamp("us", "UTC"),
            ),
            "naive": pa.array(
                [datetime(2026, 1, 5, 1, 2, 3), datetime(2020, 2, 29), None],
                pa.timestamp("ns"),
            ),
            "text": pa.array(['a,"b"\nc', "plain", None], pa.string()),
            "flag": pa.array([True, False, None], pa.bool_()),
        }
    )


def read_back(path, table):
    return pacsv.read_csv(
        path,
        convert_options=pacsv.ConvertOptions(
            column_types=table.schema, strings_can_be_null=True
        ),
    )


def test_values_round_trip_exactly(tmp_path):
    source = tmp_path / "sample.parquet"
    table = sample_table()
    pq.write_table(table, source)
    record = export.export_one(source, tmp_path / "out")

    assert [f["name"] for f in record["files"]] == ["sample.csv"]
    assert read_back(tmp_path / "out" / "sample.csv", table).equals(table)

    text = (tmp_path / "out" / "sample.csv").read_text()
    assert "2026-01-05T01:02:03.123456Z" in text
    assert "1999-12-31" in text
    assert "1.2300" in text and "0.30000000000000004" in text
    assert str(2**62) in text


def test_schema_file_describes_the_export(tmp_path):
    source = tmp_path / "sample.parquet"
    pq.write_table(sample_table(), source)
    export.export_one(source, tmp_path / "out")
    schema = json.loads((tmp_path / "out" / "sample.schema.json").read_text())

    assert schema["schema"] == export.SCHEMA
    assert schema["source"] == "sample.parquet"
    assert (schema["rows"], schema["source_rows"], schema["complete"]) == (3, 3, True)
    formats = {c["name"]: c["csv_format"] for c in schema["columns"]}
    assert formats["amount"] == "exact-decimal"
    assert formats["moment"] == "iso8601-utc-timestamp"
    assert formats["naive"] == "iso8601-timestamp"
    assert formats["day"] == "iso8601-date"
    assert formats["id"] == "as-is"
    (entry,) = schema["files"]
    data = (tmp_path / "out" / "sample.csv").read_bytes()
    assert entry["bytes"] == len(data)
    assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    assert entry["rows"] == 3


def test_nested_and_binary_values_are_text(tmp_path):
    table = pa.table(
        {
            "tags": pa.array([["a", "b"], None], pa.list_(pa.string())),
            "info": pa.array(
                [{"n": 1, "d": Decimal("2.50")}, None],
                pa.struct([("n", pa.int32()), ("d", pa.decimal128(5, 2))]),
            ),
            "blob": pa.array([b"\x00\xff", None], pa.binary()),
        }
    )
    pq.write_table(table, tmp_path / "x.parquet")
    export.export_one(tmp_path / "x.parquet", tmp_path / "out")
    rows = pacsv.read_csv(
        tmp_path / "out" / "x.csv",
        convert_options=pacsv.ConvertOptions(strings_can_be_null=True),
    ).to_pylist()
    assert rows[0] == {"tags": '["a", "b"]', "info": '{"n": 1, "d": "2.50"}', "blob": "00ff"}
    assert rows[1] == {"tags": None, "info": None, "blob": None}


def test_null_and_empty_string_differ_in_the_file(tmp_path):
    table = pa.table({"a": pa.array(["", None, "x"]), "b": [1, 2, 3]})
    pq.write_table(table, tmp_path / "n.parquet")
    export.export_one(tmp_path / "n.parquet", tmp_path / "out")
    lines = (tmp_path / "out" / "n.csv").read_text().splitlines()
    assert lines == ['"a","b"', '"",1', ",2", '"x",3']


def make_hive(root):
    for year, rows in (("2024", [1, 2]), ("2025", [3])):
        part = root / f"year={year}" / "region=us"
        part.mkdir(parents=True)
        pq.write_table(pa.table({"n": rows}), part / "part-0.parquet")
    (root / "manifest.json").write_text("{}")
    (root / ".hidden").mkdir()
    pq.write_table(pa.table({"n": [99]}), root / ".hidden" / "skip.parquet")


def test_directory_exports_partitions_as_columns_and_skips_other_files(tmp_path):
    make_hive(tmp_path / "ds")
    record = export.export_one(tmp_path / "ds", tmp_path / "out")
    assert [c["name"] for c in record["columns"]] == ["n", "year", "region"]
    rows = pacsv.read_csv(tmp_path / "out" / "ds.csv").to_pylist()
    assert rows == [
        {"n": 1, "year": 2024, "region": "us"},
        {"n": 2, "year": 2024, "region": "us"},
        {"n": 3, "year": 2025, "region": "us"},
    ]


def test_a_directory_of_different_schemas_is_refused(tmp_path):
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    pq.write_table(pa.table({"a": [1]}), mixed / "one.parquet")
    pq.write_table(pa.table({"b": ["x"]}), mixed / "two.parquet")
    with pytest.raises(export.ExportError, match=r"do not share one schema \(1 differ.*two.parquet"):
        export.export_one(mixed, tmp_path / "out")
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())
    export.export_one(mixed / "two.parquet", tmp_path / "out")


def test_columns_and_limit(tmp_path):
    pq.write_table(sample_table(), tmp_path / "s.parquet")
    record = export.export_one(
        tmp_path / "s.parquet", tmp_path / "out", columns=["text", "id"], limit=2
    )
    assert [c["name"] for c in record["columns"]] == ["text", "id"]
    assert (record["rows"], record["source_rows"], record["complete"]) == (2, 3, False)
    assert pacsv.read_csv(tmp_path / "out" / "s.csv").num_rows == 2

    with pytest.raises(export.ExportError, match="Unknown columns: nope"):
        export.export_one(tmp_path / "s.parquet", tmp_path / "o2", columns=["nope"])
    record = export.export_one(tmp_path / "s.parquet", tmp_path / "o3", limit=0)
    assert record["rows"] == 0
    assert (tmp_path / "o3" / "s.csv").read_text().splitlines() == [
        '"id","price","amount","day","moment","naive","text","flag"'
    ]


def test_output_is_split_by_size_with_a_header_in_each_part(tmp_path):
    table = pa.table({"n": list(range(1000)), "s": [f"row-{i:04d}" for i in range(1000)]})
    pq.write_table(table, tmp_path / "big.parquet")
    limit = 3_000
    record = export.export_one(
        tmp_path / "big.parquet", tmp_path / "out", max_bytes=limit, batch_rows=50
    )
    names = [f["name"] for f in record["files"]]
    assert len(names) > 2
    assert names == [f"big-{i:03d}.csv" for i in range(1, len(names) + 1)]
    assert not (tmp_path / "out" / "big.csv").exists()
    combined = []
    for entry in record["files"]:
        path = tmp_path / "out" / entry["name"]
        assert path.stat().st_size == entry["bytes"] <= limit
        assert path.read_text().splitlines()[0] == '"n","s"'
        combined.extend(pacsv.read_csv(path).to_pylist())
    assert combined == table.to_pylist()
    assert sum(f["rows"] for f in record["files"]) == 1000


def test_a_large_batch_is_split_to_fit_and_an_oversize_row_is_an_error(tmp_path):
    table = pa.table({"n": list(range(1000)), "s": [f"row-{i:04d}" for i in range(1000)]})
    pq.write_table(table, tmp_path / "big.parquet")
    record = export.export_one(tmp_path / "big.parquet", tmp_path / "out", max_bytes=3_000)
    assert len(record["files"]) > 2
    assert all(f["bytes"] <= 3_000 for f in record["files"])
    assert sum(f["rows"] for f in record["files"]) == 1000

    pq.write_table(pa.table({"s": ["x" * 500] * 10}), tmp_path / "w.parquet")
    with pytest.raises(export.ExportError, match="single row is larger than --max-mb"):
        export.export_one(tmp_path / "w.parquet", tmp_path / "out2", max_bytes=400)
    assert not any(tmp_path.joinpath("out2").iterdir())


def test_earlier_output_is_protected_and_replaced_cleanly(tmp_path):
    table = pa.table({"n": list(range(400))})
    pq.write_table(table, tmp_path / "t.parquet")
    out = tmp_path / "out"
    export.export_one(tmp_path / "t.parquet", out, max_bytes=400, batch_rows=20)
    first = sorted(p.name for p in out.iterdir())
    assert len(first) > 3
    unrelated = out / "other.csv"
    unrelated.write_text("keep")

    with pytest.raises(export.ExportError, match="Refusing to replace"):
        export.export_one(tmp_path / "t.parquet", out)
    assert sorted(p.name for p in out.iterdir()) == sorted(first + ["other.csv"])

    export.export_one(tmp_path / "t.parquet", out, overwrite=True)
    assert sorted(p.name for p in out.iterdir()) == ["other.csv", "t.csv", "t.schema.json"]
    assert unrelated.read_text() == "keep"


def test_source_is_not_changed_and_bad_sources_are_reported(tmp_path):
    source = tmp_path / "s.parquet"
    pq.write_table(sample_table(), source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    export.export_one(source, tmp_path / "out")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before

    (tmp_path / "empty").mkdir()
    with pytest.raises(export.ExportError, match="No parquet files"):
        export.export_one(tmp_path / "empty", tmp_path / "out2")
    with pytest.raises(export.ExportError, match="not found"):
        export.export_one(tmp_path / "missing.parquet", tmp_path / "out2")
    link = tmp_path / "link.parquet"
    link.symlink_to(source)
    with pytest.raises(export.ExportError, match="symlink"):
        export.export_one(link, tmp_path / "out2")
    with pytest.raises(export.ExportError, match="Unsafe output name"):
        export.export_one(source, tmp_path / "out2", name="../evil")


def test_command_line(tmp_path, capsys):
    pq.write_table(sample_table(), tmp_path / "a.parquet")
    pq.write_table(sample_table(), tmp_path / "b.parquet")
    out = tmp_path / "out"
    assert cli.main(["export", str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet"),
                     "--out", str(out), "--columns", "id,text", "--limit", "2"]) == 0
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [p["files"] for p in printed] == [["a.csv"], ["b.csv"]]
    assert sorted(p.name for p in out.iterdir()) == [
        "a.csv", "a.schema.json", "b.csv", "b.schema.json"
    ]

    with pytest.raises(SystemExit, match="Refusing to replace"):
        cli.main(["export", str(tmp_path / "a.parquet"), "--out", str(out)])
    with pytest.raises(SystemExit, match="single source"):
        cli.main(["export", str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet"),
                  "--out", str(out), "--name", "x"])
    with pytest.raises(SystemExit, match="same output name"):
        cli.main(["export", str(tmp_path / "a.parquet"), str(tmp_path / "x" / "a.parquet"),
                  "--out", str(out)])


# --- queries -------------------------------------------------------------------


def trades(tmp_path):
    table = pa.table(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB", "BBB"],
            "qty": [1, 2, 3, 4, 5],
            "price": pa.array(
                [Decimal("10.5000"), Decimal("11.2500"), Decimal("20.0000"),
                 Decimal("21.0000"), Decimal("22.5000")],
                pa.decimal128(12, 4),
            ),
        }
    )
    pq.write_table(table, tmp_path / "trades.parquet")
    return table


def test_query_result_is_exported_with_its_text_and_unknown_total(tmp_path):
    trades(tmp_path)
    record = export.export_query(
        [tmp_path / "trades.parquet"],
        "SELECT symbol, sum(qty) AS qty, max(price) AS top FROM trades GROUP BY symbol ORDER BY symbol",
        tmp_path / "out",
    )
    assert [f["name"] for f in record["files"]] == ["query.csv"]
    rows = pacsv.read_csv(tmp_path / "out" / "query.csv").to_pylist()
    assert rows == [
        {"symbol": "AAA", "qty": 3, "top": 11.25},
        {"symbol": "BBB", "qty": 12, "top": 22.5},
    ]
    schema = json.loads((tmp_path / "out" / "query.schema.json").read_text())
    assert schema["query"].startswith("SELECT symbol")
    assert (schema["rows"], schema["source_rows"], schema["complete"]) == (2, None, True)
    assert schema["source_files"] == ["trades.parquet"]
    assert {c["name"]: c["csv_format"] for c in schema["columns"]}["top"] == "exact-decimal"
    text = (tmp_path / "out" / "query.csv").read_text()
    assert "11.2500" in text and "22.5000" in text


def test_single_source_is_also_called_data_and_values_round_trip(tmp_path):
    table = sample_table()
    pq.write_table(table, tmp_path / "sample.parquet")
    export.export_query(
        [tmp_path / "sample.parquet"],
        "SELECT id, price, amount, day, moment, naive, text, flag FROM data",
        tmp_path / "out",
        name="copy",
    )
    assert read_back(tmp_path / "out" / "copy.csv", table).equals(table)


def test_several_sources_can_be_joined_by_name(tmp_path):
    trades(tmp_path)
    pq.write_table(pa.table({"symbol": ["AAA", "BBB"], "sector": ["x", "y"]}), tmp_path / "ref-data.parquet")
    export.export_query(
        [tmp_path / "trades.parquet", tmp_path / "ref-data.parquet"],
        'SELECT t.symbol, r.sector, t.qty FROM trades t JOIN ref_data r USING (symbol) WHERE t.qty > 3 ORDER BY t.qty',
        tmp_path / "out",
    )
    assert pacsv.read_csv(tmp_path / "out" / "query.csv").to_pylist() == [
        {"symbol": "BBB", "sector": "y", "qty": 4},
        {"symbol": "BBB", "sector": "y", "qty": 5},
    ]
    with pytest.raises(export.ExportError, match="share the query name 'trades'"):
        other = tmp_path / "elsewhere"
        other.mkdir()
        pq.write_table(pa.table({"a": [1]}), other / "trades.parquet")
        export.export_query(
            [tmp_path / "trades.parquet", other / "trades.parquet"], "SELECT 1", tmp_path / "o2"
        )


def test_partition_columns_are_available_to_a_query(tmp_path):
    make_hive(tmp_path / "ds")
    export.export_query(
        [tmp_path / "ds"], "SELECT n, year FROM ds WHERE year = 2024 ORDER BY n", tmp_path / "out"
    )
    assert pacsv.read_csv(tmp_path / "out" / "query.csv").to_pylist() == [
        {"n": 1, "year": 2024},
        {"n": 2, "year": 2024},
    ]
    with pytest.raises(export.ExportError, match="do not share one schema"):
        mixed = tmp_path / "mixed"
        mixed.mkdir()
        pq.write_table(pa.table({"a": [1]}), mixed / "one.parquet")
        pq.write_table(pa.table({"b": [1]}), mixed / "two.parquet")
        export.export_query([mixed], "SELECT 1", tmp_path / "o2")


def test_query_limit_and_completeness(tmp_path):
    trades(tmp_path)
    source = [tmp_path / "trades.parquet"]
    cut = export.export_query(source, "SELECT qty FROM data ORDER BY qty", tmp_path / "a", limit=2)
    assert (cut["rows"], cut["complete"]) == (2, False)
    exact = export.export_query(source, "SELECT qty FROM data ORDER BY qty", tmp_path / "b", limit=5)
    assert (exact["rows"], exact["complete"]) == (5, True)
    none = export.export_query(source, "SELECT qty FROM data WHERE qty > 99", tmp_path / "c")
    assert (none["rows"], none["complete"]) == (0, True)
    assert (tmp_path / "c" / "query.csv").read_text().splitlines() == ['"qty"']


def test_query_output_can_be_split_by_size(tmp_path):
    pq.write_table(pa.table({"n": list(range(2000))}), tmp_path / "n.parquet")
    record = export.export_query(
        [tmp_path / "n.parquet"], "SELECT n, 'row-' || n AS s FROM data", tmp_path / "out",
        max_bytes=2_000,
    )
    assert len(record["files"]) > 3
    assert all(f["bytes"] <= 2_000 for f in record["files"])
    assert sum(f["rows"] for f in record["files"]) == 2000


@pytest.mark.parametrize(
    "sql, message",
    [
        ("SELECT 1; SELECT 2", "single SELECT"),
        ("COPY (SELECT 1) TO 'leak.csv'", "single SELECT"),
        ("ATTACH 'other.db'", "single SELECT"),
        ("PRAGMA database_list", "single SELECT"),
        ("SELEC nonsense", "Invalid query"),
        ("SELECT * FROM missing_table", "Query failed"),
        ("SELECT CAST('abc' AS INTEGER)", "Query failed"),
        ("SELECT interval 1 day AS i", "Unsupported column type"),
    ],
)
def test_only_a_single_working_select_is_run(tmp_path, sql, message):
    trades(tmp_path)
    with pytest.raises(export.ExportError, match=message):
        export.export_query([tmp_path / "trades.parquet"], sql, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*.csv"))
    assert not (tmp_path / "leak.csv").exists()


def test_a_query_can_read_only_its_sources(tmp_path):
    trades(tmp_path)
    secret = tmp_path / "secret.parquet"
    pq.write_table(pa.table({"s": ["hidden"]}), secret)
    (tmp_path / "private.csv").write_text("a\n1\n")
    for sql in (
        f"SELECT * FROM read_parquet('{secret}')",
        f"SELECT * FROM read_csv('{tmp_path / 'private.csv'}')",
        "SELECT * FROM read_text('/etc/hosts')",
        f"SELECT * FROM glob('{tmp_path}/*')",
        "SELECT current_setting('home_directory')",
    ):
        try:
            export.export_query([tmp_path / "trades.parquet"], sql, tmp_path / "out", overwrite=True)
        except export.ExportError as error:
            assert "Query failed" in str(error), sql
        else:
            assert "home_directory" in sql, sql


def test_existing_output_is_checked_before_the_query_runs(tmp_path):
    trades(tmp_path)
    export.export_query([tmp_path / "trades.parquet"], "SELECT 1 AS a", tmp_path / "out")
    with pytest.raises(export.ExportError, match="Refusing to replace"):
        export.export_query([tmp_path / "trades.parquet"], "SELEC broken", tmp_path / "out")
    export.export_query(
        [tmp_path / "trades.parquet"], "SELECT 2 AS a", tmp_path / "out", overwrite=True
    )
    assert (tmp_path / "out" / "query.csv").read_text().splitlines() == ['"a"', "2"]


def test_query_command_line(tmp_path, capsys):
    trades(tmp_path)
    source, out = str(tmp_path / "trades.parquet"), str(tmp_path / "out")
    assert cli.main(["export", source, "--out", out, "--name", "totals",
                     "--query", "SELECT symbol, sum(qty) AS qty FROM data GROUP BY symbol ORDER BY 1"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["files"] == ["totals.csv"] and printed["source_rows"] is None
    (tmp_path / "q.sql").write_text("SELECT count(*) AS n\nFROM trades\n")
    assert cli.main(["export", source, "--out", out, "--query-file", str(tmp_path / "q.sql")]) == 0
    assert (tmp_path / "out" / "query.csv").read_text().splitlines() == ['"n"', "5"]
    with pytest.raises(SystemExit, match="--columns cannot be used with a query"):
        cli.main(["export", source, "--out", out, "--query", "SELECT 1", "--columns", "symbol"])
    with pytest.raises(SystemExit, match="Cannot read the query file"):
        cli.main(["export", source, "--out", out, "--query-file", str(tmp_path / "none.sql")])
    with pytest.raises(SystemExit):
        cli.main(["export", source, "--out", out, "--query", "SELECT 1", "--query-file", "q.sql"])
