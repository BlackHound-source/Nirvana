import os
import sqlite3
import tempfile

import store


def test_schema_creates_all_tables():
    with tempfile.TemporaryDirectory() as d:
        con = store.connect(os.path.join(d, "t.sqlite"))
        names = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        con.close()
    assert {"runs", "predictions", "maintenance_records", "exceptions"} <= names


def test_work_orders_are_idempotent():
    with tempfile.TemporaryDirectory() as d:
        con = store.connect(os.path.join(d, "t.sqlite"))
        rows = [{"key": "k1", "engine_id": 1, "cycle": 5, "rul": 8.0,
                 "severity": "critical", "action": "a"}]
        with con:
            first = store.write_work_orders(con, "r", "FD001", rows, "t")
            second = store.write_work_orders(con, "r", "FD001", rows, "t")
        assert first == 1 and second == 0
        assert store.count(con, "maintenance_records") == 1
        con.close()
