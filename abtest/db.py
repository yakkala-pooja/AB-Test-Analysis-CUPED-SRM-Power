"""Load simulated logs into SQLite and run the queries in sql/."""
import sqlite3
from pathlib import Path

import pandas as pd

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def load(assignments, sessions, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    assignments.to_sql("assignments", conn, index=False)
    sessions.to_sql("sessions", conn, index=False)
    conn.execute("CREATE INDEX idx_sessions_user_day ON sessions (user_id, day)")
    return conn


def query(conn, name):
    return pd.read_sql_query((SQL_DIR / name).read_text(), conn)
