#!/usr/bin/env python3
"""Bulk-run signup automations from a CSV file using local (embedded) Skyvern.

Usage:
    python scripts/bulk_signup_runner.py --csv accounts.csv --url https://example.com/signup

accounts.csv columns (header required):
    name, date_of_birth, email, phone_number, address[, signup_url]

`signup_url` is optional per row and overrides --url for that row.

Progress/results are tracked in a local SQLite database (default:
signup_state.sqlite3) so the script is safely resumable: re-running it will
skip rows already marked "success" and retry only "pending"/"failed" ones.

Concurrency note: Skyvern's embedded local mode (Skyvern.local()) allows only
one embedded client per process, backed by an async, event-loop-bound engine
(aiosqlite). Because of that, this script caps concurrency with 3 concurrent
asyncio workers on a single event loop rather than 3 OS threads -- real OS
threads would each need their own event loop, which is unsafe to use
concurrently against the same embedded ASGI transport/DB engine. The effect
(3 signups in flight at a time) is the same as what "3 concurrent threads"
asks for; the mechanism is asyncio, not `threading`.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from skyvern import Skyvern
from skyvern.client.types.task_run_response import TaskRunResponse
from skyvern.schemas.llm import LLMConfig
from skyvern.schemas.run_enums import RunEngine, RunStatus

REQUIRED_CSV_FIELDS = ("name", "date_of_birth", "email", "phone_number", "address")

# Explicit LLM configuration: Claude 3.5 Haiku, to minimize per-signup token cost.
# Note: Claude 3.5 Haiku does not support image/vision input, unlike gpt-4o-mini or
# Claude 4.5 Haiku. Skyvern can still operate on the HTML/DOM representation of the
# page, but it loses screenshot-based visual grounding, which may reduce accuracy on
# highly visual or canvas-rendered signup forms.
LLM_CONFIG = LLMConfig(
    model_name="anthropic/claude-3-5-haiku-20241022",
    required_env_vars=["ANTHROPIC_API_KEY"],
    supports_vision=False,
    add_assistant_prefix=True,
)

DEFAULT_CONCURRENCY = 3
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_STEPS = 10


@dataclass(frozen=True)
class SignupRow:
    row_id: int
    name: str
    date_of_birth: str
    email: str
    phone_number: str
    address: str
    signup_url: str | None


class StateStore:
    """SQLite-backed state tracking for the bulk signup run."""

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signups (
                row_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                date_of_birth TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone_number TEXT NOT NULL,
                address TEXT NOT NULL,
                signup_url TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                run_id TEXT,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )

    def seed(self, rows: list[SignupRow]) -> None:
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO signups
                (row_id, name, date_of_birth, email, phone_number, address, signup_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [(r.row_id, r.name, r.date_of_birth, r.email, r.phone_number, r.address, r.signup_url) for r in rows],
        )

    def pending_rows(self, max_retries: int) -> list[SignupRow]:
        cursor = self._conn.execute(
            """
            SELECT row_id, name, date_of_birth, email, phone_number, address, signup_url
            FROM signups
            WHERE status != 'success' AND attempts <= ?
            ORDER BY row_id
            """,
            (max_retries,),
        )
        return [SignupRow(*row) for row in cursor.fetchall()]

    def mark_running(self, row_id: int) -> None:
        self._conn.execute(
            """
            UPDATE signups
            SET status = 'running', attempts = attempts + 1, updated_at = datetime('now')
            WHERE row_id = ?
            """,
            (row_id,),
        )

    def mark_success(self, row_id: int, run_id: str) -> None:
        self._conn.execute(
            """
            UPDATE signups
            SET status = 'success', run_id = ?, error = NULL, updated_at = datetime('now')
            WHERE row_id = ?
            """,
            (run_id, row_id),
        )

    def mark_failed(self, row_id: int, error: str, run_id: str | None = None) -> None:
        self._conn.execute(
            """
            UPDATE signups
            SET status = 'failed', run_id = COALESCE(?, run_id), error = ?, updated_at = datetime('now')
            WHERE row_id = ?
            """,
            (run_id, error[:2000], row_id),
        )

    def summary(self) -> dict[str, int]:
        cursor = self._conn.execute("SELECT status, COUNT(*) FROM signups GROUP BY status")
        return dict(cursor.fetchall())

    def close(self) -> None:
        self._conn.close()


def load_accounts_csv(csv_path: Path) -> list[SignupRow]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [field for field in REQUIRED_CSV_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} is missing required column(s): {', '.join(missing)}")

        rows: list[SignupRow] = []
        for row_id, raw in enumerate(reader):
            rows.append(
                SignupRow(
                    row_id=row_id,
                    name=raw["name"].strip(),
                    date_of_birth=raw["date_of_birth"].strip(),
                    email=raw["email"].strip(),
                    phone_number=raw["phone_number"].strip(),
                    address=raw["address"].strip(),
                    signup_url=(raw.get("signup_url") or "").strip() or None,
                )
            )
        return rows


def build_prompt(row: SignupRow) -> str:
    return (
        "Fill out the signup/registration form on this page and submit it. "
        "Use exactly these values for the matching fields:\n"
        f"- Full name: {row.name}\n"
        f"- Date of birth: {row.date_of_birth}\n"
        f"- Email: {row.email}\n"
        f"- Phone number: {row.phone_number}\n"
        f"- Address: {row.address}\n"
        "If the form splits name/address into multiple fields (e.g. first/last name, "
        "street/city/state/zip), split the values above accordingly. Skip any field that "
        "has no corresponding value above. Submit the form once all available fields are filled."
    )


async def process_row(
    skyvern: Skyvern,
    row: SignupRow,
    default_url: str | None,
    store: StateStore,
    max_steps: int,
    semaphore: asyncio.Semaphore,
) -> None:
    url = row.signup_url or default_url
    if not url:
        store.mark_failed(row.row_id, "No signup URL provided (row has no signup_url and --url was not set)")
        print(f"[row {row.row_id}] SKIPPED - no signup URL for {row.email}")
        return

    async with semaphore:
        store.mark_running(row.row_id)
        print(f"[row {row.row_id}] starting signup for {row.email}")
        try:
            task_run: TaskRunResponse = await skyvern.run_task(
                prompt=build_prompt(row),
                url=url,
                engine=RunEngine.skyvern_v1,
                title=f"Signup - {row.email}",
                max_steps=max_steps,
                wait_for_completion=True,
            )
        except Exception as exc:  # noqa: BLE001 - persist any failure and keep the batch going
            store.mark_failed(row.row_id, f"{type(exc).__name__}: {exc}")
            print(f"[row {row.row_id}] FAILED (exception) for {row.email}: {exc}")
            return

        if task_run.status == RunStatus.completed:
            store.mark_success(row.row_id, task_run.run_id)
            print(f"[row {row.row_id}] SUCCESS for {row.email} (run_id={task_run.run_id})")
        else:
            store.mark_failed(
                row.row_id,
                task_run.failure_reason or f"Run ended with status={task_run.status}",
                task_run.run_id,
            )
            print(f"[row {row.row_id}] FAILED for {row.email} (status={task_run.status})")


async def run_batch(
    rows: list[SignupRow],
    default_url: str | None,
    store: StateStore,
    concurrency: int,
    max_steps: int,
) -> None:
    skyvern = Skyvern.local(llm_config=LLM_CONFIG, use_in_memory_db=True)
    semaphore = asyncio.Semaphore(concurrency)
    try:
        await asyncio.gather(*(process_row(skyvern, row, default_url, store, max_steps, semaphore) for row in rows))
    finally:
        await skyvern.aclose()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("accounts.csv"), help="Path to the accounts CSV file")
    parser.add_argument(
        "--db", type=Path, default=Path("signup_state.sqlite3"), help="Path to the SQLite state-tracking database"
    )
    parser.add_argument("--url", type=str, default=None, help="Signup page URL (used unless a row sets signup_url)")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max concurrent signups (default: 3)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Max attempts per row before it's left as permanently failed (default: 2)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Max agent steps per signup task (default: 10)",
    )
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    if not args.csv.exists():
        print(f"CSV file not found: {args.csv}", file=sys.stderr)
        return 1

    rows = load_accounts_csv(args.csv)
    store = StateStore(args.db)
    store.seed(rows)

    pending = store.pending_rows(args.max_retries)
    if not pending:
        print("Nothing to do - all rows already succeeded or exhausted retries.")
        print(store.summary())
        store.close()
        return 0

    print(f"Loaded {len(rows)} accounts, {len(pending)} pending/retryable, concurrency={args.concurrency}")
    start = time.monotonic()
    await run_batch(pending, args.url, store, args.concurrency, args.max_steps)
    elapsed = time.monotonic() - start

    summary = store.summary()
    print(f"\nDone in {elapsed:.1f}s. Status summary: {summary}")
    store.close()
    return 0 if summary.get("failed", 0) == 0 else 2


def main() -> int:
    args = parse_args(sys.argv[1:])
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
