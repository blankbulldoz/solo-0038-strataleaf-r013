"""Reproduce StrataLeaf cursor condition cross-binding.

Run with: python3 -I reproduce.py
Creates a throwaway database in a fresh temporary directory.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strataleaf import Database, InvalidCursor  # noqa: E402

DOCS = {
    "a": {"v": True, "seq": [1, 2]},
    "b": {"v": True, "seq": [1, 2]},
    "f": {"v": True, "seq": [1, 2]},
    "c": {"v": 1, "seq": [2, 1]},
    "d": {"v": 1, "seq": [2, 1]},
    "e": {"v": True, "seq": [1, 1, 2]},
}


def main():
    with tempfile.TemporaryDirectory(prefix="strataleaf-repro-") as directory:
        db_path = Path(directory) / "repro.db"
        with Database(db_path) as db:
            db.catalog.create("docs")
            batch = db.batch("seed")
            for key, doc in DOCS.items():
                batch.put("docs", key, doc)
            batch.commit()

            failures = []

            # 1. bool vs number: cursor of v==true reused for v==1
            p_bool = {"field": "v", "op": "eq", "value": True}
            page = db.query("docs", p_bool, limit=1)
            print("page1 (v == true), limit=1:", [(r.key, r.document) for r in page.records])
            p_num = {"field": "v", "op": "eq", "value": 1}
            try:
                wrong = db.query("docs", p_num, limit=1, cursor=page.cursor)
                print("BUG: numeric query accepted foreign cursor, returned:",
                      [(r.key, r.document) for r in wrong.records])
                failures.append("bool/number collision")
            except InvalidCursor:
                print("ok: v==1 continuation with v==true cursor rejected")

            # 2. ordered arrays: [1,2] cursor reused for [2,1] and [1,1,2]
            for seq in ([2, 1], [1, 1, 2]):
                p_first = {"field": "seq", "op": "eq", "value": [1, 2]}
                first_page = db.query("docs", p_first, limit=1)
                p_other = {"field": "seq", "op": "eq", "value": seq}
                try:
                    wrong = db.query("docs", p_other, limit=1, cursor=first_page.cursor)
                    print(f"BUG: seq=={seq} accepted [1,2] cursor, returned:",
                          [(r.key, r.document) for r in wrong.records])
                    failures.append(f"array collision {seq}")
                except InvalidCursor:
                    print(f"ok: seq=={seq} continuation with seq==[1,2] cursor rejected")

        if failures:
            print("REPRODUCED:", failures)
            return 1
        print("no collisions observed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
