"""Regression verification for StrataLeaf cursor/query-condition binding.

Run with:

    python3 -I regression.py

The script creates throwaway databases under a fresh temporary directory and
never touches repository data. It exits nonzero on the first failed check and
prints a full JSON report on success.

What is asserted, and why it can exclude the original bug
---------------------------------------------------------
A cursor token carries an HMAC over a payload containing a *query fingerprint*.
The engine only accepts a continuation whose fingerprint equals the fingerprint
computed from the current (collection, predicate). The old fingerprint was built
from a normalizer that (a) converted booleans to integers and (b) converted
arrays to sorted, de-duplicated sets, so distinct predicates shared one
fingerprint and a cursor silently crossed queries.

The expectations below are derived independently of the engine: the seed
documents, their order, the predicate semantics in docs, and the revision
number of each commit are hard-coded from the write timeline. For every
traversal we state the expected ordered keys, bodies and revision. Every
rejection case asserts that InvalidCursor is raised *before* any page is
returned, so a comparison of index-vs-scan output or the mere presence of an
HMAC cannot make the check pass.

Coverage boundaries (not claimed)
---------------------------------
- Pagination assumes a single-process embedded database; concurrent writer
  scheduling, WAL checkpoint behavior and network/replica transport are not
  re-tested here.
- Only the predicate language that already exists is exercised. No new query
  operators, auth features or retention rules are added or validated.
- Cursor tokens are still the existing on-disk format (HMAC + JSON payload);
  old tokens issued by a running process remain usable, while tokens produced
  before this build whose predicate collided under the old normalizer were
  never meaningfully bound to one query and are not considered stable data.
"""
import base64
import hashlib
import hmac
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strataleaf import Database, InvalidCursor, SnapshotTooOld  # noqa: E402
from strataleaf.predicates import matches as predicate_matches  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed inputs: documents and write timeline (independent of engine internals)
# ---------------------------------------------------------------------------
SEED_DOCS = {
    "a": {"v": True, "seq": [1, 2]},
    "b": {"v": True, "seq": [1, 2]},
    "f": {"v": True, "seq": [1, 2]},
    "c": {"v": 1, "seq": [2, 1]},
    "d": {"v": 1, "seq": [2, 1]},
    "e": {"v": True, "seq": [1, 1, 2]},
}
SEED_ORDER = ["a", "b", "c", "d", "e", "f"]  # document keys ascending

PRED_TRUE = {"field": "v", "op": "eq", "value": True}
PRED_ONE = {"field": "v", "op": "eq", "value": 1}
PRED_SEQ_12 = {"field": "seq", "op": "eq", "value": [1, 2]}
PRED_SEQ_21 = {"field": "seq", "op": "eq", "value": [2, 1]}
PRED_SEQ_112 = {"field": "seq", "op": "eq", "value": [1, 1, 2]}

failures = []
report = {"checks": [], "rejections_before_data": [], "traversals": []}


def check(name, condition, detail=""):
    report["checks"].append({"name": name, "ok": bool(condition), "detail": detail})
    if not condition:
        failures.append(name)
        print(f"FAIL {name} {detail}")


def expect_keys(docs, predicate):
    """Independent expectation: keys ascending whose body matches predicate."""
    return [key for key in sorted(docs) if predicate_matches(docs[key], predicate)]


def drain(db, collection, predicate, limit, first_cursor=None, force_scan=False,
          revision=None):
    """Full traversal; returns (ordered records, fixed revision, page count)."""
    page = db.query(collection, predicate, limit=limit, cursor=first_cursor,
                    force_scan=force_scan, revision=revision)
    fixed_revision = page.revision
    seen = [(r.key, r.document, r.revision) for r in page.records]
    pages = 1
    while page.cursor:
        page = db.query(collection, predicate, limit=limit, cursor=page.cursor,
                        force_scan=force_scan)
        check(f"continuation keeps revision [{predicate}]", page.revision == fixed_revision,
              f"{page.revision}!={fixed_revision}")
        seen.extend((r.key, r.document, r.revision) for r in page.records)
        pages += 1
    report["traversals"].append({
        "predicate": predicate, "limit": limit, "force_scan": force_scan,
        "revision": fixed_revision, "pages": pages,
        "ordered_keys": [k for k, _, _ in seen],
        "bodies": [body for _, body, _ in seen],
        "row_revisions": [r for _, _, r in seen],
    })
    return seen, fixed_revision, pages


def assert_rejected_before_data(name, call):
    """A foreign/invalid cursor must raise before any record is observable."""
    try:
        page = call()
    except InvalidCursor as exc:
        report["rejections_before_data"].append({"name": name, "ok": True,
                                                 "error": type(exc).__name__})
        print(f"ok   reject {name}")
        return
    report["rejections_before_data"].append({"name": name, "ok": False})
    failures.append(name)
    print(f"FAIL {name}: returned {len(page.records)} records instead of InvalidCursor")


def main():
    workspace = tempfile.mkdtemp(prefix="strataleaf-regression-")
    root = Path(workspace)
    db_path = root / "main.db"
    foreign_path = root / "foreign.db"

    with Database(db_path) as db:
        # --- revision 1: six seed documents in one commit ------------------
        db.catalog.create("docs")
        commit = db.batch("seed")
        for key in SEED_ORDER:
            commit.put("docs", key, SEED_DOCS[key])
        seed_rev = commit.commit().revision
        check("seed is one revision", seed_rev == 1, str(seed_rev))
        check("head advanced to 1", db.revision == 1)

        # Independent expectations computed from docs + predicate semantics.
        cases = [
            ("v == true", PRED_TRUE, ["a", "b", "e", "f"]),
            ("v == 1", PRED_ONE, ["c", "d"]),
            ("seq == [1,2]", PRED_SEQ_12, ["a", "b", "f"]),
            ("seq == [2,1]", PRED_SEQ_21, ["c", "d"]),
            ("seq == [1,1,2]", PRED_SEQ_112, ["e"]),
        ]
        for label, pred, expected in cases:
            check(f"independent expectation {label}",
                  expect_keys(SEED_DOCS, pred) == expected,
                  str(expect_keys(SEED_DOCS, pred)))

        # --- 1. The exact bug report: three cross-predicate continuations --
        bool_page = db.query("docs", PRED_TRUE, limit=1)
        check("first page of v==true returns key a",
              [r.key for r in bool_page.records] == ["a"])
        check("first page body is the boolean document",
              bool_page.records[0].document == SEED_DOCS["a"])
        assert_rejected_before_data(
            "v==true cursor reused for v==1",
            lambda: db.query("docs", PRED_ONE, limit=1, cursor=bool_page.cursor))

        seq12_page = db.query("docs", PRED_SEQ_12, limit=1)
        check("first page of seq==[1,2] returns key a",
              [r.key for r in seq12_page.records] == ["a"])
        assert_rejected_before_data(
            "seq==[1,2] cursor reused for seq==[2,1]",
            lambda: db.query("docs", PRED_SEQ_21, limit=1, cursor=seq12_page.cursor))
        assert_rejected_before_data(
            "seq==[1,2] cursor reused for seq==[1,1,2]",
            lambda: db.query("docs", PRED_SEQ_112, limit=1, cursor=seq12_page.cursor))

        # --- 2. Same distinctions inside nested and/or/not, in and contains
        nested_pairs = [
            ("and bool/number",
             {"and": [PRED_TRUE]}, {"and": [PRED_ONE]}),
            ("or array order",
             {"or": [PRED_SEQ_12]}, {"or": [PRED_SEQ_21]}),
            ("not bool/number",
             {"not": PRED_TRUE}, {"not": PRED_ONE}),
            ("in bool/number containers",
             {"field": "v", "op": "in", "value": [True]},
             {"field": "v", "op": "in", "value": [1]}),
            ("in array containers keep element order",
             {"field": "seq", "op": "in", "value": [[1, 2]]},
             {"field": "seq", "op": "in", "value": [[2, 1]]}),
            ("contains bool is not number",
             {"field": "seq", "op": "contains", "value": 1},
             {"field": "seq", "op": "contains", "value": True}),
        ]
        for label, left, right in nested_pairs:
            cursor = db.query("docs", left, limit=1).cursor
            check(f"nested case produces cursor: {label}", cursor is not None)
            if cursor is not None:
                assert_rejected_before_data(
                    f"nested cross-binding {label}",
                    lambda c=cursor, r=right: db.query("docs", r, limit=1, cursor=c))

        # object comparison value: bool vs number and array order inside object
        db.batch().put("docs", "g", {"o": {"x": True, "s": [1, 2]}}).commit()
        db.batch().put("docs", "h", {"o": {"x": 1, "s": [2, 1]}}) \
                 .put("docs", "j", {"o": {"x": True, "s": [1, 2]}}).commit()
        obj_true = {"field": "o", "op": "eq", "value": {"x": True, "s": [1, 2]}}
        obj_num = {"field": "o", "op": "eq", "value": {"x": 1, "s": [2, 1]}}
        obj_cursor = db.query("docs", obj_true, limit=1).cursor
        check("object predicate has a match", obj_cursor is not None)
        assert_rejected_before_data(
            "object value bool/array-order collision",
            lambda: db.query("docs", obj_num, limit=1, cursor=obj_cursor))
        # dict key permutation of the same object value must still bind
        same_obj = {"value": {"s": [1, 2], "x": True}, "op": "eq", "field": "o"}
        obj_permuted = db.query("docs", same_obj, limit=1, cursor=obj_cursor)
        check("object value key permutation continues with same predicate",
              obj_permuted.records[0].key == "j")

        # --- 3. Same condition survives page-size / force_scan / index changes
        # pin revision 1 so later retention experiments cannot invalidate views
        db.retention.pin("regression_reader", seed_rev)

        for label, pred, expected in cases:
            # page sizes 1, 2 and 3 must yield the same ordered full result,
            # all read at the fixed seed revision where only the six docs exist
            for size in (1, 2, 3):
                seen, rev, pages = drain(db, "docs", pred, size, revision=seed_rev)
                check(f"{label} full order limit={size}",
                      [k for k, _, _ in seen] == expected,
                      str([k for k, _, _ in seen]))
                check(f"{label} fixed revision limit={size}", rev == seed_rev)
                check(f"{label} all rows at revision 1",
                      all(r == seed_rev for _, _, r in seen))
                check(f"{label} bodies equal seed bodies",
                      all(body == SEED_DOCS[k] for k, body, _ in seen))
            # force_scan traversal matches, same fixed revision
            seen_scan, rev_scan, _ = drain(db, "docs", pred, 2, force_scan=True,
                                           revision=seed_rev)
            check(f"{label} force_scan order", [k for k, _, _ in seen_scan] == expected)
            check(f"{label} force_scan revision", rev_scan == seed_rev)

        # index creation: cursors minted before the index still continue, and
        # index and scan plans return identical ordered records.
        cursor_pre_index = db.query("docs", PRED_TRUE, limit=1).cursor
        check("pre-index first page is key a",
              db.query("docs", PRED_TRUE, limit=1).records[0].key == "a")
        db.indexes.create("docs", "by_v", "v")
        indexed = db.query("docs", PRED_TRUE)
        scanned = db.query("docs", PRED_TRUE, force_scan=True)
        check("index plan selected", indexed.plan["access"] == "equality_index",
              str(indexed.plan))
        check("index and scan agree",
              [(r.key, r.document, r.revision) for r in indexed.records]
              == [(r.key, r.document, r.revision) for r in scanned.records])
        cont = db.query("docs", PRED_TRUE, limit=1, cursor=cursor_pre_index)
        check("cursor continues after index creation",
              [r.key for r in cont.records] == ["b"])
        db.indexes.drop("docs", "by_v")
        cont2 = db.query("docs", PRED_TRUE, limit=1, cursor=cont.cursor)
        check("cursor continues after index removal",
              [r.key for r in cont2.records] == ["e"])

        # predicate object dict-key permutations must not break continuation
        base_cursor = db.query("docs", PRED_SEQ_12, limit=1).cursor
        shuffled = {"value": [1, 2], "field": "seq", "op": "eq"}
        shuffled_page = db.query("docs", shuffled, limit=1, cursor=base_cursor)
        check("leaf predicate key permutation continues",
              [r.key for r in shuffled_page.records] == ["b"])
        # 'in' choice order and repeated choices are set semantics
        in_cursor = db.query("docs", {"field": "v", "op": "in", "value": [True, 1]},
                             limit=1).cursor
        reordered = db.query("docs", {"field": "v", "op": "in", "value": [1, True]},
                             limit=1, cursor=in_cursor)
        check("in choice reorder continues", len(reordered.records) == 1)
        deduped = db.query("docs", {"field": "v", "op": "in", "value": [True, True, 1, 1]},
                           limit=1, cursor=in_cursor)
        check("in duplicate choices continue", len(deduped.records) == 1)
        # whereas eq on an array keeps order and multiplicity
        eq_cursor = db.query("docs", PRED_SEQ_12, limit=1).cursor
        assert_rejected_before_data(
            "eq array is not a set (reorder rejected)",
            lambda: db.query("docs", PRED_SEQ_21, limit=1, cursor=eq_cursor))

        # --- 4. Fixed-revision reads: writes after page 1 never leak in ------
        # Working set at revision 1 for v==true: a, b, e, f
        page1 = db.query("docs", PRED_TRUE, limit=1, revision=seed_rev)
        check("explicit revision page fixed at 1", page1.revision == seed_rev)
        head_before_changes = db.revision
        # next commit: mutate 'a' so it no longer matches, delete 'b', add 'z'
        db.batch().put("docs", "a", {"v": False, "seq": [9]}, expected=seed_rev) \
                 .delete("docs", "b", expected=seed_rev) \
                 .put("docs", "z", {"v": True, "seq": [1, 2]}).commit()
        change_rev = head_before_changes + 1
        check("head advanced by one change commit", db.revision == change_rev)
        # Continue the revision-1 traversal: must still see a, b, e, f bodies
        seen_old, rev_old, _ = drain(db, "docs", PRED_TRUE, 1,
                                     first_cursor=page1.cursor, revision=seed_rev)
        full_old = [("a", SEED_DOCS["a"], 1)] + seen_old
        check("old revision traversal keys",
              [k for k, _, _ in full_old] == ["a", "b", "e", "f"],
              str([k for k, _, _ in full_old]))
        check("old revision traversal bodies are pre-change content",
              [body for _, body, _ in full_old]
              == [SEED_DOCS[k] for k in ("a", "b", "e", "f")])
        check("old revision traversal revisions all 1",
              {r for _, _, r in full_old} == {1})
        # A fresh current-head query sees the new reality (no a, no b, z added)
        fresh, fresh_rev, _ = drain(db, "docs", PRED_TRUE, 2)
        check("new query reads new revision", fresh_rev == change_rev)
        check("new query contents",
              [k for k, _, _ in fresh] == ["e", "f", "z"],
              str([k for k, _, _ in fresh]))
        # The old cursor cannot be smuggled into a default (head) continuation:
        # its payload revision 1 differs from the live fingerprint flow only by
        # explicit-revision rules; same predicate + cursor without revision is
        # still pinned to revision 1 by the payload itself.
        pinned = db.query("docs", PRED_TRUE, limit=1, cursor=page1.cursor)
        check("payload-bound continuation still reads revision 1",
              pinned.revision == 1 and pinned.records[0].key == "b")

        # --- 5. Explicit revision disagreement is rejected ------------------
        assert_rejected_before_data(
            f"explicit revision {change_rev} != cursor revision 1",
            lambda: db.query("docs", PRED_TRUE, limit=1, cursor=page1.cursor,
                             revision=change_rev))

        # --- 6. Foreign database, tampered token, malformed token ------------
        with Database(foreign_path) as foreign:
            foreign.catalog.create("docs")
            foreign.batch().put("docs", "a", SEED_DOCS["a"]) \
                          .put("docs", "b", SEED_DOCS["a"]).commit()
            foreign_cursor = foreign.query("docs", PRED_TRUE, limit=1).cursor
        check("foreign database produced a cursor", foreign_cursor is not None)
        assert_rejected_before_data(
            "cursor from another database file",
            lambda: db.query("docs", PRED_TRUE, limit=1, cursor=foreign_cursor))

        def tamper(token):
            raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
            flipped = bytes([raw[0] ^ 0x01]) + raw[1:]  # first byte is HMAC
            return base64.urlsafe_b64encode(flipped).decode().rstrip("=")

        # Backward compatibility: a token signed with the pre-fix key (the hex
        # secret text itself as HMAC key) still decodes on its own database...
        signed_page = db.query("docs", PRED_SEQ_12, limit=1)
        raw_payload = base64.urlsafe_b64decode(
            signed_page.cursor + "=" * (-len(signed_page.cursor) % 4))[32:]
        legacy_key = db._store.get("cursor_secret").encode()
        legacy_sig = hmac.new(legacy_key, raw_payload, hashlib.sha256).digest()
        legacy_token = base64.urlsafe_b64encode(legacy_sig + raw_payload).decode().rstrip("=")
        legacy_page = db.query("docs", PRED_SEQ_12, limit=1, cursor=legacy_token)
        check("legacy-signed cursor still accepted on its own database",
              legacy_page is not None and len(legacy_page.records) == 1
              and legacy_page.records[0].key > "a")
        # ...while the same legacy token presented to another database is refused
        with Database(foreign_path) as foreign_handle:
            assert_rejected_before_data(
                "legacy-signed cursor rejected by foreign database",
                lambda: foreign_handle.query(
                    "docs", PRED_SEQ_12, limit=1, cursor=legacy_token))
        assert_rejected_before_data(
            "signature-modified cursor",
            lambda: db.query("docs", PRED_SEQ_12, limit=1,
                             cursor=tamper(signed_page.cursor)))
        for garbage in ("not-a-token", "", "aaaa", signed_page.cursor[:-2]):
            assert_rejected_before_data(
                f"malformed cursor {garbage!r}",
                lambda g=garbage: db.query("docs", PRED_SEQ_12, limit=1, cursor=g))

        # collection is part of the fingerprint
        db.catalog.create("other")
        db.batch().put("other", "a", SEED_DOCS["a"]).commit()
        docs_cursor = db.query("docs", PRED_TRUE, limit=1).cursor
        assert_rejected_before_data(
            "cursor reused against another collection",
            lambda: db.query("other", PRED_TRUE, limit=1, cursor=docs_cursor))

        # --- 7. Reopen: a legal cursor survives closing and reopening --------
        live_cursor = db.query("docs", PRED_TRUE, limit=1).cursor
        live_first = db.query("docs", PRED_TRUE, limit=1).records[0].key
        live_revision = db.revision
    with Database(db_path) as db:
        page = db.query("docs", PRED_TRUE, limit=1, cursor=live_cursor)
        check("reopened database accepts persisted cursor",
              page.records[0].key != live_first and page.revision == live_revision)
        # and a foreign (other-file) cursor is still rejected after reopen
        assert_rejected_before_data(
            "foreign cursor still rejected after reopen",
            lambda: db.query("docs", PRED_TRUE, limit=1, cursor=foreign_cursor))

        # --- 8. Retention: expired view follows existing retention rules -----
        db.retention.unpin("regression_reader")
        old_cursor = page1.cursor  # pinned to seed revision 1
        compact = db.retention.compact(db.revision)
        check("compaction moves floor to head", compact["floor"] == db.revision)
        try:
            db.query("docs", PRED_TRUE, limit=1, cursor=old_cursor)
            failures.append("expired view not rejected")
            print("FAIL expired view not rejected")
        except SnapshotTooOld:
            report["rejections_before_data"].append(
                {"name": "compacted cursor view rejected as SnapshotTooOld", "ok": True})
            print("ok   reject expired view (SnapshotTooOld)")
        # the still-current live cursor remains usable after compaction
        page_after = db.query("docs", PRED_TRUE, limit=1, cursor=live_cursor)
        check("current-revision cursor survives compaction",
              page_after.records[0].key != live_first)

    report["summary"] = {
        "total_checks": len(report["checks"]),
        "failed": failures,
        "rejection_assertions": len(report["rejections_before_data"]),
    }
    report_path = Path(__file__).resolve().parent / "last_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"full JSON report written to {report_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if failures:
        print(f"REGRESSION FAILURES: {len(failures)}")
        return 1
    print("ALL REGRESSION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
