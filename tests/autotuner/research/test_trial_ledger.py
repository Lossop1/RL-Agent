"""Pinned behaviour of the trial-ledger engine: appending costs the same no matter how long the file already is, two processes writing at once do not reuse a sequence number, a crash leaves a half-line that can be judged and healed, and a field with no JSON form is refused before it reaches the disk.

Each of these properties corresponds to one way of **quietly going wrong**, which is why
each is worth pinning on its own:

* Re-reading the whole file on append makes the cost grow with the square of the search
  length -- and a search is exactly the one user that makes the file long.  This property is
  **structural**, so it is proved with a counter (the call did not happen) rather than with
  a stopwatch.
* If the lock did not reach across processes, two writers would compute the same sequence
  number and from then on every read would fail permanently at the chain check: the ledger
  becomes a file that can never be read again, rather than a file missing one record.
* A process that dies halfway through writing a line leaves half a line.  Judging "is the
  last line complete" has to look at whether the file ends with a newline: for a single-line
  file, ``len(splitlines())`` is 1 in both the complete and the truncated shape, so the line
  count cannot tell them apart.
* Handing a model straight to ``content_hash`` goes through its ``exclude_none``, fields
  whose value is ``None`` disappear quietly, and "the record that was written" is no longer
  the same thing as "the record that was read back".
* A blank line is not an event.  The write side derives "the next sequence" from the last
  *event* it can find, so a reader that counted physical lines would disagree with the
  writer about every event after the first blank, and a perfectly successful append would
  leave a file that this module's own verify read rejects.
* A writer killed inside its critical section never releases anything itself.  If the lock
  were a file's existence, the next writer's only way forward would be to guess from that
  file's age that the holder is gone -- and the same guess takes the lock away from a holder
  that is alive but slow.  A lock the kernel holds and hands back removes the guess.
* The tail window is a *window*, not a limit on how long a line may be.  A record larger
  than it must still be found as the predecessor, or the next append starts a second chain
  at sequence 1 with an empty predecessor and no reader will ever accept the file again.
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from autotuner.research.research_ledger import canonical_json, content_hash
from autotuner.research.trial_ledger import (
    DEFAULT_TAIL_WINDOW_BYTES,
    TRIAL_LEDGER_FILENAME,
    TrialEvent,
    TrialLedgerIntegrityError,
    TrialLedgerStore,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

WORKERS = ("alpha", "beta")
EVENTS_PER_WORKER = 25


class _OptionalFieldRecord(BaseModel):
    """A record whose top-level field may be ``None``.

    It separates "the field is present and is ``None``" from "the field was dropped during
    serialization": the two read back differently, and only the first lets the read side
    recompute the same hash.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    x: str | None = None


def _event(store: TrialLedgerStore, index: int) -> TrialEvent:
    """Append one minimal record with a sequence number and return the event that was written."""
    return store.append(
        "search_trial", {"id": f"trial:{index}", "index": index}, actor="test"
    )


def _lines(path: Path) -> list[bytes]:
    """The file's non-empty lines, byte for byte."""
    return [line for line in path.read_bytes().split(b"\n") if line]


def _rewrite(path: Path, lines: list[bytes]) -> None:
    """Overwrite the file with the given lines, keeping the "every line ends with a newline" shape."""
    path.write_bytes(b"\n".join(lines) + b"\n")


def _append_worker_script(root: Path, worker: str, count: int) -> str:
    """One subprocess's script: append ``count`` events to the same root.

    ``sys.path`` is corrected explicitly rather than relying on ``python -c`` happening to
    run with the repository root as its cwd.
    """
    return "\n".join(
        [
            "import sys",
            f"sys.path.insert(0, {str(REPO_ROOT)!r})",
            "from autotuner.research.trial_ledger import TrialLedgerStore",
            # The timeout is generous: this case is about "two writers do not reuse a
            # sequence number", not about "the lock waits long enough".
            f"store = TrialLedgerStore({str(root)!r}, lock_timeout_s=30.0)",
            f"for index in range({count}):",
            f"    store.append('search_trial', {{'id': '{worker}-' + str(index)}}, actor='{worker}')",
        ]
    )


# --- The chain: intact, tampered with, dropped from -------------------------------


def test_appended_events_form_a_verified_hash_chain(tmp_path):
    store = TrialLedgerStore(tmp_path)
    _event(store, 1)
    _event(store, 2)
    _event(store, 3)

    events = store.events(verify=True)
    assert len(events) == 3
    assert tuple(event.sequence for event in events) == (1, 2, 3)
    # The head of the chain has no predecessor, so it is the empty string rather than some
    # event's hash.
    assert events[0].previous_event_hash == ""
    assert tuple(event.previous_event_hash for event in events[1:]) == tuple(
        event.event_hash for event in events[:-1]
    )


def test_a_tampered_line_is_refused(tmp_path):
    store = TrialLedgerStore(tmp_path)
    for index in (1, 2, 3):
        _event(store, index)

    lines = _lines(store.events_path)
    second = json.loads(lines[1])
    second["payload"]["id"] = "trial:9"  # a one-character change
    lines[1] = json.dumps(second, sort_keys=True).encode("utf-8")
    _rewrite(store.events_path, lines)

    with pytest.raises(TrialLedgerIntegrityError) as caught:
        store.events(verify=True)
    assert caught.value.code == "hash_mismatch"


def test_a_dropped_line_is_refused(tmp_path):
    store = TrialLedgerStore(tmp_path)
    for index in (1, 2, 3):
        _event(store, index)

    lines = _lines(store.events_path)
    _rewrite(store.events_path, [lines[0], lines[2]])

    with pytest.raises(TrialLedgerIntegrityError) as caught:
        store.events(verify=True)
    assert caught.value.code == "sequence_gap"


# --- Blank lines: tolerated, but not a place to hide a removal --------------------

BLANK_LAYOUTS = [
    pytest.param(0, 0, 1, b"", id="blank-line-at-end-of-file"),
    pytest.param(0, 0, 1, b"   ", id="whitespace-only-line-at-end-of-file"),
    pytest.param(0, 1, 0, b"", id="blank-line-between-two-events"),
    pytest.param(1, 0, 0, b"", id="blank-line-before-the-first-event"),
    pytest.param(4, 3, 4, b"", id="several-consecutive-blank-lines"),
    pytest.param(2, 2, 2, b"\t", id="several-consecutive-whitespace-lines"),
]


def _blank_filled(
    store: TrialLedgerStore, *, leading: int, middle: int, trailing: int, blank: bytes
) -> None:
    """Rewrite the store's two-event ledger with ``blank`` lines before, between and after its events."""
    events = _lines(store.events_path)
    assert len(events) == 2
    _rewrite(
        store.events_path,
        [blank] * leading + [events[0]] + [blank] * middle + [events[1]] + [blank] * trailing,
    )


@pytest.mark.parametrize(("leading", "middle", "trailing", "blank"), BLANK_LAYOUTS)
def test_a_blank_line_does_not_advance_the_sequence(tmp_path, leading, middle, trailing, blank):
    """A blank line carries no event, so it must not use up a sequence number.

    The write side computes "the next sequence" from the last *event* it can find (see
    ``_last_complete_line``), not from the number of physical lines.  A reader that counted
    lines would disagree with the writer about every event after the first blank: the append
    succeeds, the numbers it wrote are not the numbers the read side expects, and the ledger
    -- which looks perfectly healthy -- fails its own verification forever.  Blank lines are
    cheap to tolerate: what proves a record was not altered is the hash chain, not the line
    count.
    """
    store = TrialLedgerStore(tmp_path)
    _event(store, 1)
    second = _event(store, 2)
    _blank_filled(store, leading=leading, middle=middle, trailing=trailing, blank=blank)

    # The next append continues the chain from the last real event, not from the last line.
    third = _event(store, 3)
    assert third.sequence == second.sequence + 1
    assert third.previous_event_hash == second.event_hash

    events = store.events(verify=True)
    assert [event.sequence for event in events] == [1, 2, 3]
    assert [event.record_id for event in events] == ["trial:1", "trial:2", "trial:3"]

    records = store.records("search_trial")
    assert set(records) == {
        "search_trial:trial:1",
        "search_trial:trial:2",
        "search_trial:trial:3",
    }
    assert [records[f"search_trial:trial:{index}"]["index"] for index in (1, 2, 3)] == [1, 2, 3]


def test_a_blank_line_that_replaces_an_event_is_still_a_removal(tmp_path):
    """Tolerating a blank line must not tolerate a blank where an event used to be.

    Emptying the middle of three lines is a deletion, and the ledger has to say so.  The
    code that fires is ``sequence_gap`` rather than ``chain_break``: ``_read_events``
    numbers the events it actually reads and checks that sequence before it ever looks at
    the predecessor hash, and the surviving third line still carries sequence 3 while it is
    only the second event read -- so the sequence check, which runs first, is the one that
    reports it.
    """
    store = TrialLedgerStore(tmp_path)
    for index in (1, 2, 3):
        _event(store, index)

    lines = _lines(store.events_path)
    _rewrite(store.events_path, [lines[0], b"", lines[2]])

    with pytest.raises(TrialLedgerIntegrityError) as caught:
        store.events(verify=True)
    assert caught.value.code == "sequence_gap"


# --- Appending costs nothing that depends on file length --------------------------


def test_append_does_not_reread_the_whole_file(tmp_path, monkeypatch):
    """Structural proof of the O(1) claim: over ten appends the read side is never entered.

    A counter rather than a stopwatch: elapsed time moves with machine load, while "did the
    call happen" is a structural fact.
    """
    reads: list[bool] = []
    original = TrialLedgerStore._read_events

    def counting(self, verify: bool = True):
        reads.append(verify)
        return original(self, verify=verify)

    monkeypatch.setattr(TrialLedgerStore, "_read_events", counting)

    store = TrialLedgerStore(tmp_path)
    for index in range(10):
        _event(store, index)
    assert len(reads) == 0


# --- The torn last line -----------------------------------------------------------


def test_a_torn_tail_is_dropped_and_visible(tmp_path):
    """A half-line must be recognised, and the next append must clear it rather than write after it.

    Recognition uses the missing trailing newline: once a single-line file is cut in half,
    ``len(splitlines())`` is still 1, exactly as in the complete shape, so the line count is
    blind here.
    """
    store = TrialLedgerStore(tmp_path)
    _event(store, 1)

    raw = store.events_path.read_bytes()
    assert raw.endswith(b"\n") is True
    assert len(raw.splitlines()) == 1

    # The first append died halfway through writing this line: only its first half is in the
    # file, with no trailing newline.
    store.events_path.write_bytes(raw[: len(raw) // 2])
    torn = store.events_path.read_bytes()
    assert torn.endswith(b"\n") is False
    # Same line count as the complete shape: the count can prove neither completeness nor
    # truncation.
    assert len(torn.splitlines()) == 1
    assert store.summary()["incomplete_tail_bytes"] > 0

    # The write side heals itself: the leftover bytes are cut off and the new event lands
    # normally.
    _event(store, 2)
    assert store.summary()["incomplete_tail_bytes"] == 0
    events = store.events(verify=True)
    assert [event.sequence for event in events] == [1]
    assert events[-1].record_id == "trial:2"

    # The read side never feeds a half-line to json.loads: it reports "the last line is
    # incomplete" rather than a bracket error.
    healed = store.events_path.read_bytes()
    store.events_path.write_bytes(healed + b'{"half":')
    with pytest.raises(TrialLedgerIntegrityError) as caught:
        store.events(verify=True)
    assert caught.value.code == "torn_tail"


# --- Two writers ----------------------------------------------------------------


def test_two_processes_appending_do_not_produce_duplicate_sequences(tmp_path):
    """Two processes write 25 each: the read-back must hold exactly 50, with no sequence number repeated and none missing.

    An in-process lock is bound to fail here: both sides compute the sequence number from
    the same "last event", so both write a 1, and from then on every read fails at the chain
    check -- the ledger is permanently unreadable rather than merely short one record.
    """
    root = tmp_path / "ledger"
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", _append_worker_script(root, worker, EVENTS_PER_WORKER)],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        for worker in WORKERS
    ]
    for worker in workers:
        _, errors = worker.communicate(timeout=300)
        assert worker.returncode == 0, errors

    events = TrialLedgerStore(root).events(verify=True)
    assert len(events) == len(WORKERS) * EVENTS_PER_WORKER
    assert tuple(event.sequence for event in events) == tuple(
        range(1, len(WORKERS) * EVENTS_PER_WORKER + 1)
    )


# --- Location and signature -------------------------------------------------------


def test_the_store_requires_an_explicit_root():
    """root is required and has no default: a default path, once it exists, will have somebody depending on it."""
    parameter = inspect.signature(TrialLedgerStore.__init__).parameters["root"]
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        TrialLedgerStore()  # type: ignore[call-arg]


def test_trial_events_live_in_their_own_file(tmp_path):
    store = TrialLedgerStore(tmp_path)
    assert TRIAL_LEDGER_FILENAME == "trials.jsonl"
    assert store.events_path.name == TRIAL_LEDGER_FILENAME
    assert store.events_path == tmp_path / "trials.jsonl"
    # The research ledger already occupies events.jsonl: sharing the name would write both
    # kinds of event into one file.
    assert store.events_path.name != "events.jsonl"


# --- Fields with no JSON form -----------------------------------------------------


def test_a_non_json_payload_is_refused(tmp_path):
    """A set has no JSON form, so another process cannot compute its hash: it must be refused before it lands on disk.

    The exception type has to be ``ValueError`` itself -- not ``TypeError``, and not
    pydantic's ``ValidationError`` (which also subclasses ``ValueError``, hence the exact
    type comparison) -- and the message has to name the offending field, so the caller knows
    what to change.
    """
    store = TrialLedgerStore(tmp_path)

    with pytest.raises(ValueError) as caught:
        store.append("search_trial", {"id": "trial:bad", "tags": {1, 2}}, actor="test")
    assert caught.value.__class__ is ValueError
    assert "tags" in str(caught.value)

    # A nested set is refused the same way, and the path it reports still carries the field
    # name.
    with pytest.raises(ValueError) as nested:
        store.append(
            "search_trial",
            {"id": "trial:bad", "nested": {"tags": {1, 2}}},
            actor="test",
        )
    assert nested.value.__class__ is ValueError
    assert "tags" in str(nested.value)

    # The refusal happens before the write: the ledger holds not one record.
    assert store.events(verify=True) == ()


# --- Round trip of a value --------------------------------------------------------


def test_none_fields_survive_the_round_trip(tmp_path):
    """``None`` is a value, not "this field is absent".

    Every hash is computed over the result of ``model_dump(mode="json")``; hashing the model
    directly goes through ``_dump``'s ``exclude_none``, so the record read back is not the
    same thing as the one written and the hash does not match either.
    """
    store = TrialLedgerStore(tmp_path)
    record = _OptionalFieldRecord(id="trial:none")
    store.append("search_trial", record, actor="test")

    read_back = store.latest("search_trial", "trial:none")
    assert read_back is not None
    assert "x" in read_back
    assert read_back["x"] is None
    assert content_hash(record.model_dump(mode="json")) == content_hash(read_back)


def test_the_event_envelope_round_trips(tmp_path):
    """The envelope serializes and validates back unchanged: the premise for both ledgers sharing one chain mental model."""
    store = TrialLedgerStore(tmp_path)
    event = _event(store, 1)

    again = TrialEvent.model_validate_json(canonical_json(event))
    assert again.event_hash == event.event_hash
    assert again.model_dump(mode="json") == event.model_dump(mode="json")
    assert store.events(verify=True)[0].event_hash == again.event_hash


def test_an_unknown_field_in_an_event_line_is_refused(tmp_path):
    """``extra="forbid"``: a line carrying a field nobody declared must be an error, not a line read with the field dropped.

    A ledger is forensic material, and a field silently dropped while reading is exactly what
    nobody will ever notice: the record that comes back is not the record the file holds.  The
    refusal is ``bad_event`` rather than ``hash_mismatch`` because the envelope is validated
    before its hash is recomputed, so the first thing wrong with the line is what is reported.
    """
    store = TrialLedgerStore(tmp_path)
    _event(store, 1)

    raw = json.loads(_lines(store.events_path)[0])
    raw["note"] = "a field the envelope never declared"
    _rewrite(store.events_path, [json.dumps(raw, sort_keys=True).encode("utf-8")])

    with pytest.raises(TrialLedgerIntegrityError) as caught:
        store.events(verify=True)
    assert caught.value.code == "bad_event"


def test_the_record_read_back_is_the_last_version_written(tmp_path):
    """Two events under one id are a supersede; the caller reading that id back must get the second.

    Folding first-wins would hand back a record the search has already moved past, and the
    caller could not tell -- the ledger would look intact while answering with history.
    ``records`` and ``latest`` are two readings of the same fold and have to agree.
    """
    store = TrialLedgerStore(tmp_path)
    store.append("search_trial", {"id": "trial:same", "status": "running"}, actor="test")
    store.append("search_trial", {"id": "trial:same", "status": "done"}, actor="test")

    assert store.latest("search_trial", "trial:same")["status"] == "done"
    assert store.records("search_trial")["search_trial:trial:same"]["status"] == "done"
    assert [event.sequence for event in store.events(verify=True)] == [1, 2]


# --- A line the JSON scanner itself cannot handle ----------------------------------


def test_a_deeply_nested_line_is_reported_not_raised(tmp_path):
    """``json.loads`` does not only raise ``JSONDecodeError``: deep enough nesting hits the recursion limit first.

    ``RecursionError`` escaping the one place that turns a bad line into a typed error is
    worst exactly where it happens, because ``summary`` is the thing an operator runs when
    nothing else works -- its docstring promises it never raises, and its whole purpose is to
    describe a ledger that is already broken.  An uncaught scanner error there is not a wrong
    answer, it is no answer, from the only tool that was supposed to give one.

    This module's own writer cannot produce such a line; a hand-edited or externally
    concatenated one can, which is the case the read side exists to judge.
    """
    store = TrialLedgerStore(tmp_path)
    _event(store, 1)
    with store.events_path.open("ab") as handle:
        handle.write(b"[" * 5000 + b"\n")

    with pytest.raises(TrialLedgerIntegrityError) as caught:
        store.events(verify=True)
    assert caught.value.code == "bad_json"

    # The reporting path, which documents "never raises", still answers -- and counts the
    # lines it can see rather than giving up on the file.
    report = store.summary()
    assert "is not JSON" in report["integrity_error"]
    assert report["chain_verified"] is False
    assert report["event_count"] == 2


# --- The tail window is a window, not a limit on a line ---------------------------


def test_a_line_longer_than_the_tail_window_is_still_the_predecessor(tmp_path):
    """A record larger than the read window must still be found, or the chain quietly restarts at 1.

    The append reads only the last ``tail_window_bytes`` of the file, and the point of the
    doubling loop is that "the window holds no line start" is not the same as "there is no
    event": the window grows until the start of the last line is inside it.  Without that, a
    record larger than 8 KiB comes back as a fragment, and the append either fails on the
    fragment or -- worse -- starts a second chain at sequence 1 with an empty predecessor,
    after which no reader will ever accept the file again.
    """
    store = TrialLedgerStore(tmp_path)  # the default 8 KiB window
    blob = "x" * (DEFAULT_TAIL_WINDOW_BYTES * 2)
    first = store.append("search_trial", {"id": "trial:long", "blob": blob}, actor="test")
    assert store.events_path.stat().st_size > DEFAULT_TAIL_WINDOW_BYTES

    second = _event(store, 2)

    assert second.sequence == first.sequence + 1
    assert second.previous_event_hash == first.event_hash
    assert [event.sequence for event in store.events(verify=True)] == [1, 2]


# --- The precondition: a caller's check, at the moment of the write -----------------


def test_a_refusing_precondition_leaves_the_ledger_untouched_and_the_lock_free(tmp_path):
    """A caller's "may this event be written?" answer is only worth anything if it is given where the write is.

    The search tracker asks the same question -- is the run still open? -- before every trial
    write, and asks it against what it last saw.  What it last saw can be stale by the time the
    append lands, and an event that should have been refused then becomes a ledger the read
    side will not fold.  So the check is handed to ``append`` and run inside the lock, where
    nothing can slip in between: an exception from it is the caller's own (not wrapped here),
    no byte of the event reaches the file, and the lock is released on the way out.

    "Inside the lock" is pinned rather than assumed: the precondition reads the lock file and
    finds this process's pid in the note the lock writes when it is taken.  A check that ran
    before the lock was even opened would find no note, or somebody else's.  Read through the
    store's own accessor, because that is the only way to read the note while holding the lock
    -- a plain read of the file starts at the locked byte and on Windows is refused even for
    the holder's own second handle.
    """
    store = TrialLedgerStore(tmp_path)
    first = _event(store, 1)

    class _Refused(Exception):
        pass

    def refuse() -> None:
        note = store._read_note()
        assert f"pid={os.getpid()}" in note and "took=" in note
        raise _Refused("the run is closed")

    with pytest.raises(_Refused):
        store.append(
            "search_trial",
            {"id": "trial:2"},
            actor="test",
            precondition=refuse,
        )

    assert store.events(verify=True) == (first,)  # nothing was written behind the refusal
    assert _event(store, 2).sequence == 2  # and the lock is free again


def test_a_precondition_that_passes_writes_the_event_normally(tmp_path):
    """The other half: a check that says yes must not cost anything or change what is written.

    Cheap to get wrong in the obvious way -- running the precondition but then returning early,
    or hashing the record after it ran -- so the event's own fields are what this pins.
    """
    store = TrialLedgerStore(tmp_path)
    seen: list[object] = []
    event = store.append(
        "search_trial",
        {"id": "trial:1"},
        actor="test",
        precondition=lambda: seen.append(store.last_event()),
    )

    assert seen == [None]  # it can read the ledger from inside the section
    assert event.sequence == 1
    assert store.events(verify=True) == (event,)


# --- The write lock: what "held" means, and that waiting ends ----------------------


def _spawn_lock_holder(root: Path, seconds: float) -> tuple[subprocess.Popen, int]:
    """Start a process that takes the ledger's write lock, and return it once it is inside.

    The holder enters the lock through the store's own context manager rather than
    re-implementing the primitive, so what the parent contends with is exactly what any
    other writer contends with.  The line it prints is the signal the parent waits for: "the
    lock is held" has to be observed, not assumed, or the parent can win the race and every
    test built on this would pass without testing anything.
    """
    script = "\n".join(
        [
            "import os, sys, time",
            f"sys.path.insert(0, {str(REPO_ROOT)!r})",
            "from autotuner.research.trial_ledger import TrialLedgerStore",
            f"store = TrialLedgerStore({str(root)!r})",
            "with store._write_lock():",
            "    print('held', os.getpid(), flush=True)",
            f"    time.sleep({seconds!r})",
        ]
    )
    worker = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert worker.stdout is not None
    line = worker.stdout.readline().split()
    if line[:1] != ["held"]:
        worker.kill()
        errors = worker.stderr.read() if worker.stderr is not None else ""
        pytest.fail(f"the lock holder never took the lock: {errors!r}")
    return worker, int(line[1])


def test_the_lock_file_outlives_the_writers_that_use_it(tmp_path):
    """The lock file is created once and kept; what a writer leaves in it is a note, not the lock.

    Deleting a file is the one thing a lock on a path cannot do safely: a waiter that took the
    lock in the moment between the holder's check and its unlink has its lock deleted, and
    then two writers are inside at once.  This design does not delete it at all, so there is no
    moment to lose -- and the two facts that would both change if somebody reintroduced the
    unlink are that the file is still there after an append, and that what it says after the
    append is that the holder left rather than that it is still holding.
    """
    store = TrialLedgerStore(tmp_path)
    assert _event(store, 1).sequence == 1

    assert store.lock_path.exists()
    note = store.lock_path.read_bytes()
    assert note.startswith(b"LCK ")  # the preamble, then the note past the locked byte
    assert b"released=" in note


def test_a_lock_held_by_another_process_times_out_and_names_its_holder(tmp_path):
    """The wait has to end, and the verdict has to say something the reader can act on.

    A caller that gets ``lock_timeout`` is being told to try again, which is only a useful
    instruction if the holder is expected to finish.  The pid and the time in the message are
    what let a person tell "somebody else is writing" from "a process is wedged inside the
    section" without reaching for a debugger.

    It also pins the one structural choice in the note: it is written *past* the byte the lock
    covers.  A note written over that byte cannot be read by the very waiter that needs it, on
    Windows, where a read overlapping a locked range fails outright.
    """
    store = TrialLedgerStore(tmp_path, lock_timeout_s=0.25)
    worker, pid = _spawn_lock_holder(tmp_path, 30.0)
    try:
        started = time.monotonic()
        with pytest.raises(TrialLedgerIntegrityError) as caught:
            _event(store, 1)
        elapsed = time.monotonic() - started
    finally:
        worker.kill()
        worker.wait()

    assert caught.value.code == "lock_timeout"
    assert elapsed >= 0.25
    assert f"pid={pid}" in str(caught.value)
    assert "took=" in str(caught.value)
    assert not store.events_path.exists()  # nothing was written behind the refusal


def test_a_killed_holder_does_not_block_the_next_writer(tmp_path):
    """The reason the lock is a record and not a file: nobody has to notice that the holder died.

    A writer killed inside its critical section -- a scheduler reaping it, a person with
    ``kill -9``, a machine that took the job down with it -- never releases anything itself.
    With a file for a lock, the next writer's only way forward is to guess from age that the
    holder is gone, and that guess is what hands a slow-but-alive holder's section to somebody
    else.  Here the kernel closes the fd, which releases the lock, and the next append simply
    proceeds.  The holder is killed *while it is inside the section*, so this is the shape it
    claims to be.
    """
    store = TrialLedgerStore(tmp_path, lock_timeout_s=5.0)
    worker, _ = _spawn_lock_holder(tmp_path, 30.0)
    worker.kill()
    worker.wait()

    started = time.monotonic()
    event = _event(store, 1)

    assert event.sequence == 1
    assert time.monotonic() - started < 5.0  # the timeout is the failure mode this replaces
    assert [item.record_id for item in store.events(verify=True)] == ["trial:1"]


def test_two_threads_of_one_process_cannot_hold_it_at_once(tmp_path):
    """No ``threading.Lock`` is layered on the record lock, so the record lock has to exclude one process from itself.

    That is a property of the primitives, not of this module, and it differs by platform:
    ``flock`` is per open file description and ``msvcrt.locking`` is per handle, so two
    separate opens conflict in both cases -- measured, and pinned here so that a future
    replacement primitive has to hold the same line.  If it did not, two threads appending
    concurrently would write the same sequence number and the ledger would stop being
    readable, which is the failure this whole section exists to prevent.
    """
    store = TrialLedgerStore(tmp_path, lock_timeout_s=0.15)
    inside = threading.Event()
    leave = threading.Event()

    def hold() -> None:
        with store._write_lock():
            inside.set()
            leave.wait(10.0)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert inside.wait(10.0)
        with pytest.raises(TrialLedgerIntegrityError) as caught:
            store.append("search_trial", {"id": "trial:2"}, actor="probe")
        assert caught.value.code == "lock_timeout"
    finally:
        leave.set()
        holder.join(10.0)

    assert not holder.is_alive()
    assert _event(store, 1).sequence == 1  # and the lock is free again afterwards


def test_a_lock_path_that_cannot_be_used_is_refused_at_once(tmp_path):
    """Two codes, because they call for opposite reactions: ``lock_timeout`` says try again, ``lock_unavailable`` says do not.

    A lock path that cannot be opened as a file at all -- a directory where the lock belongs, a
    parent that is a file, a read-only mount -- is not the same situation as somebody else
    writing, and waiting cannot change it.  Retrying is what the timeout does, so a wait here
    would report "somebody else has it" about a ledger nobody is using; instead the open is
    attempted once and the append is refused immediately.
    """
    store = TrialLedgerStore(tmp_path, lock_timeout_s=5.0)
    store.root.mkdir(parents=True, exist_ok=True)
    store.lock_path.mkdir()

    started = time.monotonic()
    with pytest.raises(TrialLedgerIntegrityError) as caught:
        _event(store, 1)
    elapsed = time.monotonic() - started

    assert caught.value.code == "lock_unavailable"
    assert elapsed < 1.0  # not the timeout: the open failed once and that was the answer
    assert not store.events_path.exists()


# --- last_event(): the tail read, on its own --------------------------------------


def test_last_event_does_not_read_the_ledger(tmp_path, monkeypatch):
    """The question "what was written last" is asked before every trial write, so answering it must not cost the file's length.

    The search tracker asks it to find out whether the search has been closed, and asking
    through ``events()`` would re-verify the whole chain on every single write -- one search
    costing a square of its own ledger, which is the shape the append path exists to avoid.
    Pinned structurally, like the append: a counter over the read side, not a stopwatch.
    """
    reads: list[bool] = []
    original = TrialLedgerStore._read_events

    def counting(self, verify: bool = True):
        reads.append(verify)
        return original(self, verify=verify)

    monkeypatch.setattr(TrialLedgerStore, "_read_events", counting)

    store = TrialLedgerStore(tmp_path)
    assert store.last_event() is None  # no file at all
    written = _event(store, 1)

    assert store.last_event() == written
    assert reads == []


def test_last_event_ignores_a_torn_tail_and_refuses_a_tampered_one(tmp_path):
    """What the tail read returns has to be trustworthy by itself: the caller chains the next event onto it.

    Two shapes, and they must not be confused with each other.  A half-written line is a
    crash, not an event -- the append heals it, so returning it would be wrong and raising
    would wedge a healthy ledger.  A line whose content no longer matches its own hash is
    tampering, and returning it would extend the chain from a line the read side rejects:
    the ledger stops being readable rather than merely short.
    """
    store = TrialLedgerStore(tmp_path)
    first = _event(store, 1)
    assert store.last_event() == first

    # A writer died mid-line.  The fragment is not an event, and it is not an error either.
    intact = store.events_path.read_bytes()
    store.events_path.write_bytes(intact + b'{"sequence": 2, "record_')
    assert store.last_event() == first

    # A hand-edit that changed a field without recomputing the hash.  Only the intact line is
    # rewritten, so the fragment above cannot be mistaken for part of the tampered event.
    _rewrite(store.events_path, [intact.rstrip(b"\n").replace(b'"trial:1"', b'"trial:9"')])
    with pytest.raises(TrialLedgerIntegrityError) as caught:
        store.last_event()
    assert caught.value.code == "hash_mismatch"
