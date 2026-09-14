"""Append-only persistence for a search's trials: one record per line, verifiable, crash-judgeable.

There is not one line of hyperparameter code in this module.  It answers a single
question: how to append a record whose content is already decided to a file, so that
whoever reads it back can show it was not altered, dropped, or truncated.  The search
semantics (``TrialRecord`` / ``SearchTracker``) live in ``hyperparameter_search``, and
both sides share one envelope, so "the event chain" has exactly one mental model.

Why not reuse ``ResearchLedgerStore``.  Its ``append`` is "read everything -> ``sequence
= len + 1`` -> append" and its lock is an in-process ``threading.Lock``, so two processes
appending concurrently write duplicate sequence numbers and every later read fails
forever at the chain check; each append costs O(file length), so N appends cost O(N^2);
and reusing it would mean adding branches to its ``RECORD_MODELS`` and
``validate_record``, which is editing a shared module of an already-finished task.  What
is reused from it are the pure functions and the conventions -- ``content_hash`` /
``canonical_json`` / ``utc_now`` and ``LedgerEvent``'s way of recomputing the chain --
not its store.

Three mechanisms:

* **O(1) tail read.**  Appending reads only a window at the end of the file, to take the
  last complete event and derive the sequence number and the predecessor hash from it.
  Full-chain verification happens on the read side only: verifying everything on every
  append would grow with the file just like the file itself does.  The window doubles when
  it opens inside the last line, so "O(1)" is really "O(the last line)": still independent
  of how many events precede it, which is what makes one search linear rather than
  quadratic.
* **Cross-process lock.**  ``<root>/.write.lock`` carries an advisory record lock --
  ``fcntl.flock`` where there is one, ``msvcrt.locking`` on Windows -- and the critical
  section covers only "read the tail + append + fsync": it never spans a training run or
  any subprocess call.  A record lock excludes two threads of one process as well, so no
  additional ``threading.Lock`` is needed, and the kernel releases it when the holder dies,
  so a process killed inside its critical section leaves nothing behind to recover.  The
  lock file is never deleted, and what it contains is a note for a person, never an input
  to a decision.  (An earlier version created and unlinked the file instead; ``_write_lock``
  records why that shape cannot be made correct.)
* **A torn tail heals itself.**  A process that dies mid-write leaves half a line.  The
  write side truncates everything after the last newline while holding the lock; the read
  side reports those leftover bytes as an error instead of feeding them to ``json.loads``.
  Deciding that must look at the final newline: a complete line and a torn one both have
  ``len(splitlines()) == 1``, so a line count cannot tell them apart.

And one thing it *tolerates*, which the write and read sides have to agree on: a blank
line.  No writer here produces one, but a file that has one -- somebody's editor, a tool
that appended a newline, a hand-edit -- must not be read as "there is no event", because
the append that follows would then be numbered 1 again and leave a file whose sequence
numbers repeat.  So the last non-blank line is the predecessor, and blank lines do not
count toward the sequence: the sequence counts events, not physical lines.  That costs no
tamper detection -- replacing a real line with a blank still removes an event from the
chain, and the next event's position no longer matches its sequence.

Hashes are always taken over the result of ``model_dump(mode="json")``, never by handing a
``BaseModel`` to ``content_hash``: the latter goes through ``_dump``, which passes
``exclude_none=True`` for a model, so a field whose value is ``None`` is silently dropped
and the record hashes differently after a round trip than it did when it was written.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping

from pydantic import BaseModel, ConfigDict, ValidationError

from .research_ledger import LedgerEvent, canonical_json, content_hash, utc_now

# The one platform difference this module has.  Both take an exclusive lock on a byte range
# of an open fd and hand it back when the fd is closed -- including when the closing is done
# by the kernel because the process died, which is what lets the write lock stop guessing
# whether a holder is gone.
if os.name == "nt":
    import msvcrt
else:
    import fcntl


TRIAL_LEDGER_SCHEMA_VERSION = "rl-agent.trial-ledger/v1"

# Not ``events.jsonl``: ``research_ledger`` already owns that name, and sharing it would
# let a caller pointed at the research ledger's directory write two kinds of event into
# one file, each breaking the other's hash chain.
TRIAL_LEDGER_FILENAME = "trials.jsonl"

WRITE_LOCK_FILENAME = ".write.lock"

# ``O_BINARY`` on Windows: the note written into the lock file has to come back byte for
# byte, and a handle in text mode rewrites ``\n`` on the way out and back.
_LOCK_OPEN_FLAGS = os.O_CREAT | os.O_RDWR | (os.O_BINARY if os.name == "nt" else 0)

# Byte 0 is the byte the record lock covers; the note starts after it, so that reading the
# note never touches a byte somebody holds.  On Windows a read overlapping a locked range
# fails outright (measured), and a note the waiter cannot read is a note nobody has.
_LOCK_PREAMBLE = b"LCK "
_LOCK_NOTE_MAX_BYTES = 256

# How often a waiter that lost the race looks again.  Far below the cost of the section it
# is waiting for -- one append on this machine costs ~36ms, fsync included -- so the poll is
# not what a waiter waits on, and a five-second timeout is dozens of attempts, not thousands.
_LOCK_POLL_S = 0.02

DEFAULT_TAIL_WINDOW_BYTES = 8 * 1024

DEFAULT_LOCK_TIMEOUT_S = 5.0

IntegrityCode = Literal[
    "bad_event",
    "bad_json",
    "chain_break",
    "count_mismatch",
    "duplicate_close",
    "hash_mismatch",
    "lock_timeout",
    "lock_unavailable",
    "sequence_gap",
    "torn_tail",
    "trial_after_close",
    "unknown_layout",
]


class TrialLedgerIntegrityError(Exception):
    """The ledger does not hold together: a broken chain, an altered line, a truncated tail, a bad layout, or no write lock.

    It carries ``.code`` so a caller can act by category -- a torn tail can heal, a hash
    mismatch cannot -- instead of matching on words in the message.  The two lock codes are
    filed here too: "this ledger is unusable right now" is the same thing to a caller, and a
    second exception type would only mean one more ``except``.

    They are separate codes because they call for opposite reactions.  ``lock_timeout`` means
    somebody else was writing and this writer stopped waiting -- retrying is the documented
    response.  ``lock_unavailable`` means the lock path itself cannot be used (a directory
    where the file should be, a parent that is a file, a read-only mount), which no amount of
    retrying will change.
    """

    def __init__(self, code: IntegrityCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class TrialEvent(LedgerEvent):
    """The ledger envelope.

    The fields and the way the chain is recomputed follow ``LedgerEvent`` verbatim (the
    parent's twelve fields).  Two overrides: ``schema_version`` names the trial ledger,
    and ``extra="forbid"`` makes "someone wrote an extra field" an error at the moment it
    is read rather than something silently dropped -- a ledger is forensic material, and
    what it drops while reading is what nobody will ever notice.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = TRIAL_LEDGER_SCHEMA_VERSION


def _is_json_scalar(value: Any) -> bool:
    """Whether this is an atom JSON can express directly."""
    return value is None or isinstance(value, (bool, int, float, str))


def _locate_non_json(value: Any, path: str = "") -> tuple[str, str] | None:
    """Find the first leaf with no JSON form; return ``(dotted path, type name)``.

    Located here rather than by wrapping ``content_hash``'s ``TypeError``: that exception
    only says "Object of type set is not JSON serializable", never which field of the
    record it sat in.  A record nests, and a caller needs a path it can go and fix.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            found = _locate_non_json(item, f"{path}.{key}" if path else str(key))
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            found = _locate_non_json(item, f"{path}[{position}]")
            if found is not None:
                return found
        return None
    if _is_json_scalar(value):
        return None
    return (path or "<root>", type(value).__name__)


def _try_lock(handle: int) -> bool:
    """Take an exclusive lock on byte 0 of ``handle`` without waiting; ``False`` if it is held.

    Only "somebody else has it" is ``False``.  Where the platform tells the two apart, an
    error raised for any other reason -- no locks available on this filesystem, a handle
    that cannot be locked at all -- propagates instead of being read as contention: a waiter
    that can never take the lock is owed that verdict now, not after ``lock_timeout_s``
    spent pretending to contend.  ``msvcrt`` makes no such distinction (a held region and a
    broken handle are both ``EACCES``), so on Windows the two are one and it is the timeout
    that reports it.
    """
    os.lseek(handle, 0, os.SEEK_SET)
    if os.name == "nt":
        try:
            msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _rewrite_note(handle: int, note: str) -> None:
    """Replace everything in the lock file with ``_LOCK_PREAMBLE`` + ``note``.

    Called only while holding the lock, and never on a byte another process is reading:
    the note follows the preamble, and the preamble's fourth byte is the last one before the
    region nobody else may touch.
    """
    os.ftruncate(handle, 0)
    os.lseek(handle, 0, os.SEEK_SET)
    os.write(handle, _LOCK_PREAMBLE + note.encode("ascii"))


class TrialLedgerStore:
    """One directory holding one search's append-only event log.

    ``root`` is required and has no default.  ``ResearchLedgerStore``'s class default
    (``research_ledger.py:605``) and its own CLI default (``research_ledger.py:833``)
    already contradict each other, with the result that one repository can grow two
    ledgers -- once a default path exists, someone depends on it, and then nobody knows
    which copy is the real one.  That mistake is not repeated here, and the canonical
    directory is not this module's decision to make (docs/P4.4_task4_design.md section 11.3).

    One ``search_run`` record per root is a layout rule the read side enforces; the write
    side only has to append faithfully.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        tail_window_bytes: int = DEFAULT_TAIL_WINDOW_BYTES,
        lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
    ) -> None:
        if tail_window_bytes < 1:
            raise ValueError(f"tail_window_bytes must be at least 1, got {tail_window_bytes}")
        if lock_timeout_s < 0:
            # ``0`` is meaningful -- try once, never wait -- and negative is not.
            raise ValueError(f"lock_timeout_s must not be negative, got {lock_timeout_s}")
        self.root = Path(root)
        self.events_path = self.root / TRIAL_LEDGER_FILENAME
        self.lock_path = self.root / WRITE_LOCK_FILENAME
        self.tail_window_bytes = tail_window_bytes
        self.lock_timeout_s = lock_timeout_s

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        """Hold the ledger's write lock for the duration of the ``with`` body.

        Mutual exclusion across processes, and between threads of one process, is the
        kernel's job here: ``<root>/.write.lock`` is opened once and byte 0 of it is locked
        exclusively -- ``fcntl.flock`` where there is one, ``msvcrt.locking`` on Windows.
        The section covers "read the tail + append + fsync" and nothing else: it never spans
        a training run or a subprocess call, so nobody waits longer than the cost of one
        append.  Both primitives exclude two handles of a *single* process from each other
        as well (measured on both platforms), so no ``threading.Lock`` is layered on top.

        Why a record lock rather than "create the file, unlink it to release".  Creation is
        atomic, so two writers cannot both create it, but *release* is not: an unlink is one
        syscall on a path, and a waiter that creates its own lock in between has that lock
        deleted by the departing holder, which puts two writers in the section at once.
        Re-reading the file first narrows that window, it does not close it.  And a lost
        write lock does not cost a retry: two writers read the same tail, write the same
        sequence number, and from then on the ledger is unreadable for good -- the read
        side's chain check reports it, it does not repair it.  A crashed holder is the other
        half of the same problem: with a file for a lock, "the holder is gone" can only be
        guessed at from the file's age, so a holder that is alive but slow has its lock taken
        away underneath it.  A record lock answers both by not asking: the kernel hands it
        back when the fd closes, which it does when the process ends -- including a killed
        one, measured here under ``TerminateProcess`` and ``SIGKILL`` -- so there is no
        staleness to estimate, no token to compare, and no lock file to delete.  That last
        one is what leaves this class with no path-based race anywhere: the file is created
        once and never removed.

        A waiter polls instead of blocking, so that it can stop: the lock is only ever held
        for the length of one append, so still not having it after ``lock_timeout_s`` means
        the holder is stuck (an fsync on a dying mount) or wedged, and a caller is owed a
        verdict to act on rather than an unbounded block.  What the lock file holds -- the
        pid that last took it and when -- is a note for whoever reads that verdict, and
        nothing here reads it to make a decision.

        One thing this does not measure: on a network filesystem a record lock may be
        emulated by the client or ignored by the server, and the guarantee then degrades to
        whatever that mount gives.  Nothing in this repository has been run on one.
        """
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            handle = os.open(self.lock_path, _LOCK_OPEN_FLAGS)
        except OSError as exc:
            raise TrialLedgerIntegrityError(
                "lock_unavailable", f"{self.lock_path} could not be opened for locking: {exc}"
            ) from exc
        try:
            deadline = time.monotonic() + self.lock_timeout_s
            while True:
                try:
                    held = _try_lock(handle)
                except OSError as exc:
                    raise TrialLedgerIntegrityError(
                        "lock_unavailable", f"{self.lock_path} could not be locked: {exc}"
                    ) from exc
                if held:
                    break
                if time.monotonic() >= deadline:
                    note = self._read_note()
                    raise TrialLedgerIntegrityError(
                        "lock_timeout",
                        f"{self.lock_path} was still locked after {self.lock_timeout_s:.1f}s"
                        + (f" (the lock file says: {note})" if note else " (the lock file says nothing)"),
                    )
                time.sleep(_LOCK_POLL_S)
            self._note_holder(handle)
            try:
                yield
            finally:
                self._say_released(handle)
        finally:
            # Closing the fd is the release.  It happens even when the body raised, and even
            # when this whole process is killed, which is the property the file-based lock
            # could only approximate.
            os.close(handle)

    def _note_holder(self, handle: int) -> None:
        """Write down who took the lock, for whoever has to read a ``lock_timeout`` later.

        Best effort on purpose: no decision in this class reads the file's contents, so
        failing to write a note is not failing to lock, and the only thing lost is a
        diagnostic.  Written while the lock is held, so it names the holder and nobody else.
        """
        try:
            _rewrite_note(handle, f"pid={os.getpid()} took={time.time():.3f}")
        except OSError:
            pass

    def _say_released(self, handle: int) -> None:
        """Say that the holder left, so a later timeout does not blame it for somebody else's block.

        This matters for the honesty of the one message it feeds.  Between a holder's release
        and the next holder's note there is a gap of microseconds, and a waiter whose deadline
        lands in that gap would otherwise report a pid that has already finished as the reason
        it is still waiting.
        """
        try:
            _rewrite_note(handle, f"pid={os.getpid()} released={time.time():.3f}")
        except OSError:
            pass

    def _read_note(self) -> str:
        """What the lock file says, verbatim; ``""`` when there is nothing to read.

        Seeks past the locked byte rather than reading the file from the start.  On Windows a
        read that overlaps a locked range is refused -- to other processes *and* to the
        holder's own second handle (both measured) -- so a note written over byte 0 would be
        unreadable by exactly the two readers that want it: the waiter deciding what to say
        about the wait, and the holder checking that it is inside.
        """
        try:
            with open(self.lock_path, "rb") as stream:
                stream.seek(len(_LOCK_PREAMBLE))
                return stream.read(_LOCK_NOTE_MAX_BYTES).decode("ascii", errors="replace")
        except OSError:
            return ""

    def _tail_window(self, size: int, window: int) -> tuple[int, bytes]:
        """At most ``window`` bytes at the end of the file, with the offset they start at."""
        start = max(0, size - window)
        with self.events_path.open("rb") as handle:
            handle.seek(start)
            return (start, handle.read())

    def _incomplete_tail_bytes(self) -> int:
        """How many trailing bytes are not terminated by a newline; 0 means every line was written whole.

        Only a window at the end is read, doubling as needed: a torn line is always at the
        end, and reading the whole file to measure it would buy nothing.
        """
        if not self.events_path.is_file():
            return 0
        size = self.events_path.stat().st_size
        if size == 0:
            return 0
        window = self.tail_window_bytes
        while True:
            start, data = self._tail_window(size, window)
            if data.endswith(b"\n"):
                return 0
            index = data.rfind(b"\n")
            if index >= 0:
                return len(data) - index - 1
            if start == 0:
                return size
            window *= 2

    def _last_complete_line(self) -> bytes:
        """The last complete **and non-blank** line in the file, or empty bytes when there is no event.

        When the file does not end in a newline, that trailing fragment is discarded and
        the line before it is used.  If the window cut off the start of that line, there
        is no way to tell whether the line is complete, so the window doubles and the read
        is retried -- unless the window already reaches the start of the file.

        Blank complete lines are skipped rather than returned.  No writer here produces one,
        but a file that ends in one (somebody's editor, a hand-edit, a tool that appended a
        newline) used to be read as "there is no event": the next append then built a second
        event numbered 1 with an empty predecessor, returned success, and left a file whose
        sequence numbers repeat -- which nothing will ever read again.  Skipping back to the
        last real line turns that silent corruption into an ordinary append.
        """
        if not self.events_path.is_file():
            return b""
        size = self.events_path.stat().st_size
        if size == 0:
            return b""
        window = self.tail_window_bytes
        while True:
            start, data = self._tail_window(size, window)
            full_end = data.rfind(b"\n") + 1
            if full_end == 0:
                if start == 0:
                    return b""
                window *= 2
                continue
            body = data[:full_end]
            while True:
                if body.endswith(b"\n"):
                    body = body[:-1]
                index = body.rfind(b"\n")
                line = body[index + 1:]
                if line.strip():
                    if index < 0 and start > 0:
                        # The window opened *inside* this line, so its first byte may be a
                        # cut-off middle: what this returns would be a JSON fragment, and the
                        # write path feeds it to ``json.loads`` and reports the ledger's own
                        # last event as corrupt.  Every later append would fail on a file
                        # that is perfectly healthy and that the read side still verifies,
                        # which is the worst shape of failure this module can have.  Widen
                        # and read again -- the line's start is either inside the next window
                        # or at offset 0.  Only a line longer than the window can keep this
                        # going, and each round doubles it, so the loop ends.
                        break
                    return line
                if index < 0:
                    break
                body = body[:index]
            if start == 0:
                return b""
            window *= 2

    def _trim_incomplete_tail(self) -> int:
        """Drop the trailing bytes that no newline terminates; return how many were dropped.

        Must be called while holding the write lock: it reads the tail and then truncates,
        and no other writer may slip in between.  Truncating rather than appending after
        the fragment is the point -- that half line can never be completed into a valid
        event, and leaving it would only make the next read fail in the same place.
        """
        if not self.events_path.is_file():
            return 0
        size = self.events_path.stat().st_size
        if size == 0:
            return 0
        window = self.tail_window_bytes
        while True:
            start, data = self._tail_window(size, window)
            if data.endswith(b"\n"):
                return 0
            index = data.rfind(b"\n")
            if index >= 0:
                keep = start + index + 1
            elif start == 0:
                keep = 0
            else:
                window *= 2
                continue
            with self.events_path.open("r+b") as handle:
                handle.truncate(keep)
                handle.flush()
                os.fsync(handle.fileno())
            return size - keep

    def _parse_event(self, line: bytes, where: str) -> TrialEvent:
        """Parse one line into an event; say which line and why when it will not parse.

        ``RecursionError`` is caught alongside the two JSON errors because CPython's scanner
        raises it rather than ``JSONDecodeError`` once nesting is deep enough
        (``json.loads("[" * 5000)`` on 3.13).  It would otherwise escape the one place that
        turns a bad line into a typed error, so :meth:`summary` -- whose whole purpose is to
        describe a ledger that is already broken -- would crash instead of reporting which
        line is bad.  This module's own writer cannot produce such a line; a hand-edited or
        externally concatenated one can.
        """
        try:
            raw = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
            raise TrialLedgerIntegrityError("bad_json", f"{where} is not JSON: {error}") from error
        if not isinstance(raw, dict):
            raise TrialLedgerIntegrityError(
                "bad_event", f"{where} is a {type(raw).__name__}, not a trial event object"
            )
        try:
            return TrialEvent.model_validate(raw)
        except ValidationError as error:
            raise TrialLedgerIntegrityError("bad_event", f"{where} is not a trial event: {error}") from error

    def _read_tail_event(self) -> TrialEvent | None:
        """The last complete event, checking that it is self-consistent on the way.

        This one sha256 is cheap (a single line) and it closes the door on extending the
        chain from a tampered tail.  The full-chain check is not done here: that belongs
        to the read side, and recomputing every link on every append would grow with the
        file.
        """
        line = self._last_complete_line()
        if not line.strip():
            return None
        event = self._parse_event(line, "the last event line")
        body = event.model_dump(mode="json", exclude={"event_hash"})
        if event.event_hash != content_hash(body):
            raise TrialLedgerIntegrityError(
                "hash_mismatch",
                f"the last event (sequence {event.sequence}) does not match its own hash",
            )
        return event

    def append(
        self,
        record_type: str,
        record: BaseModel | Mapping[str, Any],
        *,
        actor: str,
        event_type: Literal["append", "supersede"] = "append",
        precondition: Callable[[], None] | None = None,
    ) -> TrialEvent:
        """Append one record as an event and return the event that was written.

        ``actor`` is required.  Who wrote a ledger line is not something that can be
        inferred, and giving it a default would make every caller share one signature, so
        nobody could later tell a tracker's write from a human's.

        ``record`` may be a model, but it is turned into ``model_dump(mode="json")``
        before it is hashed or written: handing the model straight to ``content_hash``
        goes through its ``exclude_none``, dropping fields whose value is ``None``, and
        then the record that comes back is not the record that went in.

        ``precondition`` is a callable run *inside* the lock, after the torn tail has been
        healed and before a byte is written: whatever it raises is what the caller sees, and
        the ledger is left exactly as it was.  It exists because a caller's "is this write
        still legal?" question has to be answered at the moment of the write to mean
        anything -- a check made before the call, or after the lock is dropped, can be
        overtaken by a writer that slips in between, and the ledger then holds an event the
        read side refuses to fold.  So the check belongs where the write is, not next to it.
        """
        payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else dict(record)
        broken = _locate_non_json(payload)
        if broken is not None:
            path, kind = broken
            raise ValueError(f"the {record_type} payload holds a {kind} at {path}, which has no JSON form")
        payload_hash = content_hash(payload)
        with self._write_lock():
            # Healing happens before the tail is read: whether the "last complete event"
            # that follows the leftover bytes really is the last one depends on having
            # removed them first.  It also happens before the precondition for the same
            # reason -- a caller reading the tail must see what this append will extend.
            self._trim_incomplete_tail()
            if precondition is not None:
                precondition()
            previous = self._read_tail_event()
            event = TrialEvent(
                event_id=f"event:{uuid.uuid4().hex}",
                sequence=previous.sequence + 1 if previous is not None else 1,
                event_type=event_type,
                record_type=record_type,
                record_id=str(payload.get("id") or ""),
                actor=actor,
                created_at=utc_now(),
                payload=payload,
                payload_hash=payload_hash,
                previous_event_hash=previous.event_hash if previous is not None else "",
                event_hash="",
            )
            event.event_hash = content_hash(event.model_dump(mode="json", exclude={"event_hash"}))
            self.root.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
                # canonical_json receives an already-dumped plain dict, which _dump passes
                # through untouched; handing it a model would drop the None fields.
                handle.write(canonical_json(event.model_dump(mode="json")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return event

    def _read_events(self, verify: bool = True) -> tuple[TrialEvent, ...]:
        """Read every line back.  O(file length).

        **The write path does not go through here**, so one append costs the same whether
        the file holds one event or a million.  That structural fact is guarded by
        acceptance case A4 with a call counter rather than by a timing.
        """
        if not self.events_path.is_file():
            return ()
        data = self.events_path.read_bytes()
        lines = data.split(b"\n")
        if lines[-1] == b"":
            lines.pop()
        else:
            # Feeding half a line to json.loads only yields an error about brackets,
            # which hides the real cause.
            raise TrialLedgerIntegrityError(
                "torn_tail",
                f"the file ends with {len(lines[-1])} bytes that are not a complete line: "
                f"the last append was interrupted",
            )
        events: list[TrialEvent] = []
        previous = ""
        index = 0
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                # A blank line carries no event, so it does not advance the sequence: the
                # sequence counts events, not physical lines.  The two are kept apart on
                # purpose -- the write side computes "the next sequence" from the last
                # *event* (see _last_complete_line), so a reader that counted blank lines
                # would disagree with the writer about the number of every event after the
                # first blank one.  A consistent append would then produce a file that this
                # very method rejects.  Tolerating the blank costs nothing: what proves a
                # record is intact is the hash chain, not the line count, and replacing a
                # real line with a blank removes an event from the chain -- the event after
                # it no longer sits at its own position, and the sequence check (which runs
                # before the chain check) reports that.  The one mutation a blank can hide --
                # a blank where the *last* line used to be -- is indistinguishable from
                # deleting the last line, which is invisible to any reader holding no
                # external record of the count.
                continue
            index += 1
            event = self._parse_event(line, f"line {line_number}")
            if verify:
                if event.sequence != index:
                    raise TrialLedgerIntegrityError(
                        "sequence_gap",
                        f"line {line_number} carries sequence {event.sequence}, not {index}",
                    )
                if event.previous_event_hash != previous:
                    raise TrialLedgerIntegrityError(
                        "chain_break",
                        f"line {line_number} does not follow the event before it: an event was "
                        f"removed or reordered",
                    )
                body = event.model_dump(mode="json", exclude={"event_hash"})
                if event.event_hash != content_hash(body):
                    raise TrialLedgerIntegrityError(
                        "hash_mismatch", f"line {line_number} does not match its own hash"
                    )
            previous = event.event_hash
            events.append(event)
        return tuple(events)

    def events(self, verify: bool = True) -> tuple[TrialEvent, ...]:
        """Read every event back.

        With ``verify=True`` each line is checked for a continuous sequence, a predecessor
        hash that connects to the line before, and a hash that matches its own content.
        Together those three turn "a line was deleted", "two lines were swapped" and "a
        line's content was changed" into discoverable errors.
        """
        return self._read_events(verify=verify)

    def last_event(self) -> TrialEvent | None:
        """The last complete event, from the tail of the file alone: O(1) in the file's length.

        This is the same read the append path performs, exposed because "what was written
        last" is a question a *writer* has to answer before it writes -- "has this ledger
        been closed" is the one that matters -- and answering it through :meth:`events` would
        make each write cost the length of everything written so far.

        The tail's own hash is checked, so a tampered last line raises rather than being
        returned.  ``None`` means there is no complete event yet (no file, an empty file, or
        one that holds nothing but a half-written line).
        """
        return self._read_tail_event()

    def _complete_line_count(self) -> int:
        """Number of complete, non-empty lines.  No parsing, no verification -- for a rough count once the ledger is already broken."""
        if not self.events_path.is_file():
            return 0
        data = self.events_path.read_bytes()
        if not data.endswith(b"\n"):
            index = data.rfind(b"\n")
            data = data[: index + 1] if index >= 0 else b""
        return sum(1 for line in data.split(b"\n") if line.strip())

    def records(self, record_type: str = "") -> dict[str, dict[str, Any]]:
        """The latest version of every record, folded by ``record_id``.

        Keys are ``f"{record_type}:{record_id}"`` rather than bare ``record_id``: when two
        record types happen to use the same id, a bare key would let the later one push
        the earlier one out without leaving a trace, and this module does not do that.
        """
        latest: dict[str, dict[str, Any]] = {}
        for event in self._read_events():
            latest[f"{event.record_type}:{event.record_id}"] = event.payload
        if not record_type:
            return latest
        prefix = f"{record_type}:"
        return {key: payload for key, payload in latest.items() if key.startswith(prefix)}

    def latest(self, record_type: str, record_id: str) -> dict[str, Any] | None:
        """The latest version of one record; ``None`` when nothing was ever written under that id."""
        for event in reversed(self._read_events()):
            if event.record_type == record_type and event.record_id == record_id:
                return event.payload
        return None

    def summary(self) -> dict[str, Any]:
        """The ledger's current state.  **Never raises**: being able to look at a broken ledger is one of the reasons it exists.

        A broken chain is reported honestly in ``integrity_error`` with ``chain_verified``
        left False, rather than raising so the caller cannot even see which step broke,
        and rather than pretending the ledger is intact.
        """
        report: dict[str, Any] = {
            "schema_version": TRIAL_LEDGER_SCHEMA_VERSION,
            "root": str(self.root.resolve()),
            "events_path": str(self.events_path.resolve()),
            "events_filename": TRIAL_LEDGER_FILENAME,
            "incomplete_tail_bytes": self._incomplete_tail_bytes(),
            "event_count": 0,
            "chain_verified": False,
            "integrity_error": "",
            "latest_event": "",
        }
        try:
            events = self._read_events(verify=True)
        except TrialLedgerIntegrityError as error:
            report["integrity_error"] = str(error)
            report["event_count"] = self._complete_line_count()
        else:
            report["event_count"] = len(events)
            report["chain_verified"] = True
            report["latest_event"] = events[-1].event_id if events else ""
        return report


__all__ = [
    "DEFAULT_LOCK_TIMEOUT_S",
    "DEFAULT_TAIL_WINDOW_BYTES",
    "IntegrityCode",
    "TRIAL_LEDGER_FILENAME",
    "TRIAL_LEDGER_SCHEMA_VERSION",
    "TrialEvent",
    "TrialLedgerIntegrityError",
    "TrialLedgerStore",
    "WRITE_LOCK_FILENAME",
]
