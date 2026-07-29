# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""One trainer window per library — the counterpart of ``app/instance.hpp``.

Starting the trainer from the game's TRAIN AI entry, from its desktop icon or
from a terminal used to open it again: each press another window, all watching
the same runs. The launcher cannot fix that — no portable way exists to bring
another process's window forward, and Wayland forbids it outright — but the
*launched* program can: a second trainer that finds a twin already serving the
same library hands its activation over and exits, and the twin raises itself,
which every platform allows. The game does the same for itself on the other
side (``app/instance.cpp``); the two never speak to each other's channel, so
nothing here is a wire format shared with C++.

The identity is the library directory, resolved: the trainer opened on the
default library is one thing however it was started, while a developer's
``python -m missile_defense.ui /some/other/runs`` is deliberately another and
still gets its own window.

Everything here is standard library on purpose. Qt would make the channel four
lines shorter, but this module runs *before* the decision to be a window at
all — the forwarding twin should cost milliseconds, not a PySide6 import — and
a channel with no Qt in it is testable on the machines CI actually has
(AGENTS.md: the gate installs no PySide6).

The channel is a ``multiprocessing.connection`` listener — a named pipe on
Windows, a Unix socket elsewhere — with one round trip on it: the server
greets with its pid, the client grants that pid the right to take the
foreground (Windows; everywhere else the greeting is just proof of life) and
answers ``activate``, the server raises its window. Peers get
:data:`_REPLY_DEADLINE` seconds before they are treated as absent: opening a
second window beats hanging a launch on a wedged process.
"""

from __future__ import annotations

import os
import sys
from multiprocessing import connection
from pathlib import Path
from threading import Event, Thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

#: Set to ``"0"`` to keep every launch a fresh window — the escape hatch for
#: tests and for anyone who genuinely wants two trainers on one library.
DISABLE_VARIABLE = "MD_SINGLE_INSTANCE"

#: How the server introduces itself; anything else on the endpoint is a
#: stranger whose protocol this is not, and is left alone.
_GREETING = b"md1"
_ACTIVATE = b"activate"

#: Seconds a peer gets to say its piece before it is treated as absent.
_REPLY_DEADLINE = 2.0


def engaged(environ: Mapping[str, str] | None = None) -> bool:
    """Whether single-instance behaviour is on at all."""
    environ = os.environ if environ is None else environ
    return environ.get(DISABLE_VARIABLE) != "0"


def trainer_key(library_dir: Path) -> str:
    """This launch's identity: the trainer *on this library* is one thing.

    Resolved, so the spelling cannot depend on who launched it: the game, the
    desktop entry and a shell in the checkout all mean the same library.
    """
    return "trainer\n" + str(Path(library_dir).resolve())


def _digest(text: str) -> str:
    """FNV-1a, 64 bit, as 16 hex digits.

    Not :func:`hash`, which is salted per process and would give the two
    processes that must agree two different names.
    """
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8", "surrogatepass"):
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def endpoint(
    key: str,
    *,
    user: str | None = None,
    runtime_dir: str | None = None,
    platform: str = sys.platform,
) -> str:
    """Where a launch with this ``key`` rendezvouses with its twin.

    The user is folded into the digest because the pipe namespace on Windows
    and ``/tmp`` on a bare Unix are machine-wide, and two people's trainers
    deduplicating against each other would be both wrong and a question of who
    may open whose window. ``user`` and ``runtime_dir`` are injected so this
    is testable as itself.
    """
    if user is None:
        user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    tag = _digest(user + "\0" + key)
    if platform == "win32":
        return rf"\\.\pipe\missile-defense-trainer-{tag}"
    if runtime_dir is None:
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    # Joined as text, not as a Path: `platform` decides the shape of the
    # address, and a Path object would quietly re-spell it for the machine
    # running this line — which is how a test asking the Linux question on a
    # Windows box got an answer full of backslashes.
    return f"{runtime_dir.rstrip('/')}/missile-defense-trainer-{tag}.sock"


def _grant_foreground(pid: int) -> None:
    """Windows: pass our right to take the foreground along to ``pid``.

    This process was just started by whatever the person is looking at — the
    game's menu, the desktop — so it holds the privilege; the twin, idle in
    the background, does not, and without the grant its raise degrades to a
    taskbar flash. Elsewhere this is a no-op: X11 honours the activation
    itself and Wayland's compositor will do what it will do.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes  # noqa: PLC0415 — Windows-only, and only on this path

        # No `type: ignore` on windll: the platform guard above narrows it —
        # mypy on Windows sees the attribute, mypy on Linux sees dead code.
        ctypes.windll.user32.AllowSetForegroundWindow(int(pid))
    except (AttributeError, OSError):
        pass  # the raise still happens; it just may not take focus


def forward(address: str) -> bool:
    """Offer this launch's activation to whoever already serves ``address``.

    True means a twin answered and will raise its window — the caller's next
    move is to exit quietly. False covers every other outcome: nobody there, a
    corpse of a socket, a stranger's protocol, a twin too wedged to answer.
    """
    try:
        peer = connection.Client(address)
    except OSError:
        return False  # nobody serving: FileNotFoundError, ConnectionRefused…
    try:
        if not peer.poll(_REPLY_DEADLINE):
            return False
        parts = peer.recv_bytes(64).split()
        if len(parts) != 2 or parts[0] != _GREETING:
            return False
        _grant_foreground(int(parts[1]))
        peer.send_bytes(_ACTIVATE)
    except (EOFError, OSError, ValueError):
        return False
    else:
        return True
    finally:
        peer.close()


class SingleInstance:
    """The serving half: claim an endpoint, raise the window on each twin.

    ``on_activate`` runs on the claim's own thread — whoever installs it
    marshals to their event loop themselves (``app.main`` emits a Qt signal,
    which Qt queues across threads on its own).
    """

    def __init__(self, address: str, on_activate: Callable[[], None]) -> None:
        self._address = address
        self._on_activate = on_activate
        self._listener: connection.Listener | None = None
        self._thread: Thread | None = None
        self._stop = Event()

    def claim(self) -> bool:
        """Take the endpoint and start answering.

        False when a live twin already holds it — the caller should
        :func:`forward` instead. A socket file a crashed twin left behind is
        swept up rather than treated as a twin; Windows has no such state,
        its pipes die with their process.
        """
        try:
            self._listener = connection.Listener(self._address)
        except OSError:
            if sys.platform == "win32" or not self._sweep_corpse():
                return False
        self._thread = Thread(target=self._serve, name="single-instance", daemon=True)
        self._thread.start()
        return True

    def release(self) -> None:
        """Stop answering and leave the endpoint clean. Idempotent."""
        self._stop.set()
        if self._thread is not None:
            # Unblock the accept by being, briefly, the client it waits for.
            try:
                connection.Client(self._address).close()
            except OSError:
                pass
            self._thread.join(timeout=5)
            self._thread = None
        if self._listener is not None:
            try:
                self._listener.close()  # on Unix this also unlinks the socket
            except OSError:
                pass
            self._listener = None

    def _sweep_corpse(self) -> bool:
        """A path nobody listens on is a crash's leavings, not a twin."""
        try:
            connection.Client(self._address).close()
        except OSError:
            pass  # refused: a corpse, and ours to remove
        else:
            return False  # answered: a live twin after all
        try:
            os.unlink(self._address)
            self._listener = connection.Listener(self._address)
        except OSError:
            return False
        return True

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                peer = self._listener.accept() if self._listener is not None else None
            except OSError:
                break  # closed under us: released
            if peer is None:
                break
            try:
                if self._stop.is_set():
                    break  # the nudge from release(), not a person
                peer.send_bytes(b"%s %d" % (_GREETING, os.getpid()))
                if peer.poll(_REPLY_DEADLINE) and peer.recv_bytes(64) == _ACTIVATE:
                    self._on_activate()
            except (EOFError, OSError):
                pass  # a peer that vanished mid-sentence was not raising anything
            finally:
                peer.close()
