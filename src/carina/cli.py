"""The ``carina`` command.

    carina [--data DIR] [--decisions DIR] COMMAND ...

Commands:
    refresh market|sec ...  refresh market data or SEC financial facts
    collect | build | verify  capture SEC facts and new companies, build, verify
    macro refresh|verify    refresh macro series and record their revisions
    audit financial|market  audit a verified dataset
    sync status|pull|push company|macro
                            fast-forward-only Drive sync of a staged dataset
    drive download|upload   non-destructive Drive sync of explicit paths

``--data`` (or ``CARINA_DATA``) is the directory holding ``staged/``, ``runs/``
and ``receipts/``. ``--decisions`` (or ``CARINA_DECISIONS``) holds the decision
files. Run ``carina COMMAND --help`` for a command's options.
"""

from __future__ import annotations

import sys

from carina import settings


def _store(action):
    def run(argv):
        from carina import store

        return store.main([action, *argv]) or 0

    return run


def _refresh(argv):
    if argv and argv[0] == "sec":
        from carina import sec_refresh

        return sec_refresh.main(argv[1:])
    from carina import refresh

    return refresh.main(argv)


def _macro(argv):
    from carina import macro

    return macro.main(argv)


def _audit(argv):
    from carina import audit

    kinds = {"financial": audit.financial, "market": audit.market}
    if not argv or argv[0] not in kinds:
        raise SystemExit("usage: carina audit {financial,market} ...")
    return kinds[argv[0]](argv[1:])


def _sync(argv):
    from carina import sync

    return sync.main(argv)


def _drive(argv):
    from carina import drive

    return drive.main(argv)


COMMANDS = {
    "refresh": _refresh,
    "collect": _store("collect"),
    "build": _store("build"),
    "verify": _store("verify"),
    "macro": _macro,
    "audit": _audit,
    "sync": _sync,
    "drive": _drive,
}


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    options = {}
    while args and args[0] in {"--data", "--decisions"}:
        if len(args) < 2:
            raise SystemExit(f"{args[0]} needs a directory")
        options[args[0][2:]] = args[1]
        args = args[2:]
    if not args or args[0] in {"-h", "--help"} or args[0] not in COMMANDS:
        print(__doc__)
        return 0 if args and args[0] in {"-h", "--help"} else 2
    settings.configure(**options)
    return COMMANDS[args[0]](args[1:])
