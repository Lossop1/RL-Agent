from __future__ import annotations

from autotuner.adapter.remote_executors import RemoteSSHTransportAdapter


class _LegacyRemote:
    def __init__(self, *, return_code: int, stderr: str = ""):
        self.return_code = return_code
        self.stderr = stderr
        self.commands: list[str] = []

    def exec(self, command: str, timeout: int = 30):
        self.commands.append(command)
        marker = RemoteSSHTransportAdapter._RETURN_CODE_MARKER
        return f"command output\n{marker}{self.return_code}\n", self.stderr

    def put(self, local: str, remote: str) -> None:
        pass


def test_legacy_transport_does_not_treat_stderr_warning_as_failure() -> None:
    remote = _LegacyRemote(return_code=0, stderr="harmless warning")
    output, return_code = RemoteSSHTransportAdapter(remote).exec("echo ok")

    assert output == "command output"
    assert return_code == 0


def test_legacy_transport_preserves_nonzero_return_code() -> None:
    remote = _LegacyRemote(return_code=23)
    output, return_code = RemoteSSHTransportAdapter(remote).exec("false")

    assert output == "command output"
    assert return_code == 23
