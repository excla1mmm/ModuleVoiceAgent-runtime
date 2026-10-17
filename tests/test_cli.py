from unittest.mock import patch

import pytest

from voice_agent_runtime.cli import main


@pytest.mark.parametrize(("env", "url"), [
    ({}, "http://127.0.0.1:8443/health"),
    ({"VA_LISTEN_PORT": "9000"}, "http://127.0.0.1:9000/health"),
    ({"VA_TLS_CERT": "/c", "VA_TLS_KEY": "/k"}, "https://127.0.0.1:8443/health"),
])
def test_healthcheck_uses_listener_settings(env, url):
    with patch.dict("os.environ", env, clear=True), patch("voice_agent_runtime.cli.healthcheck", return_value=0) as check:
        assert main(["healthcheck"]) == 0
        assert check.call_args.args == (url,)


def test_explicit_healthcheck_url_is_preserved():
    with patch("voice_agent_runtime.cli.healthcheck", return_value=0) as check:
        assert main(["healthcheck", "--url", "http://localhost:123/health"]) == 0
        assert check.call_args.args == ("http://localhost:123/health",)
