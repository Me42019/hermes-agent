"""Regression: ``resolve_provider_client("minimax-oauth", ...)`` must build a
refresh-capable Anthropic auxiliary client, not silently return (None, None).

The auxiliary router dispatches on ``PROVIDER_REGISTRY[*].auth_type``.
``minimax-oauth`` registers ``auth_type == "oauth_minimax"``; without an
explicit arm the resolver falls through the ``oauth_device_code`` /
``oauth_external`` branch, logs a one-time warning, and returns (None, None).
Every aux task pinned to ``provider: minimax-oauth`` (compression,
title_generation, …) then silently re-routes to the Step-2 fallback chain —
the operator's explicit configuration never reaches the wire.

The fix routes through ``resolve_minimax_oauth_runtime_credentials(as_token_provider=True)``
and wraps the resulting Anthropic SDK client in ``AnthropicAuxiliaryClient``. The callable
bearer mints a fresh access token per outbound request because MiniMax's tokens live
~15 minutes and a static string would 401 mid-session.

``is_oauth`` is derived via ``anthropic_route_is_oauth(base_url, token_provider)``: the
MiniMax host is a third-party Anthropic-protocol endpoint, so the wrapper must NOT carry
the Claude Code OAuth identity (mcp__ tool-name wire transforms, system-prompt rewrites,
response prefix stripping) — those are native api.anthropic.com-only (#114967).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

INFERENCE_BASE_URL = "https://api.minimax.io/anthropic"


def _runtime_creds():
    """Return what resolve_minimax_oauth_runtime_credentials(as_token_provider=True) yields."""
    return {
        "provider": "minimax-oauth",
        "api_key": MagicMock(name="minimax_token_provider", return_value="fresh-bearer"),
        "base_url": INFERENCE_BASE_URL,
        "source": "oauth",
    }


def test_resolve_minimax_oauth_builds_anthropic_wrapper_with_oauth_semantics():
    """Happy path: token-provider + base_url → AnthropicAuxiliaryClient with the
    third-party is_oauth invariant (False for api.minimax.io), Anthropic SDK built
    with the callable bearer (not a static string), and the resolved model passed
    through verbatim.
    """
    from agent.auxiliary_client import (
        AnthropicAuxiliaryClient,
        resolve_provider_client,
    )

    fake_anthropic_client = MagicMock(name="anthropic_sdk_client")
    creds = _runtime_creds()

    with patch(
        "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
        return_value=creds,
    ), patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=fake_anthropic_client,
    ) as mock_build:
        client, model = resolve_provider_client("minimax-oauth", "MiniMax-M3")

    assert client is not None, (
        "minimax-oauth must produce a configured client when credentials are "
        "present, but the resolver returned (None, None). The oauth_minimax "
        "arm in _resolve_registry_branch is missing."
    )
    assert isinstance(client, AnthropicAuxiliaryClient), (
        f"minimax-oauth must build an AnthropicAuxiliaryClient (the inference "
        f"endpoint is /anthropic). Got {type(client).__name__}."
    )
    assert client.chat.completions._is_oauth is False, (
        "MiniMax is a third-party Anthropic-protocol endpoint: is_oauth enables "
        "Claude Code-native transforms (mcp__ tool-name wire prefixing, identity "
        "rewrites, response prefix stripping) that 401/403 or corrupt tool calls "
        "there. anthropic_route_is_oauth('https://api.minimax.io/anthropic', …) "
        "is False by contract (tests/agent/test_anthropic_route_oauth_identity.py)."
    )
    # The callable token provider — not a string — must reach the Anthropic SDK so
    # the SDK mints a fresh access token per outbound request (MiniMax tokens
    # are short-lived; a static bearer 401s mid-session).
    positional, _kwargs = mock_build.call_args[0], mock_build.call_args[1]
    assert positional[0] is creds["api_key"], (
        "build_anthropic_client must receive the callable token provider, "
        "not a stringified snapshot of the current bearer."
    )
    assert positional[1] == INFERENCE_BASE_URL
    assert model == "MiniMax-M3"


def test_resolve_minimax_oauth_tool_names_unprefixed_on_wire():
    """Witness for the user-visible contract: a tool list sent through the
    minimax-oauth auxiliary wrapper must reach the wire with its registry names
    verbatim. ``is_oauth=True`` would rename ``read_file`` to ``mcp__read_file``
    (and alias session_search/memory) on a host that never round-trips those
    names — every tool call against MiniMax would 400 or name a nonexistent tool.
    """
    from agent.auxiliary_client import resolve_provider_client

    captured = {}

    def _fake_create(client, api_kwargs, **kwargs):
        captured["tools"] = api_kwargs.get("tools")
        return SimpleNamespace(
            content=[],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    with patch(
        "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
        return_value=_runtime_creds(),
    ), patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=MagicMock(name="anthropic_sdk_client"),
    ), patch(
        "agent.anthropic_adapter.create_anthropic_message",
        side_effect=_fake_create,
    ):
        client, _model = resolve_provider_client("minimax-oauth", "MiniMax-M3")
        assert client is not None
        client.chat.completions.create(
            model="MiniMax-M3",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {"type": "function", "function": {"name": "read_file", "description": "x", "parameters": {}}},
                {"type": "function", "function": {"name": "session_search", "description": "y", "parameters": {}}},
            ],
        )
    wire_names = sorted(t["name"] for t in captured["tools"])
    assert wire_names == ["read_file", "session_search"], (
        "Third-party Anthropic-protocol endpoints must see unprefixed tool names; "
        "the Claude Code OAuth wire renamer must not run for api.minimax.io."
    )


def test_resolve_minimax_oauth_missing_credentials_returns_none_without_raising():
    """AuthError from the runtime resolver → (None, None), no exception.

    The resolver contract is "absent → call_llm's fallback chain", never
    "absent → exception"; the compression step must fall through to its
    Step-2 providers instead of crashing the turn.
    """
    from agent.auxiliary_client import resolve_provider_client

    with patch(
        "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
        side_effect=Exception("not logged in"),
    ):
        client, model = resolve_provider_client("minimax-oauth", "MiniMax-M3")

    assert client is None
    assert model is None
