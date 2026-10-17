"""The other instance of a blue-green pair (``VA_PEER_URL``), reached over the operator API.

A deploy starts the new instance next to the old one. The new one takes the agent bundles over from the old
one before it reports ready (memory to memory: bundles never touch the disk), and a PBX asking it for the
result of a call the old instance served is answered from the old one. The operator token
(``VA_ADMIN_TOKEN``, the same in both instances) authenticates the instances to each other; the PBX tokens
are never forwarded. A peer that is not running answers nothing: everything then works as without a peer.
"""

from __future__ import annotations

import logging
from urllib.parse import quote

import aiohttp

log = logging.getLogger("voice_agent.peer")
CONNECT_TIMEOUT_S = 2.0
HANDOFF_TIMEOUT_S = 30.0


class Peer:
    def __init__(self, url: str, admin_token: str):
        self.url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {admin_token}"}
        self._session: aiohttp.ClientSession | None = None

    def _client(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._headers)
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    def _timeout(self, total: float) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(total=total, connect=CONNECT_TIMEOUT_S)

    def _call_path(self, tenant: str, call_id: str) -> str:
        return f"{self.url}/internal/calls/{quote(tenant, safe='')}/{quote(call_id, safe='')}/result"

    async def bundles(self) -> list[dict]:
        """Every bundle the peer holds: [{tenant, agent_id, data}]; [] when the peer is not there."""
        try:
            async with self._client().get(f"{self.url}/internal/handoff",
                                          timeout=self._timeout(HANDOFF_TIMEOUT_S)) as resp:
                if resp.status != 200:
                    log.warning("peer handoff: HTTP %d", resp.status)
                    return []
                body = await resp.json()
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            log.info("no peer to take bundles over from (%s)", exc.__class__.__name__)
            return []
        items = body.get("bundles") if isinstance(body, dict) else None
        return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []

    async def result(self, tenant: str, call_id: str, wait: float) -> tuple[int, dict] | None:
        """The peer's answer for a call result (200 result, 202 active), or None when it does not know it."""
        try:
            async with self._client().get(self._call_path(tenant, call_id), params={"wait": str(wait)},
                                          timeout=self._timeout(wait + 5)) as resp:
                if resp.status not in (200, 202):
                    return None
                return resp.status, await resp.json()
        except (aiohttp.ClientError, TimeoutError, ValueError):
            return None

    async def acknowledge(self, tenant: str, call_id: str) -> None:
        try:
            async with self._client().delete(self._call_path(tenant, call_id), timeout=self._timeout(5)):
                pass
        except (aiohttp.ClientError, TimeoutError):
            pass  # the result expires at the peer after VA_RESULT_TTL
