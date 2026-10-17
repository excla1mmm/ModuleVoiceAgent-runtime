"""In-memory stores. Nothing here ever touches the disk: agent data and call results live in RAM only.

* ``BundleStore``: agent bundles keyed by tenant and agent id. A restart empties it; the PBX heartbeat sees
  the missing versions and pushes the bundles again. On a blue-green deploy the new instance takes them over
  from the old one over the operator API (memory to memory, ``peer.py``).
* ``ResultStore``: call results keyed by (tenant, PBX call id). A result is retained until acknowledged by
  the PBX; unclaimed results expire after a TTL, and a PBX that never acknowledges keeps at most
  ``max_per_tenant`` results (the oldest are evicted first). While a call is still active a reader may wait
  a little for it to finish (the PBX asks right after Dial() returns, which can be a moment before the
  WebSocket closes).
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass

from .bundle import AgentBundle
from .hotwords import bundle_terms
from .prompt import static_prompt


@dataclass
class StoredBundle:
    bundle: AgentBundle
    static_prompt: str
    loaded_at: float
    terms: tuple[str, ...] = ()  # speech recognition hints (hotwords.py)
    raw: dict | None = None  # the bundle JSON as the PBX sent it, handed over to the next instance on a deploy


class BundleStore:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, StoredBundle]] = {}

    def put(self, tenant: str, bundle: AgentBundle, raw: dict | None = None) -> StoredBundle:
        stored = StoredBundle(bundle, static_prompt(bundle), time.time(), bundle_terms(bundle), raw)
        self._items.setdefault(tenant, {})[bundle.agent_id] = stored
        return stored

    def get(self, tenant: str, agent_id: str) -> StoredBundle | None:
        return self._items.get(tenant, {}).get(agent_id)

    def has(self, tenant: str, agent_id: str) -> bool:
        return agent_id in self._items.get(tenant, {})

    def delete(self, tenant: str, agent_id: str) -> bool:
        items = self._items.get(tenant)
        if not items or items.pop(agent_id, None) is None:
            return False
        if not items:
            del self._items[tenant]
        return True

    def list(self, tenant: str) -> list[StoredBundle]:
        return list(self._items.get(tenant, {}).values())

    def count(self, tenant: str) -> int:
        return len(self._items.get(tenant, {}))

    def counts(self) -> dict[str, int]:
        return {tenant: len(items) for tenant, items in self._items.items()}

    def all(self) -> list[tuple[str, StoredBundle]]:
        return [(tenant, stored) for tenant, items in self._items.items() for stored in items.values()]

    def __len__(self) -> int:
        return sum(len(items) for items in self._items.values())


class ResultStore:
    def __init__(self, ttl_s: float, max_per_tenant: int = 1000):
        self.ttl_s = ttl_s
        self.max_per_tenant = max_per_tenant
        # insertion-ordered per tenant, so the oldest result of a tenant is the first item
        self._results: dict[str, OrderedDict[str, tuple[float, dict]]] = {}
        self._active: dict[tuple[str, str], asyncio.Event] = {}

    def start(self, tenant: str, call_id: str) -> None:
        self._active[(tenant, call_id)] = asyncio.Event()

    def finish(self, tenant: str, call_id: str, result: dict) -> int:
        """Stores the result of a call (first finish wins) and wakes up waiting readers.

        Returns how many older results of the tenant had to be evicted to stay within the per-tenant cap.
        """
        results = self._results.setdefault(tenant, OrderedDict())
        results.setdefault(call_id, (time.monotonic(), result))
        evicted = 0
        while len(results) > self.max_per_tenant:
            results.popitem(last=False)
            evicted += 1
        event = self._active.pop((tenant, call_id), None)
        if event is not None:
            event.set()
        return evicted

    def is_active(self, tenant: str, call_id: str) -> bool:
        return (tenant, call_id) in self._active

    def active_count(self) -> int:
        return len(self._active)

    def _get(self, tenant: str, call_id: str) -> dict | None:
        item = self._results.get(tenant, {}).get(call_id)
        if item is None:
            return None
        if item[0] < time.monotonic() - self.ttl_s:
            self.acknowledge(tenant, call_id)
            return None
        return item[1]

    async def take(self, tenant: str, call_id: str, wait_s: float = 0.0) -> tuple[str, dict | None]:
        """Returns ("ready", result) until acknowledged, ("active", None) while the call runs, ("unknown", None) otherwise."""
        key = (tenant, call_id)
        event = self._active.get(key)
        result = self._get(tenant, call_id)
        if result is None and event is not None and wait_s > 0:
            try:
                await asyncio.wait_for(event.wait(), timeout=wait_s)
            except TimeoutError:
                pass
            result = self._get(tenant, call_id)
        if result is not None:
            return "ready", result
        return ("active", None) if key in self._active else ("unknown", None)

    def acknowledge(self, tenant: str, call_id: str) -> None:
        results = self._results.get(tenant)
        if results is not None:
            results.pop(call_id, None)
            if not results:
                del self._results[tenant]

    def expire(self) -> None:
        cutoff = time.monotonic() - self.ttl_s
        for tenant, results in list(self._results.items()):
            for call_id in [k for k, (ts, _) in results.items() if ts < cutoff]:
                del results[call_id]
            if not results:
                del self._results[tenant]

    def is_pending(self, tenant: str, call_id: str) -> bool:
        return call_id in self._results.get(tenant, {})

    def pending_count(self, tenant: str | None = None) -> int:
        if tenant is not None:
            return len(self._results.get(tenant, {}))
        return sum(len(r) for r in self._results.values())

    def pending_counts(self) -> dict[str, int]:
        return {tenant: len(r) for tenant, r in self._results.items()}
