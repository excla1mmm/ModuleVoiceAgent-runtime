"""Agent bundle: everything a PBX tells the runtime about one agent. Kept in memory only.

The PBX module builds the bundle from its own database (agent settings, directory, knowledge base) and
pushes it with ``PUT /v1/agents/{agent_id}``; ``version`` is the PBX-side hash of the content, so the PBX
can see which agents the runtime already has after a heartbeat. Optional fields (absent in older modules):
``hotwords`` - terms that speech recognition should prefer (see ``hotwords.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import DEFAULT_VOICE

SCHEMA_VERSION = 1
EXTENSION = re.compile(r"^[0-9]{1,20}$")
VERSION = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
# Any voice id is accepted: one this runtime does not load (a newer PBX module) speaks with the default voice.
VOICE_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")
MAX_TEXT = 4000
MAX_ENTRIES = 2000
MAX_HOTWORDS = 200
MAX_HOTWORD_LENGTH = 100


class BundleError(ValueError):
    pass


@dataclass(frozen=True)
class DirectoryEntry:
    number: str
    name: str
    position: str = ""
    department: str = ""
    aliases: tuple[str, ...] = ()
    route_to: str = ""  # "do not connect directly, transfer to this number instead"


@dataclass(frozen=True)
class QueueEntry:
    number: str
    name: str


@dataclass(frozen=True)
class KnowledgeEntry:
    topic: str
    answer: str


@dataclass(frozen=True)
class AgentBundle:
    agent_id: str
    version: str
    name: str
    extension: str
    company_name: str
    greeting: str
    voice: str
    rules: str
    fallback_extension: str
    callbacks_enabled: bool
    transfer_grace_s: float
    goodbye_grace_s: float
    directory: tuple[DirectoryEntry, ...] = ()
    queues: tuple[QueueEntry, ...] = ()
    knowledge: tuple[KnowledgeEntry, ...] = ()
    hotwords: tuple[str, ...] = ()

    def transfer_targets(self) -> set[str]:
        """Numbers the agent may transfer to: the directory, the queues and the fallback number."""
        targets = {e.number for e in self.directory} | {q.number for q in self.queues}
        targets |= {e.route_to for e in self.directory if e.route_to}
        if self.fallback_extension:
            targets.add(self.fallback_extension)
        return targets

    def target_name(self, extension: str) -> str:
        """Whom a number reaches: an employee or a department queue ("" for an unknown number)."""
        extension = re.sub(r"\D", "", extension or "")
        entry = next((e for e in self.directory if e.number == extension), None)
        if entry is not None:
            return f"{entry.name}, {entry.position}" if entry.position else entry.name
        return next((re.sub(r"\s*\([^)]*\)", "", q.name).strip() for q in self.queues if q.number == extension), "")

    def resolve_transfer(self, extension: str) -> str | None:
        """The number to transfer to, following "route to X" rules; None when the number is not allowed."""
        extension = re.sub(r"\D", "", extension or "")
        for entry in self.directory:
            if entry.number == extension and entry.route_to:
                return entry.route_to
        return extension if extension in self.transfer_targets() else None


def _text(data: dict, key: str, limit: int = MAX_TEXT, required: bool = False) -> str:
    value = data.get(key, "")
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise BundleError(f"{key} must be a string")
    value = value.strip()
    if required and not value:
        raise BundleError(f"{key} is required")
    if len(value) > limit:
        raise BundleError(f"{key} is longer than {limit} characters")
    return value


def _number(data: dict, key: str, required: bool) -> str:
    value = _text(data, key, 20)
    if not value and not required:
        return ""
    if not EXTENSION.match(value):
        raise BundleError(f"{key} must be 1-20 digits")
    return value


def _grace(data: dict, key: str, default: float) -> float:
    value = data.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 30:
        raise BundleError(f"{key} must be a number of seconds between 0 and 30")
    return float(value)


def _list(data: dict, key: str) -> list:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise BundleError(f"{key} must be a list")
    if len(value) > MAX_ENTRIES:
        raise BundleError(f"{key} has more than {MAX_ENTRIES} entries")
    if not all(isinstance(item, dict) for item in value):
        raise BundleError(f"{key} entries must be objects")
    return value


def _hotwords(data: dict) -> tuple[str, ...]:
    value = data.get("hotwords")
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise BundleError("hotwords must be a list of strings")
    if len(value) > MAX_HOTWORDS:
        raise BundleError(f"hotwords has more than {MAX_HOTWORDS} entries")
    if any(len(item) > MAX_HOTWORD_LENGTH for item in value):
        raise BundleError(f"hotwords entries must be at most {MAX_HOTWORD_LENGTH} characters")
    return tuple(item.strip() for item in value if item.strip())


def parse_bundle(agent_id: str, data: dict) -> AgentBundle:
    if not isinstance(data, dict):
        raise BundleError("bundle must be a JSON object")
    if data.get("schema") != SCHEMA_VERSION:
        raise BundleError(f"unsupported bundle schema {data.get('schema')!r}, expected {SCHEMA_VERSION}")
    if not re.fullmatch(r"[0-9]{1,16}", agent_id):
        raise BundleError("agent id must be numeric")
    version = _text(data, "version", 128, required=True)
    if not VERSION.match(version):
        raise BundleError("invalid version")
    voice = _text(data, "voice", 64) or DEFAULT_VOICE
    if not VOICE_ID.fullmatch(voice):
        raise BundleError("invalid voice")
    directory = []
    for item in _list(data, "directory"):
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise BundleError("directory aliases must be a list of strings")
        directory.append(DirectoryEntry(
            number=_number(item, "number", True), name=_text(item, "name", 200, required=True),
            position=_text(item, "position", 200), department=_text(item, "department", 200),
            aliases=tuple(a.strip() for a in aliases if a.strip())[:20], route_to=_number(item, "route_to", False)))
    queues = [QueueEntry(_number(item, "number", True), _text(item, "name", 200, required=True))
              for item in _list(data, "queues")]
    knowledge = [KnowledgeEntry(_text(item, "topic", 300, required=True), _text(item, "answer", MAX_TEXT, required=True))
                 for item in _list(data, "knowledge")]
    return AgentBundle(
        agent_id=agent_id, version=version, name=_text(data, "name", 200), extension=_number(data, "extension", True),
        company_name=_text(data, "company_name", 200, required=True), greeting=_text(data, "greeting", 500, required=True),
        voice=voice, rules=_text(data, "rules", MAX_TEXT), fallback_extension=_number(data, "fallback_extension", False),
        callbacks_enabled=bool(data.get("callbacks_enabled", True)),
        transfer_grace_s=_grace(data, "transfer_grace_seconds", 2.0),
        goodbye_grace_s=_grace(data, "goodbye_grace_seconds", 3.0),
        directory=tuple(directory), queues=tuple(queues), knowledge=tuple(knowledge), hotwords=_hotwords(data))
