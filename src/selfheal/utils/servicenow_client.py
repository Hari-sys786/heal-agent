from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import requests
from requests import Response
from requests.auth import HTTPBasicAuth
from requests.exceptions import JSONDecodeError

# Configure module logger
logger = logging.getLogger(__name__)

FIELD_PREFIX = "u_"

FIELD_MAP: Dict[str, str] = {
    "short_description": f"{FIELD_PREFIX}short_description",
    "description": f"{FIELD_PREFIX}description",
    "assignment_group": f"{FIELD_PREFIX}assignment_group",
    "state": f"{FIELD_PREFIX}state",
    "category": f"{FIELD_PREFIX}category",
    "subcategory": f"{FIELD_PREFIX}subcategory",
    "urgency": f"{FIELD_PREFIX}urgency",
    "impact": f"{FIELD_PREFIX}impact",
    "priority": f"{FIELD_PREFIX}priority",
    "emergency": f"{FIELD_PREFIX}emergency",
    "ai_done": f"{FIELD_PREFIX}ai_done",
    "work_notes": f"{FIELD_PREFIX}notes",
    "caller_id": f"{FIELD_PREFIX}caller_id",
}

FIELD_MAP_REVERSE: Dict[str, str] = {service: canonical for canonical, service in FIELD_MAP.items()}

BOOLEAN_FIELDS = {"ai_done"}
BOOLEAN_SERVICE_FIELDS = {FIELD_MAP[field] for field in BOOLEAN_FIELDS if field in FIELD_MAP}

RESERVED_WRITE_FIELDS = {
    "sys_id",
    "sys_created_on",
    "sys_updated_on",
    "sys_created_by",
    "sys_updated_by",
    "sys_mod_count",
    "sys_class_name",
    "sys_import_state",
    "sys_import_state_comment",
    "sys_import_set",
    "sys_import_row",
    "sys_transform_map",
    "sys_tags",
    "sys_target_sys_id",
    "sys_target_table",
    "sys_row_error",
    "import_set_run",
    "number",
    "raw",
}


@dataclass(slots=True)
class ServiceNowConfig:
    """Connection details for the ServiceNow instance."""

    instance_url: str
    username: str | None = None
    password: str | None = None
    token: str | None = None
    table: str = "u_heal_agent"
    assignment_group: str | None = "Auto-Bot"
    include_display_values: bool = False
    default_fields: Dict[str, Any] = field(
        default_factory=lambda: {
            "caller_id": "auto.bot",
            "category": "request",
            "subcategory": "software",
            "impact": "3",
            "urgency": "3",
            "state": "1",
            "ai_done": False,
        }
    )

    def table_url(self) -> str:
        """Return the fully-qualified REST endpoint for the configured table."""
        base = self.instance_url.rstrip("/")
        return f"{base}/api/now/table/{self.table}"


class ServiceNowClient:
    """Lightweight wrapper around the ServiceNow Table API."""

    def __init__(self, config: ServiceNowConfig) -> None:
        self.config = config
        self._session = requests.Session()
        logger.debug(
            "Initializing ServiceNowClient (instance=%s table=%s include_display=%s)",
            config.instance_url,
            config.table,
            config.include_display_values,
        )
        if config.token:
            self._session.headers["Authorization"] = f"Bearer {config.token}"
        elif config.username and config.password:
            self._session.auth = HTTPBasicAuth(config.username, config.password)
        else:
            raise ValueError("ServiceNowConfig requires either token or username/password credentials.")
        self._session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def create_incident(
        self,
        short_description: str,
        description: str,
        *,
        fields: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new record assigned to the automation group."""
        combined_fields: Dict[str, Any] = dict(self.config.default_fields)
        combined_fields.update(
            {
                "short_description": short_description,
                "description": description,
            }
        )
        if self.config.assignment_group:
            combined_fields.setdefault("assignment_group", self.config.assignment_group)
        if fields:
            combined_fields.update(dict(fields))

        payload: MutableMapping[str, Any] = _map_to_service_fields(combined_fields)
        logger.info("Creating ServiceNow record with payload: %s", combined_fields)

        response = self._session.post(self.config.table_url(), json=payload, timeout=30)
        response.raise_for_status()
        record = _normalize_single(response)
        logger.info("Created ServiceNow record %s", record.get("sys_id") or record.get("number"))
        return record

    def fetch_incident(self, sys_id: str) -> Dict[str, Any]:
        """Retrieve a single record by sys_id."""
        logger.debug("Fetching ServiceNow record %s", sys_id)
        response = self._session.get(f"{self.config.table_url()}/{sys_id}", timeout=30)
        response.raise_for_status()
        return _normalize_single(response)

    def list_incidents(self, *, query: Optional[str] = None, limit: int = 20) -> Iterable[Dict[str, Any]]:
        """List records in the configured table."""
        logger.debug("Listing ServiceNow incidents (query=%s limit=%s)", query, limit)
        params: Dict[str, Any] = {"sysparm_limit": str(limit)}
        if self.config.include_display_values:
            params["sysparm_display_value"] = "all"
        if query:
            params["sysparm_query"] = query
        if self.config.assignment_group:
            params[FIELD_MAP["assignment_group"]] = self.config.assignment_group

        response = self._session.get(self.config.table_url(), params=params, timeout=30)
        response.raise_for_status()
        return _normalize_many(response)

    def append_work_note(self, sys_id: str, note: str) -> None:
        """Append a note to the record's work notes field."""
        if not note:
            return
        logger.debug("Appending work note to %s: %s", sys_id, note)
        record = self.fetch_incident(sys_id)
        existing = record.get("work_notes") or ""
        combined = f"{existing}\n{note}".strip() if existing else note
        payload = _map_to_service_fields({"work_notes": combined})
        response = self._session.patch(f"{self.config.table_url()}/{sys_id}", json=payload, timeout=30)
        response.raise_for_status()
        logger.debug("Work note appended to %s", sys_id)

    def update_fields(self, sys_id: str, fields: Mapping[str, Any]) -> None:
        """Patch arbitrary fields on a record."""
        payload = _map_to_service_fields(fields)
        if not payload:
            return
        logger.debug("Updating fields on %s: %s", sys_id, payload)
        response = self._session.patch(f"{self.config.table_url()}/{sys_id}", json=payload, timeout=30)
        response.raise_for_status()
        logger.debug("Fields updated on %s", sys_id)


def _map_to_service_fields(values: Mapping[str, Any]) -> Dict[str, Any]:
    mapped: Dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        if key in RESERVED_WRITE_FIELDS:
            continue
        field_name = _resolve_service_field(key)
        if field_name is None:
            continue
        if key in BOOLEAN_FIELDS or field_name in BOOLEAN_SERVICE_FIELDS:
            mapped[field_name] = "true" if _to_bool(value) else "false"
        else:
            mapped[field_name] = value
    return mapped


def _resolve_service_field(key: str) -> Optional[str]:
    if key in FIELD_MAP:
        return FIELD_MAP[key]
    if key.startswith(FIELD_PREFIX):
        return key
    candidate = f"{FIELD_PREFIX}{key}"
    return candidate


def _normalize_many(response: Response) -> Iterable[Dict[str, Any]]:
    payload = _parse_payload(response)
    result = _extract_result(payload)
    records = _ensure_list(result)
    return [_normalize_record(_ensure_dict(record)) for record in records]


def _normalize_single(response: Response) -> Dict[str, Any]:
    payload = _parse_payload(response)
    result = _extract_result(payload)
    if isinstance(result, list):
        record = result[0] if result else {}
    else:
        record = result
    return _normalize_record(_ensure_dict(record))


def _parse_payload(response: Response) -> Any:
    try:
        return response.json()
    except JSONDecodeError as exc:
        _debug_response(response)
        raise RuntimeError("ServiceNow API did not return valid JSON.") from exc


def _extract_result(payload: Any) -> Any:
    if isinstance(payload, dict):
        if "result" in payload:
            return payload["result"]
        if len(payload) == 1:
            return _extract_result(next(iter(payload.values())))
    return payload


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _ensure_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"ServiceNow API returned an unexpected record payload: {value!r}")


def _normalize_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = dict(record)

    for service_field, canonical in FIELD_MAP_REVERSE.items():
        if service_field in record and canonical not in normalized:
            normalized[canonical] = _coerce_value(canonical, record[service_field])

    for key, value in record.items():
        if key.startswith(FIELD_PREFIX):
            canonical = FIELD_MAP_REVERSE.get(key, key[len(FIELD_PREFIX) :])
            if canonical not in normalized:
                normalized[canonical] = _coerce_value(canonical, value)

    normalized.setdefault("sys_id", record.get("sys_id"))
    normalized.setdefault("short_description", normalized.get("short_description") or record.get(FIELD_MAP["short_description"]))
    normalized.setdefault("description", normalized.get("description") or record.get(FIELD_MAP["description"]))
    normalized.setdefault("assignment_group", normalized.get("assignment_group") or record.get(FIELD_MAP["assignment_group"]))
    normalized.setdefault("state", normalized.get("state") or record.get(FIELD_MAP["state"]))
    normalized.setdefault("urgency", normalized.get("urgency") or record.get(FIELD_MAP["urgency"]))
    normalized.setdefault("impact", normalized.get("impact") or record.get(FIELD_MAP["impact"]))
    normalized.setdefault("priority", normalized.get("priority") or (FIELD_MAP.get("priority") and record.get(FIELD_MAP["priority"])))
    normalized.setdefault("emergency", normalized.get("emergency") or (FIELD_MAP.get("emergency") and record.get(FIELD_MAP["emergency"])))
    normalized.setdefault("category", normalized.get("category") or record.get(FIELD_MAP["category"]))
    normalized.setdefault("subcategory", normalized.get("subcategory") or record.get(FIELD_MAP["subcategory"]))
    normalized.setdefault("work_notes", normalized.get("work_notes") or record.get(FIELD_MAP["work_notes"]))
    normalized.setdefault("number", record.get("number") or record.get("u_number") or normalized.get("short_description") or normalized.get("sys_id"))
    normalized.setdefault("sys_updated_on", record.get("sys_updated_on"))
    normalized.setdefault("sys_created_on", record.get("sys_created_on"))
    normalized.setdefault("ai_done", normalized.get("ai_done", False))
    normalized["raw"] = dict(record)
    return normalized


def _coerce_value(canonical: str, value: Any) -> Any:
    if canonical in BOOLEAN_FIELDS:
        return _to_bool(value)
    return value


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _debug_response(response: Response) -> None:
    print(
        "[ServiceNowClient] Unexpected response:\n"
        f"Status: {response.status_code}\n"
        f"Headers: {dict(response.headers)}\n"
        f"Body:\n{response.text}\n"
        "========================="
    )


__all__ = ["ServiceNowClient", "ServiceNowConfig"]
