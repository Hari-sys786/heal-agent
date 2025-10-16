from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from selfheal.utils.servicenow_client import ServiceNowClient


@dataclass(slots=True)
class TicketUpdaterConfig:
    """Configuration options for the ticket updater."""

    auto_resolve: bool = False
    review_assignment_group: str | None = "Auto-Bot Review"
    resolved_state: str = "3"
    review_state: str = "2"
    reassigned_state: str = "2"


class TicketUpdater:
    """Encapsulates the writes back to ServiceNow."""

    def __init__(self, sn_client: ServiceNowClient, config: TicketUpdaterConfig | None = None) -> None:
        self.sn_client = sn_client
        self.config = config or TicketUpdaterConfig()
        self._logger = logging.getLogger(__name__)

    def add_work_note(self, sys_id: str, note: str) -> None:
        """Append a work note to the record."""
        self._logger.debug("Adding work note to %s: %s", sys_id, note)
        self.sn_client.append_work_note(sys_id, note)

    def mark_resolved(self, sys_id: str, resolution: str, extra: Mapping[str, Any] | None = None) -> None:
        """Mark the ticket as resolved and flag automation completion."""
        self._logger.info("Marking ticket %s as resolved (auto_resolve=%s)", sys_id, self.config.auto_resolve)
        fields: Dict[str, Any] = {
            "state": self.config.resolved_state,
            "ai_done": True,
        }
        if resolution:
            fields["work_notes"] = resolution
        if extra:
            fields.update(extra)
        self._logger.debug("Resolved fields for %s: %s", sys_id, fields)
        self.sn_client.update_fields(sys_id, fields)

    def mark_for_review(self, sys_id: str, note: str | None = None) -> None:
        """Keep the ticket in a reviewable state."""
        self._logger.info("Marking ticket %s for review", sys_id)
        fields: Dict[str, Any] = {
            "state": self.config.review_state,
        }
        if self.config.review_assignment_group:
            fields["assignment_group"] = self.config.review_assignment_group
        if note:
            fields["work_notes"] = note
        self._logger.debug("Review fields for %s: %s", sys_id, fields)
        self.sn_client.update_fields(sys_id, fields)

    def reassign_out_of_scope(self, sys_id: str, group: str, note: str) -> None:
        """Reassign the ticket when it falls outside automation scope."""
        self._logger.info("Reassigning ticket %s to group %s", sys_id, group)
        fields: Dict[str, Any] = {
            "state": self.config.reassigned_state,
            "assignment_group": group,
            "work_notes": note,
        }
        self._logger.debug("Reassign fields for %s: %s", sys_id, fields)
        self.sn_client.update_fields(sys_id, fields)


__all__ = ["TicketUpdater", "TicketUpdaterConfig"]
