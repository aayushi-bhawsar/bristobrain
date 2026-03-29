"""
utils/whatsapp.py
Twilio WhatsApp delivery helper.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from data.schema import ActionCard

log = structlog.get_logger(__name__)


def send_action_cards(cards: list["ActionCard"]) -> int:
    """
    Send a list of ActionCards to the configured WhatsApp number via Twilio.
    Returns the number of messages successfully sent.
    """
    try:
        from twilio.rest import Client
    except ImportError:
        raise RuntimeError("twilio not installed. Run: pip install twilio")

    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    from_number = os.environ["TWILIO_WHATSAPP_FROM"]
    to_number = os.environ["RESTAURANT_OWNER_WHATSAPP"]

    client = Client(account_sid, auth_token)
    sent = 0

    for card in cards:
        try:
            client.messages.create(
                body=card.to_whatsapp_message(),
                from_=from_number,
                to=to_number,
            )
            sent += 1
            log.info("whatsapp_sent", card_id=card.card_id, priority=card.priority)
        except Exception as exc:
            log.error("whatsapp_send_failed", card_id=card.card_id, error=str(exc))

    return sent
