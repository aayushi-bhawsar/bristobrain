"""
api/routes/webhook.py
Twilio WhatsApp webhook — handles inbound messages from restaurant owners
and routes them to the InsightAgent for real-time Q&A.
"""
from __future__ import annotations

import os

import structlog
from fastapi import APIRouter, Form, Response

from agents.insight_agent import InsightAgent
from data.schema import QueryRequest

log = structlog.get_logger(__name__)
router = APIRouter()

WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "")
OWNER_NUMBER = os.getenv("RESTAURANT_OWNER_WHATSAPP", "")

_insight_agent: InsightAgent | None = None


def _get_agent() -> InsightAgent:
    global _insight_agent
    if _insight_agent is None:
        _insight_agent = InsightAgent()
    return _insight_agent


def _twiml_response(message: str) -> Response:
    """Wrap a message in Twilio TwiML XML."""
    # Escape special XML characters
    safe = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{safe}</Message>
</Response>"""
    return Response(content=xml, media_type="application/xml")


@router.post("/webhook/whatsapp")
async def whatsapp_webhook(
    Body: str = Form(...),
    From: str = Form(...),
    To: str = Form(...),
):
    """
    Twilio inbound WhatsApp webhook.

    Receives a message from the restaurant owner and returns
    an insight via the InsightAgent.

    Configure your Twilio number's webhook URL to:
    POST https://your-domain.com/api/webhook/whatsapp
    """
    log.info("whatsapp_inbound", from_=From, body_preview=Body[:60])

    # Security: only respond to the configured owner number
    if OWNER_NUMBER and From != OWNER_NUMBER:
        log.warning("unauthorized_whatsapp", from_=From)
        return _twiml_response("⛔ Unauthorized number. Contact your BistroBrain admin.")

    body = Body.strip()

    # Handle help command
    if body.lower() in ("help", "hi", "hello", "start"):
        return _twiml_response(
            "👋 Hi! I'm *BistroBrain*, your AI Store Manager.\n\n"
            "You can ask me anything about your restaurant:\n"
            "• 'Why were margins low yesterday?'\n"
            "• 'What's my best seller this week?'\n"
            "• 'Send me today's digest'\n\n"
            "Just type your question!"
        )

    # Handle digest shortcut
    if body.lower() in ("digest", "today", "cards", "summary"):
        return _twiml_response(
            "📋 Fetching today's digest... check your API at /api/digest "
            "or run: python scripts/run_daily_digest.py --send"
        )

    # Route everything else to InsightAgent
    try:
        agent = _get_agent()
        response = agent.answer(QueryRequest(question=body))

        reply = f"🧠 *BistroBrain*\n\n{response.answer}"
        if response.follow_up_suggestions:
            reply += "\n\n💡 *Try asking:*"
            for suggestion in response.follow_up_suggestions[:2]:
                reply += f"\n• {suggestion}"

        return _twiml_response(reply)

    except Exception as exc:
        log.error("whatsapp_insight_failed", error=str(exc))
        return _twiml_response(
            "Sorry, I couldn't process that right now. "
            "Please try again in a moment. 🙏"
        )
