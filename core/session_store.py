"""
ADK Session Store
=================
Configures the Google ADK DatabaseSessionService which persists multi-turn
conversation state across requests.

The service automatically manages four tables:
  StorageSession     — session metadata (app_name, user_id, session_id, state)
  StorageAppState    — app-level shared state (persists across all sessions)
  StorageUserState   — per-user state (persists across that user's sessions)
  StorageEvent       — conversation events with role/content/tool_calls

Supported backends:
  SQLite   (dev)    — sqlite+aiosqlite:///./data/sessions.db
  PostgreSQL (prod) — postgresql+asyncpg://user:pass@host/dbname

The session_service_uri is passed to get_fast_api_app() so the ADK Dev UI
and /run endpoint both use the same persistent store.
"""
from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

# Singleton session service (created once at startup)
_session_service = None


def get_session_service():
    """
    Return the singleton ADK session service.

    Tries DatabaseSessionService first (persistent, multi-node safe).
    Falls back to InMemorySessionService if ADK is not installed.
    """
    global _session_service
    if _session_service is not None:
        return _session_service

    from config.settings import get_settings
    settings = get_settings()

    db_url = (
        settings.session.session_service_uri
        or settings.session.session_db_url
    )

    try:
        from google.adk.sessions import DatabaseSessionService
        _session_service = DatabaseSessionService(db_url=db_url)
        logger.info(
            "ADK DatabaseSessionService initialized",
            db_url=_mask_url(db_url),
        )
    except ImportError:
        logger.warning("google-adk not installed — using InMemorySessionService")
        try:
            from google.adk.sessions import InMemorySessionService
            _session_service = InMemorySessionService()
        except ImportError:
            _session_service = None

    return _session_service


def get_session_service_uri() -> str | None:
    """
    Return the session_service_uri to pass to get_fast_api_app().

    get_fast_api_app expects the raw database URL string, not a service object.
    Returns None to fall back to ADK's default InMemorySessionService.
    """
    from config.settings import get_settings
    settings = get_settings()

    uri = settings.session.session_service_uri or settings.session.session_db_url
    # Only return a URI if it's not the default ephemeral sqlite
    if uri and "sessions.db" in uri:
        # Still a valid persistent URI — return it
        return uri
    return uri or None


def _mask_url(url: str) -> str:
    """Mask password in database URL for logging."""
    import re
    return re.sub(r"(://[^:@]+:)[^@]+@", r"\1****@", url)
