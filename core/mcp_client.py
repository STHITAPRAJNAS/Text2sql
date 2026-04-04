"""
MCP (Model Context Protocol) Client
=====================================
Connects to any MCP server (Databricks or custom) and exposes its tools
to PRISM agents via a persistent background session.

Transport options:
  stdio — Launches a local subprocess (e.g. ``databricks mcp start``)
  sse   — Connects to a running HTTP/SSE server (e.g. http://localhost:3001/sse)

The client runs the MCP session in a dedicated background thread so the
async context managers stay alive for the process lifetime.  PRISM agent
tool functions call ``call_tool_sync()`` which submits a coroutine to the
background event loop and blocks until the result arrives.

Lifecycle:
    init_mcp_client()          ← call once at app startup
    get_mcp_client()           ← returns None if not configured / failed
    client.call_tool_sync(...) ← synchronous tool invocation from agent tools
    shutdown_mcp_client()      ← call at app shutdown (optional, daemon thread)

Databricks MCP server standard tool names (configurable via MCPSettings):
    execute_statement  — execute SQL on a SQL warehouse
    list_catalogs      — list Unity Catalog catalogs
    list_schemas       — list schemas in a catalog
    list_tables        — list tables in a catalog.schema
    get_table          — get column metadata for a table
    search_tables      — full-text search over table names/comments
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ── Optional MCP import (graceful degradation when not installed) ──────────────
try:
    from mcp import ClientSession
    from mcp.types import Tool as MCPTool
    _MCP_AVAILABLE = True
except ImportError:
    ClientSession = None  # type: ignore
    MCPTool = None  # type: ignore
    _MCP_AVAILABLE = False


# ── Singleton ──────────────────────────────────────────────────────────────────
_mcp_client: "MCPClient | None" = None


class MCPConnectionError(RuntimeError):
    """Raised when the MCP client cannot connect to the server."""


class MCPClient:
    """
    Persistent MCP client session running in a background thread.

    The stdio/SSE context managers are kept alive by an asyncio.Event
    (``_shutdown``) that is only set when ``close()`` is called.  All tool
    calls are dispatched to the background loop via
    ``asyncio.run_coroutine_threadsafe()``.
    """

    def __init__(
        self,
        transport: str = "stdio",
        server_command: str = "",
        server_url: str = "",
        discovery_timeout: int = 10,
        call_timeout: int = 30,
        # Tool name mapping (override for non-standard servers)
        tool_execute_query: str = "execute_statement",
        tool_list_catalogs: str = "list_catalogs",
        tool_list_schemas: str = "list_schemas",
        tool_list_tables: str = "list_tables",
        tool_get_table: str = "get_table",
        tool_search_tables: str = "search_tables",
    ) -> None:
        if not _MCP_AVAILABLE:
            raise ImportError(
                "The 'mcp' package is required for MCP client support. "
                "Install it with: pip install mcp>=1.0.0"
            )

        self._transport = transport
        self._server_command = server_command
        self._server_url = server_url
        self._discovery_timeout = discovery_timeout
        self._call_timeout = call_timeout

        # Tool name mapping
        self._tool_map = {
            "execute_query": tool_execute_query,
            "list_catalogs": tool_list_catalogs,
            "list_schemas": tool_list_schemas,
            "list_tables": tool_list_tables,
            "get_table": tool_get_table,
            "search_tables": tool_search_tables,
        }

        # Runtime state — set inside the background loop
        self._session: ClientSession | None = None
        self._tools: dict[str, Any] = {}       # name → MCPTool schema
        self._ready = threading.Event()
        self._error: Exception | None = None

        # Start background event loop
        self._loop = asyncio.new_event_loop()
        self._shutdown: asyncio.Event | None = None   # created inside the loop
        self._thread = threading.Thread(
            target=self._run_background_loop,
            name="mcp-session",
            daemon=True,
        )
        self._thread.start()

        # Block until connected (or timeout)
        if not self._ready.wait(timeout=discovery_timeout):
            raise MCPConnectionError(
                f"MCP client did not connect within {discovery_timeout}s "
                f"(transport={transport})"
            )
        if self._error:
            raise MCPConnectionError(
                f"MCP client failed to connect: {self._error}"
            ) from self._error

        logger.info(
            "MCP client connected",
            transport=transport,
            tools=list(self._tools.keys()),
        )

    # ------------------------------------------------------------------ #
    # Background loop                                                      #
    # ------------------------------------------------------------------ #

    def _run_background_loop(self) -> None:
        """Entry point for the background daemon thread."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._background_session())
        except Exception as exc:
            logger.error("MCP background session error", error=str(exc))
        finally:
            self._loop.close()

    async def _background_session(self) -> None:
        """Maintain the MCP session context manager until shutdown."""
        self._shutdown = asyncio.Event()

        try:
            if self._transport == "stdio":
                await self._run_stdio_session()
            elif self._transport == "sse":
                await self._run_sse_session()
            else:
                raise ValueError(f"Unknown MCP transport: {self._transport!r}")
        except Exception as exc:
            self._error = exc
            self._ready.set()   # unblock the constructor

    async def _run_stdio_session(self) -> None:
        from mcp.client.stdio import stdio_client
        from mcp import StdioServerParameters

        parts = self._server_command.split()
        if not parts:
            raise ValueError("MCP_SERVER_COMMAND is empty for stdio transport")

        params = StdioServerParameters(command=parts[0], args=parts[1:])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await self._setup_session(session)
                assert self._shutdown is not None
                await self._shutdown.wait()      # keep alive

    async def _run_sse_session(self) -> None:
        from mcp.client.sse import sse_client

        if not self._server_url:
            raise ValueError("MCP_SERVER_URL is empty for sse transport")

        async with sse_client(self._server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await self._setup_session(session)
                assert self._shutdown is not None
                await self._shutdown.wait()      # keep alive

    async def _setup_session(self, session: "ClientSession") -> None:
        """Initialize session and discover tools."""
        self._session = session
        await session.initialize()
        result = await session.list_tools()
        self._tools = {t.name: t for t in result.tools}
        self._ready.set()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def list_tools(self) -> list[str]:
        """Return names of all tools advertised by the MCP server."""
        return list(self._tools.keys())

    def get_tool_schema(self, name: str) -> dict[str, Any] | None:
        """Return the JSON schema for a tool, or None if not found."""
        tool = self._tools.get(name)
        if tool is None:
            return None
        schema = getattr(tool, "inputSchema", None)
        if schema is None:
            return None
        # Pydantic model or plain dict
        return schema.model_dump() if hasattr(schema, "model_dump") else dict(schema)

    def resolve_tool_name(self, logical_name: str) -> str:
        """
        Map a logical PRISM operation to the actual MCP tool name.

        If the mapped name is not available on this server, falls back
        to the logical name itself (allows custom servers that happen
        to use the same names).
        """
        mapped = self._tool_map.get(logical_name, logical_name)
        if mapped in self._tools:
            return mapped
        # Fallback: try the logical name directly
        if logical_name in self._tools:
            return logical_name
        return mapped   # Return mapped name even if absent (caller will handle error)

    def has_tool(self, logical_name: str) -> bool:
        """Return True if the server exposes the tool for this logical operation."""
        return self.resolve_tool_name(logical_name) in self._tools

    def call_tool_sync(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: int | None = None,
    ) -> Any:
        """
        Synchronously call an MCP tool.  Dispatches to the background loop.

        Args:
            tool_name:  Actual MCP tool name (use ``resolve_tool_name`` first).
            arguments:  Tool input arguments dict.
            timeout:    Max seconds to wait (defaults to ``call_timeout``).

        Returns:
            Raw MCP result (CallToolResult) — callers should inspect
            ``.content`` or convert via ``parse_result()``.
        """
        if not self._session:
            raise RuntimeError("MCP session not available")

        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(tool_name, arguments),
            self._loop,
        )
        return future.result(timeout=timeout or self._call_timeout)

    def call_tool_json(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Call a tool and return the result as a plain dict.

        Parses the MCP ``CallToolResult`` content list into a normalised dict:
          - ``{"success": True, "content": <text or dict>, "raw": [...]}``.
          - On error: ``{"success": False, "error": "<msg>"}``.
        """
        try:
            result = self.call_tool_sync(tool_name, arguments, timeout=timeout)
        except asyncio.TimeoutError:
            return {"success": False, "error": f"MCP tool {tool_name!r} timed out"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        # Parse MCP content list
        content_items = getattr(result, "content", []) or []
        is_error = getattr(result, "isError", False)

        if is_error:
            msgs = [
                getattr(c, "text", str(c))
                for c in content_items
                if hasattr(c, "text")
            ]
            return {"success": False, "error": " ".join(msgs) or "MCP tool returned an error"}

        # Collect text + embedded JSON from content items
        texts = []
        parsed_json: dict[str, Any] | None = None

        for item in content_items:
            item_type = getattr(item, "type", "text")
            if item_type == "text":
                text = getattr(item, "text", "")
                texts.append(text)
                # Attempt to parse as JSON
                if parsed_json is None and text.strip().startswith("{"):
                    try:
                        import json
                        parsed_json = json.loads(text)
                    except Exception:
                        pass
            elif item_type == "resource":
                resource = getattr(item, "resource", None)
                if resource:
                    text = getattr(resource, "text", None) or str(resource)
                    texts.append(text)

        combined_text = "\n".join(texts)

        return {
            "success": True,
            "content": parsed_json if parsed_json is not None else combined_text,
            "raw": [getattr(c, "text", str(c)) for c in content_items],
        }

    def close(self) -> None:
        """Signal the background session to exit."""
        if self._shutdown is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._shutdown.set)
        self._thread.join(timeout=5)
        logger.info("MCP client closed")


# ── Module-level singleton helpers ────────────────────────────────────────────

def get_mcp_client() -> MCPClient | None:
    """Return the singleton MCP client, or None if not initialised."""
    return _mcp_client


def init_mcp_client() -> MCPClient | None:
    """
    Initialise the MCP client from application settings.

    Silently returns None (with a log warning) if:
      - MCP is not enabled (``MCP_ENABLED=false``)
      - The ``mcp`` package is not installed
      - The server is unreachable within ``discovery_timeout``
    """
    global _mcp_client

    if _mcp_client is not None:
        return _mcp_client

    try:
        from config.settings import get_settings
        settings = get_settings()
        mcp_cfg = settings.mcp
    except Exception as exc:
        logger.debug("Could not read MCP settings", error=str(exc))
        return None

    if not mcp_cfg.is_configured:
        return None

    if not _MCP_AVAILABLE:
        logger.warning(
            "MCP is enabled but the 'mcp' package is not installed. "
            "Run: pip install mcp>=1.0.0"
        )
        return None

    try:
        _mcp_client = MCPClient(
            transport=mcp_cfg.transport,
            server_command=mcp_cfg.server_command,
            server_url=mcp_cfg.server_url,
            discovery_timeout=mcp_cfg.discovery_timeout,
            call_timeout=mcp_cfg.call_timeout,
            tool_execute_query=mcp_cfg.tool_execute_query,
            tool_list_catalogs=mcp_cfg.tool_list_catalogs,
            tool_list_schemas=mcp_cfg.tool_list_schemas,
            tool_list_tables=mcp_cfg.tool_list_tables,
            tool_get_table=mcp_cfg.tool_get_table,
            tool_search_tables=mcp_cfg.tool_search_tables,
        )
        return _mcp_client
    except ImportError as exc:
        logger.warning("MCP package missing", error=str(exc))
    except MCPConnectionError as exc:
        logger.warning("MCP connection failed — falling back to native connector", error=str(exc))
    except Exception as exc:
        logger.error("Unexpected error initialising MCP client", error=str(exc))

    return None


def shutdown_mcp_client() -> None:
    """Close the MCP client at application shutdown."""
    global _mcp_client
    if _mcp_client is not None:
        _mcp_client.close()
        _mcp_client = None
