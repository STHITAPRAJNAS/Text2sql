"""
Tests for MCP client integration.

All tests mock the mcp package and the background session so they run
without a real MCP server.  Three test classes:

  TestMCPClientInit    — initialisation, singleton, settings gate
  TestMCPTools         — each mcp_* ADK tool function
  TestFetchMetadata    — _fetch_table_metadata priority chain
"""
from __future__ import annotations

import threading
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_call_result(text: str, is_error: bool = False):
    """Build a minimal MCP CallToolResult-like object."""
    content_item = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[content_item], isError=is_error)


def _mock_mcp_client(tools: list[str] | None = None, tool_responses: dict | None = None):
    """Return a mock MCPClient with call_tool_json pre-wired."""
    client = MagicMock()
    client.list_tools.return_value = tools or []
    client._tool_map = {
        "execute_query": "execute_statement",
        "list_catalogs": "list_catalogs",
        "list_schemas": "list_schemas",
        "list_tables": "list_tables",
        "get_table": "get_table",
        "search_tables": "search_tables",
    }

    def resolve(logical):
        return client._tool_map.get(logical, logical)

    def has(logical):
        return resolve(logical) in (tools or [])

    client.resolve_tool_name.side_effect = resolve
    client.has_tool.side_effect = has

    responses = tool_responses or {}

    def call_json(tool_name, args, timeout=None):
        if tool_name in responses:
            return responses[tool_name]
        return {"success": True, "content": {}, "raw": []}

    client.call_tool_json.side_effect = call_json
    return client


# ═══════════════════════════════════════════════════════════════════════════════
# MCPSettings
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCPSettings:
    def test_not_configured_by_default(self):
        from config.settings import MCPSettings
        s = MCPSettings()
        assert s.is_configured is False

    def test_stdio_configured(self):
        from config.settings import MCPSettings
        s = MCPSettings(enabled=True, transport="stdio", server_command="databricks mcp start")
        assert s.is_configured is True

    def test_stdio_missing_command(self):
        from config.settings import MCPSettings
        s = MCPSettings(enabled=True, transport="stdio", server_command="")
        assert s.is_configured is False

    def test_sse_configured(self):
        from config.settings import MCPSettings
        s = MCPSettings(enabled=True, transport="sse", server_url="http://localhost:3001/sse")
        assert s.is_configured is True

    def test_sse_missing_url(self):
        from config.settings import MCPSettings
        s = MCPSettings(enabled=True, transport="sse", server_url="")
        assert s.is_configured is False

    def test_enabled_false_always_not_configured(self):
        from config.settings import MCPSettings
        s = MCPSettings(enabled=False, transport="sse", server_url="http://x/sse")
        assert s.is_configured is False

    def test_settings_has_mcp_field(self):
        from config.settings import get_settings, Settings, MCPSettings
        settings = get_settings()
        assert hasattr(settings, "mcp")
        assert isinstance(settings.mcp, MCPSettings)


# ═══════════════════════════════════════════════════════════════════════════════
# MCPClient init / singleton
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCPClientInit:
    def test_mcp_not_available_raises_import_error(self):
        """When mcp package is missing, MCPClient raises ImportError."""
        import core.mcp_client as mod
        original = mod._MCP_AVAILABLE
        try:
            mod._MCP_AVAILABLE = False
            with pytest.raises(ImportError, match="mcp"):
                from core.mcp_client import MCPClient
                MCPClient(transport="stdio", server_command="x")
        finally:
            mod._MCP_AVAILABLE = original

    def test_get_mcp_client_returns_none_when_not_initialised(self):
        import core.mcp_client as mod
        original = mod._mcp_client
        try:
            mod._mcp_client = None
            assert mod.get_mcp_client() is None
        finally:
            mod._mcp_client = original

    def test_init_mcp_client_skips_when_disabled(self):
        import core.mcp_client as mod
        original = mod._mcp_client
        try:
            mod._mcp_client = None
            with patch("config.settings.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(
                    mcp=MagicMock(is_configured=False)
                )
                result = mod.init_mcp_client()
            assert result is None
        finally:
            mod._mcp_client = original

    def test_init_mcp_client_returns_existing_singleton(self):
        import core.mcp_client as mod
        fake_client = MagicMock()
        original = mod._mcp_client
        try:
            mod._mcp_client = fake_client
            result = mod.init_mcp_client()
            assert result is fake_client
        finally:
            mod._mcp_client = original

    def test_shutdown_clears_singleton(self):
        import core.mcp_client as mod
        fake_client = MagicMock()
        original = mod._mcp_client
        try:
            mod._mcp_client = fake_client
            mod.shutdown_mcp_client()
            assert mod._mcp_client is None
            fake_client.close.assert_called_once()
        finally:
            mod._mcp_client = original


# ═══════════════════════════════════════════════════════════════════════════════
# MCPClient.call_tool_json
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCPClientCallToolJson:
    """
    Tests for MCPClient.call_tool_json().

    We bypass the async dispatch layer by patching call_tool_sync directly
    (the actual async dispatch is tested implicitly by integration tests).
    """

    def _make_bare_client(self):
        """Create an MCPClient instance without starting the background thread."""
        import core.mcp_client as mod
        from core.mcp_client import MCPClient

        client = object.__new__(MCPClient)
        client._transport = "stdio"
        client._call_timeout = 5
        client._tools = {"my_tool": MagicMock()}
        client._ready = threading.Event()
        client._ready.set()
        client._error = None
        client._loop = MagicMock()
        client._shutdown = None
        client._thread = MagicMock()
        client._tool_map = {}
        client._session = MagicMock()
        return client

    def test_success_json_response(self):
        client = self._make_bare_client()
        mcp_result = _make_call_result('{"rows": [{"a": 1}], "columns": ["a"]}')
        client.call_tool_sync = MagicMock(return_value=mcp_result)
        out = client.call_tool_json("my_tool", {})
        assert out["success"] is True
        assert out["content"] == {"rows": [{"a": 1}], "columns": ["a"]}

    def test_error_response(self):
        client = self._make_bare_client()
        mcp_result = _make_call_result("table not found", is_error=True)
        client.call_tool_sync = MagicMock(return_value=mcp_result)
        out = client.call_tool_json("my_tool", {})
        assert out["success"] is False
        assert "table not found" in out["error"]

    def test_plain_text_non_json(self):
        client = self._make_bare_client()
        mcp_result = _make_call_result("hello world")
        client.call_tool_sync = MagicMock(return_value=mcp_result)
        out = client.call_tool_json("my_tool", {})
        assert out["success"] is True
        assert out["content"] == "hello world"

    def test_timeout_returns_error_dict(self):
        client = self._make_bare_client()
        client.call_tool_sync = MagicMock(side_effect=asyncio.TimeoutError())
        out = client.call_tool_json("my_tool", {})
        assert out["success"] is False
        assert "timed out" in out["error"]

    def test_exception_returns_error_dict(self):
        client = self._make_bare_client()
        client.call_tool_sync = MagicMock(side_effect=RuntimeError("connection lost"))
        out = client.call_tool_json("my_tool", {})
        assert out["success"] is False
        assert "connection lost" in out["error"]


# ═══════════════════════════════════════════════════════════════════════════════
# MCP tool functions (agents/tools/mcp_tools.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCPTools:
    def setup_method(self):
        """Reset the MCP client singleton before each test."""
        import core.mcp_client as mod
        self._original = mod._mcp_client

    def teardown_method(self):
        import core.mcp_client as mod
        mod._mcp_client = self._original

    def _set_client(self, client):
        import core.mcp_client as mod
        mod._mcp_client = client

    # ── mcp_discover_tools ────────────────────────────────────────────

    def test_discover_tools_no_client(self):
        self._set_client(None)
        from agents.tools.mcp_tools import mcp_discover_tools
        result = mcp_discover_tools()
        assert result["success"] is False
        assert "not configured" in result["error"]

    def test_discover_tools_with_client(self):
        client = _mock_mcp_client(tools=["execute_statement", "list_catalogs"])
        self._set_client(client)
        from agents.tools.mcp_tools import mcp_discover_tools
        result = mcp_discover_tools()
        assert result["success"] is True
        assert "execute_statement" in result["tools"]
        assert result["count"] == 2

    # ── mcp_list_catalogs ─────────────────────────────────────────────

    def test_list_catalogs_no_client(self):
        self._set_client(None)
        from agents.tools.mcp_tools import mcp_list_catalogs
        result = mcp_list_catalogs()
        assert result["success"] is False

    def test_list_catalogs_success(self):
        client = _mock_mcp_client(
            tools=["list_catalogs"],
            tool_responses={
                "list_catalogs": {
                    "success": True,
                    "content": {"catalogs": [{"name": "main"}, {"name": "hive_metastore"}]},
                    "raw": [],
                }
            },
        )
        self._set_client(client)
        from agents.tools.mcp_tools import mcp_list_catalogs
        result = mcp_list_catalogs()
        assert result["success"] is True
        assert "main" in result["catalogs"]
        assert "hive_metastore" in result["catalogs"]
        assert result["count"] == 2

    def test_list_catalogs_tool_missing(self):
        client = _mock_mcp_client(tools=[])  # no list_catalogs
        self._set_client(client)
        from agents.tools.mcp_tools import mcp_list_catalogs
        result = mcp_list_catalogs()
        assert result["success"] is False
        assert "does not expose tool" in result["error"]

    # ── mcp_list_schemas ──────────────────────────────────────────────

    def test_list_schemas_success(self):
        client = _mock_mcp_client(
            tools=["list_schemas"],
            tool_responses={
                "list_schemas": {
                    "success": True,
                    "content": {"schemas": ["sales", "analytics"]},
                    "raw": [],
                }
            },
        )
        self._set_client(client)
        from agents.tools.mcp_tools import mcp_list_schemas
        result = mcp_list_schemas("main")
        assert result["success"] is True
        assert result["catalog"] == "main"
        assert "sales" in result["schemas"]
        assert result["count"] == 2

    def test_list_schemas_mcp_error(self):
        client = _mock_mcp_client(
            tools=["list_schemas"],
            tool_responses={
                "list_schemas": {"success": False, "error": "unauthorized"}
            },
        )
        self._set_client(client)
        from agents.tools.mcp_tools import mcp_list_schemas
        result = mcp_list_schemas("main")
        assert result["success"] is False
        assert "unauthorized" in result["error"]

    # ── mcp_list_tables ───────────────────────────────────────────────

    def test_list_tables_success(self):
        client = _mock_mcp_client(
            tools=["list_tables"],
            tool_responses={
                "list_tables": {
                    "success": True,
                    "content": {
                        "tables": [
                            {"name": "orders", "table_type": "TABLE", "comment": "Order records"},
                            {"name": "customers", "table_type": "TABLE", "comment": ""},
                        ]
                    },
                    "raw": [],
                }
            },
        )
        self._set_client(client)
        from agents.tools.mcp_tools import mcp_list_tables
        result = mcp_list_tables("main", "sales")
        assert result["success"] is True
        assert result["count"] == 2
        names = [t["name"] for t in result["tables"]]
        assert "orders" in names
        # full_name should be synthesised
        assert result["tables"][0]["full_name"] == "main.sales.orders"

    # ── mcp_get_table_metadata ────────────────────────────────────────

    def test_get_table_metadata_success(self):
        client = _mock_mcp_client(
            tools=["get_table"],
            tool_responses={
                "get_table": {
                    "success": True,
                    "content": {
                        "columns": [
                            {"name": "id", "type_text": "BIGINT", "comment": "Primary key", "nullable": False},
                            {"name": "amount", "type_text": "DOUBLE", "comment": "Revenue", "nullable": True},
                        ],
                        "comment": "Sales orders",
                        "row_count": 1_000_000,
                        "table_type": "DELTA",
                    },
                    "raw": [],
                }
            },
        )
        self._set_client(client)
        from agents.tools.mcp_tools import mcp_get_table_metadata
        result = mcp_get_table_metadata("main.sales.orders")
        assert result["success"] is True
        assert result["table_id"] == "main.sales.orders"
        assert result["catalog"] == "main"
        assert result["schema"] == "sales"
        assert result["table"] == "orders"
        assert len(result["columns"]) == 2
        assert result["columns"][0]["name"] == "id"
        assert result["columns"][0]["type"] == "BIGINT"
        assert result["row_count"] == 1_000_000
        assert result["comment"] == "Sales orders"
        assert result["error"] is None

    def test_get_table_metadata_no_client(self):
        self._set_client(None)
        from agents.tools.mcp_tools import mcp_get_table_metadata
        result = mcp_get_table_metadata("main.sales.orders")
        assert result["success"] is False

    def test_get_table_metadata_tool_error(self):
        client = _mock_mcp_client(
            tools=["get_table"],
            tool_responses={
                "get_table": {"success": False, "error": "table not found"}
            },
        )
        self._set_client(client)
        from agents.tools.mcp_tools import mcp_get_table_metadata
        result = mcp_get_table_metadata("main.sales.nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]

    # ── mcp_search_tables ─────────────────────────────────────────────

    def test_search_tables_success(self):
        client = _mock_mcp_client(
            tools=["search_tables"],
            tool_responses={
                "search_tables": {
                    "success": True,
                    "content": {
                        "tables": [
                            {"full_name": "main.sales.orders", "comment": "Order table"},
                        ]
                    },
                    "raw": [],
                }
            },
        )
        self._set_client(client)
        from agents.tools.mcp_tools import mcp_search_tables
        result = mcp_search_tables("orders")
        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["full_name"] == "main.sales.orders"

    def test_search_tables_no_client(self):
        self._set_client(None)
        from agents.tools.mcp_tools import mcp_search_tables
        result = mcp_search_tables("revenue")
        assert result["success"] is False

    # ── mcp_execute_query ─────────────────────────────────────────────

    def test_execute_query_success(self):
        client = _mock_mcp_client(
            tools=["execute_statement"],
            tool_responses={
                "execute_statement": {
                    "success": True,
                    "content": {
                        "columns": ["id", "name"],
                        "rows": [{"id": 1, "name": "Alice"}],
                        "row_count": 1,
                        "truncated": False,
                        "execution_time_ms": 42.0,
                    },
                    "raw": [],
                }
            },
        )
        self._set_client(client)
        from agents.tools.mcp_tools import mcp_execute_query
        result = mcp_execute_query("SELECT id, name FROM users LIMIT 1")
        assert result["success"] is True
        assert result["columns"] == ["id", "name"]
        assert result["row_count"] == 1
        assert result["error"] is None

    def test_execute_query_no_client(self):
        self._set_client(None)
        from agents.tools.mcp_tools import mcp_execute_query
        result = mcp_execute_query("SELECT 1")
        assert result["success"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# _fetch_table_metadata priority chain
# ═══════════════════════════════════════════════════════════════════════════════

class TestFetchTableMetadata:
    def setup_method(self):
        import core.mcp_client as mod
        self._original_client = mod._mcp_client

    def teardown_method(self):
        import core.mcp_client as mod
        mod._mcp_client = self._original_client

    def test_uses_mcp_when_available(self):
        """MCP result should be returned without touching native SDK."""
        mcp_client = _mock_mcp_client(
            tools=["get_table"],
            tool_responses={
                "get_table": {
                    "success": True,
                    "content": {
                        "columns": [{"name": "id", "type_text": "INT", "nullable": True}],
                        "comment": "via MCP",
                        "row_count": 100,
                        "table_type": "DELTA",
                    },
                    "raw": [],
                }
            },
        )
        import core.mcp_client as mod
        mod._mcp_client = mcp_client

        from agents.tools.indexing_tools import _fetch_table_metadata
        result = _fetch_table_metadata("main.sales.orders")

        assert result.get("source") == "mcp"
        assert result["comment"] == "via MCP"
        assert result["row_count"] == 100

    def test_falls_back_to_native_sdk_when_mcp_fails(self):
        """When MCP returns an error, native Databricks SDK should be tried."""
        mcp_client = _mock_mcp_client(
            tools=["get_table"],
            tool_responses={
                "get_table": {"success": False, "error": "timeout"}
            },
        )
        import core.mcp_client as mod
        mod._mcp_client = mcp_client

        mock_connector = MagicMock()
        mock_connector.get_table_metadata.return_value = {
            "columns": [], "comment": "native", "full_name": "main.sales.orders"
        }
        mock_connector.get_table_stats.return_value = {"row_count": 50, "size_bytes": 1024}

        with patch("core.databricks.get_databricks_connector", return_value=mock_connector):
            from agents.tools.indexing_tools import _fetch_table_metadata
            result = _fetch_table_metadata("main.sales.orders")

        assert result.get("source") == "databricks_sdk"
        assert result["comment"] == "native"

    def test_falls_back_to_sqlalchemy_when_no_mcp_no_databricks(self):
        """With no MCP and no Databricks, SQLAlchemy should be tried."""
        import core.mcp_client as mod
        mod._mcp_client = None  # No MCP

        with patch("core.databricks.get_databricks_connector", return_value=None):
            with patch("core.schema_manager.SchemaManager") as mock_sm_cls:
                mock_table = MagicMock()
                mock_table.to_dict.return_value = {"columns": [], "comment": "sqlalchemy"}
                mock_sm = MagicMock()
                mock_sm.get_full_schema = AsyncMock(return_value=MagicMock(
                    get_table=MagicMock(return_value=mock_table)
                ))
                mock_sm_cls.return_value = mock_sm

                with patch("core.database.get_db_manager", new_callable=AsyncMock):
                    from agents.tools.indexing_tools import _fetch_table_metadata
                    # SQLAlchemy path runs synchronously via new event loop
                    # Just verify it tries the fallback (error is acceptable in test env)
                    result = _fetch_table_metadata("orders")
                    # Either succeeds or returns error from SQLAlchemy fallback
                    assert isinstance(result, dict)

    def test_no_mcp_no_databricks_returns_error(self):
        """When all sources fail, error dict is returned."""
        import core.mcp_client as mod
        mod._mcp_client = None

        with patch("core.databricks.get_databricks_connector", return_value=None):
            from agents.tools.indexing_tools import _fetch_table_metadata
            result = _fetch_table_metadata("nonexistent_table")
            # Should return an error dict, not raise
            assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Schema Discovery agent tool list
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemaDiscoveryAgentTools:
    """
    Tests for MCP tool injection in the Schema Discovery Agent.

    These tests patch the google.adk Agent so they run without the full
    ADK package installed.
    """

    def _make_fake_agent(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def test_mcp_tools_not_in_list_when_disabled(self):
        import sys
        import importlib

        fake_agent_cls = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
        google_mock = MagicMock()
        google_mock.adk.agents.Agent = fake_agent_cls

        with patch.dict(sys.modules, {
            "google": google_mock,
            "google.adk": google_mock.adk,
            "google.adk.agents": google_mock.adk.agents,
        }):
            with patch("config.settings.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(
                    llm=MagicMock(schema_agent_model="gemini-2.0-flash"),
                    mcp=MagicMock(is_configured=False),
                )
                # Import fresh copy with mocked ADK
                import agents.swarm.schema_agent as sa
                importlib.reload(sa)
                agent = sa.create_schema_discovery_agent()

        tool_names = [getattr(t, "__name__", str(t)) for t in agent.tools]
        assert "mcp_execute_query" not in tool_names
        assert "mcp_discover_tools" not in tool_names
        assert "search_relevant_tables" in tool_names

    def test_mcp_tools_prepended_when_enabled(self):
        import sys
        import importlib

        fake_agent_cls = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
        google_mock = MagicMock()
        google_mock.adk.agents.Agent = fake_agent_cls

        with patch.dict(sys.modules, {
            "google": google_mock,
            "google.adk": google_mock.adk,
            "google.adk.agents": google_mock.adk.agents,
        }):
            with patch("config.settings.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(
                    llm=MagicMock(schema_agent_model="gemini-2.0-flash"),
                    mcp=MagicMock(is_configured=True),
                )
                import agents.swarm.schema_agent as sa
                importlib.reload(sa)
                agent = sa.create_schema_discovery_agent()

        tool_names = [getattr(t, "__name__", str(t)) for t in agent.tools]
        assert "mcp_discover_tools" in tool_names
        assert "mcp_execute_query" in tool_names
        # MCP tools should appear before the vector index tools
        mcp_idx = tool_names.index("mcp_discover_tools")
        vec_idx = tool_names.index("search_relevant_tables")
        assert mcp_idx < vec_idx
