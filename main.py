"""
Text2SQL PRISM — Main Entry Point
Enterprise Text-to-SQL system using Google ADK PRISM Swarm with Deep Think

Usage:
  # Start the API server
  python main.py serve

  # Interactive CLI mode
  python main.py query "What are the top 10 customers by revenue?"

  # Load few-shot examples
  python main.py load-examples ./data/examples.json

  # Discover schema
  python main.py schema
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# Configure structured logging early
import logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = structlog.get_logger(__name__)
console = Console()
app = typer.Typer(
    name="text2sql",
    help="Text2SQL PRISM — Enterprise Text-to-SQL with Google ADK",
    add_completion=False,
)


def _setup_env():
    """Load environment variables from .env file and initialize OTel."""
    from pathlib import Path
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        logger.info("Environment loaded from .env")

    # Initialize OTel tracing + structlog correlation
    from config.settings import get_settings
    from core.telemetry import init_telemetry
    s = get_settings()
    init_telemetry(
        service_name="text2sql-prism",
        otlp_endpoint=s.observability.otel_exporter_otlp_endpoint,
        enabled=s.observability.enable_tracing,
    )

    # Initialize MCP client (no-op when MCP_ENABLED=false)
    from core.mcp_client import init_mcp_client
    init_mcp_client()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8080, help="Port to listen on"),
    workers: int = typer.Option(1, help="Number of workers"),
    reload: bool = typer.Option(False, help="Enable hot reload (dev only)"),
    log_level: str = typer.Option("info", help="Log level"),
):
    """Start the Text2SQL PRISM API server."""
    _setup_env()

    try:
        import uvicorn
        console.print(
            Panel.fit(
                "[bold green]Text2SQL PRISM API[/bold green]\n"
                f"[cyan]http://{host}:{port}[/cyan]\n"
                f"[dim]Docs: http://{host}:{port}/docs[/dim]",
                title="Starting Server",
            )
        )
        uvicorn.run(
            "api.app:app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level=log_level,
        )
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install uvicorn[/red]")
        sys.exit(1)


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural language question"),
    database: str = typer.Option("default", help="Database name"),
    execute: bool = typer.Option(True, help="Execute the generated SQL"),
    max_rows: int = typer.Option(20, help="Max rows to display"),
    show_sql: bool = typer.Option(True, help="Show the generated SQL"),
):
    """Run a single natural language query through the PRISM pipeline."""
    _setup_env()

    async def _run():
        from agents.runner import run_prism_query

        console.print(
            Panel.fit(
                f"[bold]{question}[/bold]",
                title="[cyan]PRISM Query[/cyan]",
            )
        )

        with console.status("[bold green]Running PRISM pipeline...[/bold green]"):
            result = await run_prism_query(
                query=question,
                database_name=database,
                max_rows=max_rows,
                execute_query=execute,
            )

        # Show SQL
        if show_sql and result.get("generated_sql"):
            sql = result.get("optimized_sql") or result.get("generated_sql")
            console.print("\n[bold blue]Generated SQL:[/bold blue]")
            console.print(Syntax(sql, "sql", theme="monokai", line_numbers=True))

        # Show answer
        if result.get("answer"):
            console.print(
                Panel(
                    result["answer"],
                    title="[green]Answer[/green]",
                    border_style="green",
                )
            )

        # Show results table
        if result.get("rows") and result.get("columns"):
            table = Table(title=f"Results ({result['row_count']} rows)")
            for col in result["columns"]:
                table.add_column(col, style="cyan")
            for row in result["rows"][:max_rows]:
                table.add_row(*[str(row.get(c, "")) for c in result["columns"]])
            console.print(table)
            if result.get("truncated"):
                console.print(f"[yellow]Results truncated at {max_rows} rows[/yellow]")

        # Show metadata
        console.print(
            f"\n[dim]Pipeline: {result.get('pipeline_time_ms', 0):.0f}ms | "
            f"Confidence: {result.get('confidence', 0):.0%} | "
            f"Stages: {', '.join(result.get('pipeline_stages', [])[:3])}...[/dim]"
        )

        if not result.get("success") and result.get("error"):
            console.print(f"\n[red]Error: {result['error']}[/red]")

    asyncio.run(_run())


@app.command()
def schema(
    samples: bool = typer.Option(True, help="Include sample column values"),
):
    """Display the database schema."""
    _setup_env()

    async def _run():
        from agents.tools.schema_tools import get_database_schema

        with console.status("[bold green]Discovering schema...[/bold green]"):
            schema_data = await get_database_schema(include_samples=samples)

        if "error" in schema_data:
            console.print(f"[red]Error: {schema_data['error']}[/red]")
            return

        console.print(
            Panel.fit(
                f"Database: [bold]{schema_data.get('database_name', 'unknown')}[/bold]\n"
                f"Dialect: {schema_data.get('dialect', 'unknown')}\n"
                f"Tables: {schema_data.get('total_tables', 0)}",
                title="[cyan]Database Schema[/cyan]",
            )
        )

        tables = schema_data.get("tables", {})
        for table_name, table_info in tables.items():
            cols = table_info.get("columns", [])
            table = Table(title=f"[bold]{table_name}[/bold]", show_header=True)
            table.add_column("Column", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Nullable")
            table.add_column("PK")
            table.add_column("FK")
            table.add_column("Samples", style="dim")

            for col in cols:
                fk = col.get("foreign_key")
                fk_str = f"{fk['table']}.{fk['column']}" if fk else ""
                samples_str = ", ".join(str(v) for v in col.get("sample_values", [])[:3])
                table.add_row(
                    col.get("name", ""),
                    col.get("data_type", ""),
                    "Yes" if col.get("is_nullable") else "No",
                    "✓" if col.get("is_primary_key") else "",
                    fk_str,
                    samples_str,
                )
            console.print(table)
            console.print()

    asyncio.run(_run())


@app.command(name="load-examples")
def load_examples(
    file_path: str = typer.Argument(..., help="Path to JSON file with examples"),
):
    """Load few-shot examples from a JSON file into the vector store."""
    _setup_env()

    from agents.tools.few_shot_tools import load_examples_from_file

    with console.status(f"[bold green]Loading examples from {file_path}...[/bold green]"):
        result = load_examples_from_file(file_path)

    if result.get("success"):
        console.print(
            Panel.fit(
                f"Loaded: [green]{result['loaded']}[/green] | "
                f"Failed: [red]{result['failed']}[/red] | "
                f"Total: {result['total']}",
                title="[cyan]Examples Loaded[/cyan]",
            )
        )
    else:
        console.print(f"[red]Failed to load examples: {result.get('error')}[/red]")


@app.command()
def setup_demo():
    """Set up a demo SQLite database with sample data for testing."""
    _setup_env()
    asyncio.run(_setup_demo_db())


async def _setup_demo_db():
    """Create and populate a demo SQLite database."""
    import aiosqlite
    from pathlib import Path

    db_path = Path("./data/demo.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]Setting up demo database...[/bold cyan]")

    async with aiosqlite.connect(db_path) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY,
                customer_name TEXT NOT NULL,
                email TEXT UNIQUE,
                country TEXT,
                segment TEXT,
                created_at DATE DEFAULT CURRENT_DATE
            );

            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY,
                product_name TEXT NOT NULL,
                category TEXT,
                unit_price REAL,
                is_active INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER REFERENCES customers(customer_id),
                order_date DATE,
                status TEXT,
                total_amount REAL
            );

            CREATE TABLE IF NOT EXISTS order_items (
                item_id INTEGER PRIMARY KEY,
                order_id INTEGER REFERENCES orders(order_id),
                product_id INTEGER REFERENCES products(product_id),
                quantity INTEGER,
                unit_price REAL,
                discount REAL DEFAULT 0
            );

            INSERT OR IGNORE INTO customers VALUES
                (1, 'Acme Corp', 'acme@example.com', 'USA', 'Enterprise', '2023-01-15'),
                (2, 'TechStart Inc', 'techstart@example.com', 'UK', 'SMB', '2023-03-20'),
                (3, 'Global Traders', 'global@example.com', 'India', 'Enterprise', '2023-02-10'),
                (4, 'Local Shop', 'local@example.com', 'USA', 'SMB', '2023-06-01'),
                (5, 'MegaCorp', 'mega@example.com', 'Germany', 'Enterprise', '2022-11-30');

            INSERT OR IGNORE INTO products VALUES
                (1, 'Widget Pro', 'Electronics', 99.99, 1),
                (2, 'Gadget Plus', 'Electronics', 149.99, 1),
                (3, 'Service Pack', 'Services', 299.99, 1),
                (4, 'Data Bundle', 'Software', 49.99, 1),
                (5, 'Legacy Item', 'Electronics', 19.99, 0);

            INSERT OR IGNORE INTO orders VALUES
                (1, 1, '2024-01-10', 'completed', 1249.93),
                (2, 2, '2024-01-15', 'completed', 299.99),
                (3, 3, '2024-02-01', 'completed', 3499.80),
                (4, 1, '2024-02-20', 'pending', 599.97),
                (5, 4, '2024-03-01', 'completed', 149.99),
                (6, 5, '2024-03-10', 'completed', 5999.50);

            INSERT OR IGNORE INTO order_items VALUES
                (1, 1, 1, 5, 99.99, 0),
                (2, 1, 3, 2, 299.99, 0.1),
                (3, 2, 3, 1, 299.99, 0),
                (4, 3, 2, 10, 149.99, 0.05),
                (5, 3, 4, 20, 49.99, 0),
                (6, 4, 1, 6, 99.99, 0),
                (7, 5, 2, 1, 149.99, 0),
                (8, 6, 3, 20, 299.99, 0);
        """)
        await db.commit()

    console.print(f"[green]Demo database created at: {db_path}[/green]")
    console.print("\nUpdate your .env file:")
    console.print(f"  DATABASE_URL=sqlite+aiosqlite:///{db_path.resolve()}")
    console.print("\nSample queries to try:")
    console.print('  python main.py query "Who are the top 3 customers by total orders?"')
    console.print('  python main.py query "What is the monthly revenue trend?"')
    console.print('  python main.py query "Which product categories generate the most revenue?"')


if __name__ == "__main__":
    app()
