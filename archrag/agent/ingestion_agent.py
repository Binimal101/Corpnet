"""Conversational ingestion agent for guided data onboarding.

This agent provides an interactive interface for producers to:
- Connect to their databases (with persistent credential storage)
- Discover and select data to index
- Configure sync preferences
- Run the ingestion pipeline
- Query the indexed data

The agent uses LLM-powered tool calling to intelligently guide
the user through the process.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from archrag.agent.connection_store import ConnectionStore
from archrag.agent.prompts import SYSTEM_PROMPT, GREETING_PROMPT, SUMMARY_PROMPT
from archrag.agent.tools import AgentTools, ToolResult

if TYPE_CHECKING:
    from archrag.ports.llm import LLMPort
    from archrag.services.orchestrator import ArchRAGOrchestrator

log = logging.getLogger(__name__)


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # Tool name for tool messages


@dataclass
class AgentState:
    """Current state of the agent."""
    session_id: int = 0
    messages: list[Message] = field(default_factory=list)
    current_connection: str | None = None
    awaiting_confirmation: bool = False
    pending_action: dict[str, Any] | None = None


class IngestionAgent:
    """Conversational agent for guiding data ingestion.
    
    This agent uses the LLM to understand user intent and call
    appropriate tools to connect databases, sync data, and more.
    
    Usage:
        from archrag.config import build_orchestrator
        from archrag.agent import IngestionAgent, ConnectionStore
        
        orch = build_orchestrator()
        store = ConnectionStore()
        agent = IngestionAgent(orch, orch._llm, store)
        
        # Interactive mode
        agent.run_interactive()
        
        # Or single message
        response = agent.chat("I want to index my people database")
    """
    
    def __init__(
        self,
        orchestrator: "ArchRAGOrchestrator",
        llm: "LLMPort",
        connection_store: ConnectionStore | None = None,
        max_tool_iterations: int = 5,
    ):
        """Initialize the agent.
        
        Args:
            orchestrator: The ArchRAG orchestrator instance.
            llm: LLM port for conversation and tool calling.
            connection_store: Persistent storage for connections.
            max_tool_iterations: Max tool calls per user message.
        """
        self._orch = orchestrator
        self._llm = llm
        self._store = connection_store or ConnectionStore()
        self._max_tool_iterations = max_tool_iterations
        
        # Initialize tools
        self._tools = AgentTools(orchestrator, self._store)
        
        # Agent state
        self._state = AgentState()
    
    def start_session(self) -> str:
        """Start a new conversation session."""
        self._state = AgentState()
        self._state.session_id = self._store.start_session()
        
        # Add system message
        self._state.messages.append(Message(
            role="system",
            content=SYSTEM_PROMPT,
        ))
        
        # Save greeting
        self._store.save_message(
            self._state.session_id,
            role="assistant",
            content=GREETING_PROMPT,
        )
        
        return GREETING_PROMPT
    
    def end_session(self) -> None:
        """End the current session."""
        if self._state.session_id:
            # Generate summary
            summary = self._generate_summary()
            self._store.end_session(self._state.session_id, summary)
    
    def chat(self, user_message: str) -> str:
        """Process a user message and return the response.
        
        This is the main entry point for conversation. It:
        1. Adds the user message to history
        2. Calls the LLM with tool definitions
        3. Executes any tool calls
        4. Returns the final response
        
        Args:
            user_message: The user's input.
            
        Returns:
            The agent's response.
        """
        # Add user message
        self._state.messages.append(Message(
            role="user",
            content=user_message,
        ))
        self._store.save_message(
            self._state.session_id,
            role="user",
            content=user_message,
        )
        
        # Run agent loop
        response = self._agent_loop()
        
        # Save assistant response
        self._store.save_message(
            self._state.session_id,
            role="assistant",
            content=response,
        )
        
        return response
    
    def _agent_loop(self) -> str:
        """Run the agent loop until we get a final response."""
        iterations = 0
        
        while iterations < self._max_tool_iterations:
            iterations += 1
            
            # Call LLM with tools
            response = self._call_llm_with_tools()
            
            # Check if we have tool calls
            if response.tool_calls:
                # Add assistant message with tool calls FIRST
                self._state.messages.append(Message(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                ))
                
                # THEN execute tools and add results to messages
                self._execute_tool_calls(response.tool_calls)
            else:
                # No tool calls, return the response
                self._state.messages.append(Message(
                    role="assistant",
                    content=response.content,
                ))
                return response.content
        
        # Max iterations reached
        return "I've reached the maximum number of operations. Let me know if you'd like me to continue."
    
    def _call_llm_with_tools(self) -> Any:
        """Call the LLM with the current messages and tool definitions."""
        # Convert messages to OpenAI format
        # OpenAI requires: tool messages must immediately follow assistant messages with tool_calls
        messages = []
        i = 0
        while i < len(self._state.messages):
            msg = self._state.messages[i]
            
            if msg.tool_calls:
                # Assistant message with tool calls
                # tool_calls should already be in OpenAI format from the LLM adapter
                messages.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": msg.tool_calls,
                })
                
                # Add the corresponding tool response messages that follow
                i += 1
                while i < len(self._state.messages) and self._state.messages[i].role == "tool":
                    tool_msg = self._state.messages[i]
                    messages.append({
                        "role": "tool",
                        "content": tool_msg.content,
                        "tool_call_id": tool_msg.tool_call_id,
                    })
                    i += 1
            elif msg.role == "tool":
                # Orphaned tool message (shouldn't happen with correct ordering)
                # Skip it to avoid errors
                i += 1
            else:
                # Regular message (system, user, assistant without tool_calls)
                messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })
                i += 1
        
        # Get tool schemas
        tools = self._tools.get_tool_schemas()
        
        # Call LLM
        return self._llm.chat_with_tools(messages, tools)
    
    def _execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        """Execute tool calls and add results to messages."""
        for tc in tool_calls:
            tool_name = tc.get("function", {}).get("name", "")
            tool_args_str = tc.get("function", {}).get("arguments", "{}")
            tool_call_id = tc.get("id", "")
            
            try:
                tool_args = json.loads(tool_args_str)
            except json.JSONDecodeError:
                tool_args = {}
            
            log.debug("Executing tool: %s(%s)", tool_name, tool_args)
            
            # Execute the tool
            result = self._tools.execute_tool(tool_name, tool_args)
            
            # Add tool result message
            self._state.messages.append(Message(
                role="tool",
                content=json.dumps(result.to_dict()),
                tool_call_id=tool_call_id,
                name=tool_name,
            ))
            
            # Save to session
            self._store.save_message(
                self._state.session_id,
                role="tool",
                content=json.dumps(result.to_dict()),
                tool_calls=[{"name": tool_name, "args": tool_args, "result": result.to_dict()}],
            )
    
    def _generate_summary(self) -> str:
        """Generate a summary of the session."""
        # Simple summary based on state
        if self._state.current_connection:
            return f"Connected to '{self._state.current_connection}' and synced data."
        return "Session ended."
    
    def run_interactive(self, greeting: bool = True) -> None:
        """Run an interactive CLI session.
        
        Args:
            greeting: Whether to show the greeting message.
        """
        print("\n🤖 ArchRAG Ingestion Assistant")
        print("━" * 40)
        
        if greeting:
            greeting_msg = self.start_session()
            print(f"\n{greeting_msg}\n")
        else:
            self.start_session()
        
        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    break
                
                if not user_input:
                    continue
                
                if user_input.lower() in ("quit", "exit", "q", "bye"):
                    print("\nGoodbye! Your session has been saved.\n")
                    break
                
                if user_input.lower() == "help":
                    self._show_help()
                    continue
                
                # Process message
                try:
                    response = self.chat(user_input)
                    print(f"\nAssistant: {response}\n")
                except Exception as e:
                    log.exception("Error processing message")
                    print(f"\n❌ Error: {e}\n")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!\n")
        
        finally:
            self.end_session()
    
    def _show_help(self) -> None:
        """Show help information."""
        print("""
Available commands:
  help     - Show this help message
  quit     - Exit the agent (also: exit, q, bye)

Things I can help you with:
  • Connect to SQL databases (PostgreSQL, MySQL, SQLite)
  • Connect to NoSQL databases (MongoDB)
  • Save connection details for future use
  • Sync data from your database to ArchRAG
  • Configure automatic syncing
  • Search your indexed data

Just tell me what you want to do in plain English!
""")


class GuidedSetup:
    """Non-LLM guided setup for users without API access.
    
    This provides a form-based flow for connecting databases
    without requiring LLM calls.
    """
    
    def __init__(
        self,
        orchestrator: "ArchRAGOrchestrator",
        connection_store: ConnectionStore | None = None,
    ):
        self._orch = orchestrator
        self._store = connection_store or ConnectionStore()
        self._tools = AgentTools(orchestrator, self._store)
    
    def run(self) -> None:
        """Run the guided setup flow."""
        print("\n🤖 ArchRAG Guided Setup")
        print("━" * 40)
        print("I'll help you connect your database step by step.\n")
        
        try:
            # Step 1: Check for saved connections
            saved = self._check_saved_connections()
            if saved:
                return
            
            # Step 2: Get connection type
            conn_type = self._get_connection_type()
            if not conn_type:
                return
            
            # Step 3: Get connection details
            config = self._get_connection_details(conn_type)
            if not config:
                return
            
            # Step 4: Connect
            name = self._connect_and_save(conn_type, config)
            if not name:
                return
            
            # Step 5: Select tables and sync
            self._select_and_sync(name)
            
            print("\n✅ Setup complete! Your data has been indexed.")
            print("You can now use 'archrag query' to search your data.\n")
        
        except KeyboardInterrupt:
            print("\n\nSetup cancelled.\n")
    
    def _check_saved_connections(self) -> bool:
        """Check for saved connections and offer to use one."""
        result = self._tools.execute_tool("list_saved_connections", {})
        
        if not result.data:
            return False
        
        print("Found saved connections:")
        for i, conn in enumerate(result.data, 1):
            print(f"  {i}. {conn['name']} ({conn['type']}) - {conn['description'] or 'no description'}")
        
        print(f"  {len(result.data) + 1}. Create new connection")
        
        choice = input("\nSelect a connection (number): ").strip()
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(result.data):
                name = result.data[idx]["name"]
                print(f"\nConnecting to '{name}'...")
                
                connect_result = self._tools.execute_tool("connect_database", {"saved_name": name})
                if connect_result.success:
                    print(f"✅ {connect_result.message}")
                    self._select_and_sync(name)
                    return True
                else:
                    print(f"❌ {connect_result.message}")
        except ValueError:
            pass
        
        return False
    
    def _get_connection_type(self) -> str | None:
        """Get the database type from user."""
        print("\nWhat type of database do you want to connect?")
        print("  1. PostgreSQL")
        print("  2. MySQL")
        print("  3. SQLite")
        print("  4. MongoDB")
        print("  5. Other SQL (via connection string)")
        
        choice = input("\nSelect (1-5): ").strip()
        
        mapping = {
            "1": "sql",
            "2": "sql",
            "3": "sql",
            "4": "nosql",
            "5": "sql",
        }
        
        return mapping.get(choice)
    
    def _get_connection_details(self, conn_type: str) -> dict[str, Any] | None:
        """Get connection details from user."""
        print("\nEnter your connection details:")
        
        if conn_type == "sql":
            print("Example: postgresql://user:password@localhost:5432/database")
            conn_string = input("Connection string: ").strip()
            
            if not conn_string:
                print("❌ Connection string required.")
                return None
            
            return {"connection_string": conn_string}
        
        else:  # nosql
            host = input("Host [localhost]: ").strip() or "localhost"
            port = input("Port [27017]: ").strip() or "27017"
            database = input("Database name: ").strip()
            
            if not database:
                print("❌ Database name required.")
                return None
            
            return {
                "host": host,
                "port": int(port),
                "database": database,
            }
    
    def _connect_and_save(self, conn_type: str, config: dict[str, Any]) -> str | None:
        """Connect to the database and optionally save."""
        print("\nConnecting...")
        
        # Try to connect
        try:
            result = self._orch.connect_database(conn_type, config)
            tables = result.get("tables", [])
            print(f"✅ Connected! Found {len(tables)} tables.")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return None
        
        # Ask to save
        save = input("\nSave this connection for future use? (y/n): ").strip().lower()
        
        if save == "y":
            name = input("Give it a name (e.g., 'people', 'sales'): ").strip()
            description = input("Description (optional): ").strip()
            
            if name:
                save_result = self._tools.execute_tool("save_connection", {
                    "name": name,
                    "connector_type": conn_type,
                    **config,
                    "description": description,
                })
                print(f"✅ Saved as '{name}'")
                return name
        
        return "__temp__"
    
    def _select_and_sync(self, name: str) -> None:
        """Select tables and sync."""
        # List tables
        result = self._tools.execute_tool("list_tables", {})
        if not result.success:
            print(f"❌ {result.message}")
            return
        
        tables = result.data
        print(f"\nAvailable tables ({len(tables)}):")
        for i, table in enumerate(tables, 1):
            print(f"  {i}. {table}")
        
        print(f"\nWhich tables to index?")
        print("Enter numbers separated by commas, or 'all' for all tables:")
        
        choice = input("Tables: ").strip()
        
        if choice.lower() == "all":
            selected_tables = tables
        else:
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(",")]
                selected_tables = [tables[i] for i in indices if 0 <= i < len(tables)]
            except (ValueError, IndexError):
                print("Invalid selection, syncing all tables.")
                selected_tables = tables
        
        print(f"\nSyncing {len(selected_tables)} table(s)...")
        
        sync_result = self._tools.execute_tool("sync_database", {
            "tables": selected_tables,
            "incremental": True,
            "save_preferences": True,
        })
        
        if sync_result.success:
            print(f"✅ {sync_result.message}")
        else:
            print(f"❌ {sync_result.message}")
