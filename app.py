import streamlit as st
import asyncio
import json
import sys
import os
from datetime import datetime
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from perception import extract_perception
from memory import MemoryManager, MemoryItem
from decision import generate_plan
from action import execute_tool

# Page configuration
st.set_page_config(
    page_title="KEDB Resolution System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'mcp_session' not in st.session_state:
    st.session_state.mcp_session = None
if 'tools' not in st.session_state:
    st.session_state.tools = []
if 'memory' not in st.session_state:
    st.session_state.memory = MemoryManager()
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"session-{int(datetime.now().timestamp())}"
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .error-box {
        background-color: #fee;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #d32f2f;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4caf50;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_mcp_session():
    """Initialize MCP session and cache it"""
    try:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["example3.py"],
            cwd=os.getcwd()
        )
        return server_params
    except Exception as e:
        st.error(f"Failed to initialize MCP server parameters: {e}")
        return None

async def get_tools_and_session():
    """Get or create MCP session and tools"""
    if st.session_state.mcp_session is None or not st.session_state.tools:
        server_params = initialize_mcp_session()
        if server_params is None:
            return None, []
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                tools = tools_result.tools
                tool_descriptions = "\n".join(
                    f"- {tool.name}: {getattr(tool, 'description', 'No description')}" 
                    for tool in tools
                )
                # Store tools in session state
                st.session_state.tools = tools
                st.session_state.tool_descriptions = tool_descriptions
                return session, tools
    return st.session_state.mcp_session, st.session_state.tools

async def process_query(user_input: str, max_steps: int = 3):
    """Process user query using the agent system"""
    results = []
    errors = []
    
    try:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["example3.py"],
            cwd=os.getcwd()
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                tools_result = await session.list_tools()
                tools = tools_result.tools
                tool_descriptions = "\n".join(
                    f"- {tool.name}: {getattr(tool, 'description', 'No description')}" 
                    for tool in tools
                )
                
                memory = st.session_state.memory
                session_id = st.session_state.session_id
                query = user_input
                step = 0
                current_input = user_input
                
                results.append({
                    "type": "info",
                    "message": f"Starting agent processing for: {user_input}",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                while step < max_steps:
                    results.append({
                        "type": "info",
                        "message": f"Step {step + 1} of {max_steps}",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Perception
                    perception = extract_perception(current_input)
                    results.append({
                        "type": "info",
                        "message": f"Intent detected: {perception.intent}",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Memory retrieval
                    retrieved = memory.retrieve(query=current_input, top_k=3, session_filter=session_id)
                    
                    # Generate plan
                    plan = generate_plan(perception, retrieved, tool_descriptions=tool_descriptions)
                    results.append({
                        "type": "info",
                        "message": f"Plan: {plan[:200]}...",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    if plan.startswith("FINAL_ANSWER:"):
                        final_answer = plan.replace("FINAL_ANSWER:", "").strip()
                        results.append({
                            "type": "success",
                            "message": final_answer,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        break
                    
                    # Execute tool
                    try:
                        tool_result = await execute_tool(session, tools, plan)
                        results.append({
                            "type": "success",
                            "message": f"Tool '{tool_result.tool_name}' executed successfully",
                            "details": str(tool_result.result)[:500],
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Try to parse KEDB result if it's from kedb_resolve
                        if tool_result.tool_name == "kedb_resolve":
                            try:
                                result_data = tool_result.result
                                # Handle different result formats
                                if isinstance(result_data, list) and len(result_data) > 0:
                                    result_str = result_data[0] if isinstance(result_data[0], str) else json.dumps(result_data[0])
                                    try:
                                        kedb_result = json.loads(result_str) if isinstance(result_str, str) else result_str
                                        if isinstance(kedb_result, dict) and "issue" in kedb_result:
                                            results.append({
                                                "type": "kedb_result",
                                                "data": kedb_result,
                                                "timestamp": datetime.now().strftime("%H:%M:%S")
                                            })
                                    except Exception as parse_error:
                                        # If JSON parsing fails, try to extract dict directly
                                        if isinstance(result_data[0], dict):
                                            results.append({
                                                "type": "kedb_result",
                                                "data": result_data[0],
                                                "timestamp": datetime.now().strftime("%H:%M:%S")
                                            })
                                elif isinstance(result_data, dict) and "issue" in result_data:
                                    results.append({
                                        "type": "kedb_result",
                                        "data": result_data,
                                        "timestamp": datetime.now().strftime("%H:%M:%S")
                                    })
                            except Exception as e:
                                # Log but don't fail
                                errors.append({
                                    "type": "warning",
                                    "message": f"Could not parse KEDB result: {str(e)}",
                                    "timestamp": datetime.now().strftime("%H:%M:%S")
                                })
                        
                        memory.add(MemoryItem(
                            text=f"Tool call: {tool_result.tool_name} with {tool_result.arguments}, got: {tool_result.result}",
                            type="tool_output",
                            tool_name=tool_result.tool_name,
                            user_query=user_input,
                            tags=[tool_result.tool_name],
                            session_id=session_id
                        ))
                        
                        current_input = f"Original task: {query}\nPrevious output: {tool_result.result}\nWhat should I do next?"
                        
                    except Exception as e:
                        errors.append({
                            "type": "error",
                            "message": f"Tool execution failed: {str(e)}",
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        break
                    
                    step += 1
                
    except Exception as e:
        errors.append({
            "type": "error",
            "message": f"Processing error: {str(e)}",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    return results, errors

def render_kedb_result(kedb_data: dict):
    """Render KEDB resolution result in a formatted way"""
    st.markdown("### 📊 KEDB Resolution Report")
    
    # Step 1: Issue & Cause
    if "issue" in kedb_data:
        with st.expander("🔍 Step 1: Issue & Cause", expanded=True):
            st.markdown(f"**Issue:** {kedb_data['issue']}")
            if "similarity" in kedb_data:
                st.caption(f"Similarity Score: {kedb_data.get('similarity', 0):.3f}")
    
    if "cause" in kedb_data:
        st.info(f"**Cause:** {kedb_data['cause']}")
    
    # Step 2: Analysis & SQL Execution
    if "analysis" in kedb_data or ("sql" in kedb_data and kedb_data['sql']):
        with st.expander("⚙️ Step 2: Analysis & SQL Execution", expanded=True):
            if "analysis" in kedb_data and kedb_data['analysis']:
                st.markdown("**Analysis:**")
                st.text(kedb_data['analysis'][:500])
            
            if "sql" in kedb_data and isinstance(kedb_data['sql'], list) and kedb_data['sql']:
                st.markdown("**SQL Executed:**")
                for i, sql in enumerate(kedb_data['sql'], 1):
                    st.code(sql, language="sql")
            
            if "rows" in kedb_data and isinstance(kedb_data['rows'], list):
                total_rows = sum(len(rows) for rows in kedb_data['rows'])
                st.success(f"✅ Executed {len(kedb_data['sql'])} SQL query/queries, returned {total_rows} total row(s)")
                
                for i, rows in enumerate(kedb_data['rows'], 1):
                    if rows:
                        st.markdown(f"**Query {i} Results ({len(rows)} row(s)):**")
                        try:
                            import pandas as pd
                            if rows and isinstance(rows[0], (tuple, list)):
                                df = pd.DataFrame(rows)
                                st.dataframe(df, use_container_width=True, height=200)
                            else:
                                st.json(rows)
                        except Exception as e:
                            # Fallback to JSON if pandas fails
                            if rows:
                                st.text(f"Data: {str(rows)[:500]}")
                    else:
                        st.warning(f"Query {i} returned no rows.")
    
    # Step 3: Outcome
    if "outcome" in kedb_data and kedb_data['outcome']:
        with st.expander("✅ Step 3: Outcome", expanded=True):
            st.markdown(f"**Outcome:** {kedb_data['outcome']}")
    
    # Show markdown report if available
    if "report_md" in kedb_data:
        with st.expander("📄 Full Report (Markdown)", expanded=False):
            st.markdown(kedb_data['report_md'])

# Main UI
st.markdown('<div class="main-header">🔍 KEDB Resolution System</div>', unsafe_allow_html=True)

st.markdown("""
This system uses **RAG (Retrieval-Augmented Generation)** and **Vector Embeddings** to resolve Knowledge-Based Engineering Database (KEDB) queries.
- Uses pure RAG/Vector embeddings (no regex/hardcoding)
- Automatically extracts parameters from queries using LLM
- Executes SQL against PostgreSQL database
- Provides structured resolution reports
""")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    max_steps = st.slider("Max Steps", 1, 5, 3)
    show_details = st.checkbox("Show Detailed Logs", value=True)
    
    st.header("📚 About")
    st.markdown("""
    **KEDB System:**
    - Step 1: Find Issue & Cause (KEDB.txt)
    - Step 2: Find & Execute SQL (KEDB_Analysis.txt)
    - Step 3: Get Outcome (KEDB_Result.txt)
    """)

# Main input form
with st.form("query_form", clear_on_submit=False):
    user_query = st.text_area(
        "Enter your query:",
        placeholder="Example: 'Client is requesting for user details for client_id = 125678945'",
        height=100,
        key="user_query_input"
    )
    
    submitted = st.form_submit_button("🔍 Submit Query", use_container_width=True)

# Process query when submitted
if submitted and user_query:
    with st.spinner("Processing your query..."):
        # Add to conversation history
        st.session_state.conversation_history.append({
            "query": user_query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Run async processing
        results, errors = asyncio.run(process_query(user_query, max_steps))
        
        # Display results
        st.markdown("---")
        st.markdown("### 📋 Results")
        
        kedb_result_shown = False
        
        for result in results:
            if result["type"] == "kedb_result":
                render_kedb_result(result["data"])
                kedb_result_shown = True
            elif result["type"] == "success":
                if not kedb_result_shown:
                    st.markdown(f'<div class="success-box">{result["message"]}</div>', unsafe_allow_html=True)
                    if "details" in result:
                        st.text(result["details"])
            elif result["type"] == "info" and show_details:
                st.caption(f'[{result["timestamp"]}] {result["message"]}')
        
        for error in errors:
            st.markdown(f'<div class="error-box">❌ {error["message"]}</div>', unsafe_allow_html=True)

# Conversation history
if st.session_state.conversation_history:
    with st.expander("📜 Conversation History"):
        for i, entry in enumerate(reversed(st.session_state.conversation_history[-5:]), 1):
            st.markdown(f"**Query {i}** ({entry['timestamp']}):")
            st.text(entry['query'])
            st.markdown("---")

