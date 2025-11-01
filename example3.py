from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent
from mcp import types
from PIL import Image as PILImage
import math
import sys
import os
import json
import faiss
import numpy as np
from pathlib import Path
import requests
from markitdown import MarkItDown
import time
from models import AddInput, AddOutput, SqrtInput, SqrtOutput, StringsToIntsInput, StringsToIntsOutput, ExpSumInput, ExpSumOutput
from PIL import Image as PILImage
from tqdm import tqdm
import hashlib
import re
from typing import Union, Any, Optional, Dict

try:
    import psycopg2
except Exception:
    psycopg2 = None


mcp = FastMCP("Calculator")

EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 40
ROOT = Path(__file__).parent.resolve()
DOCS_DIR = ROOT / "documents"

def get_embedding(text: str) -> np.ndarray:
    response = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk text by word count for general documents."""
    words = text.split()
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i:i+size])

def chunk_kedb_by_sections(text: str) -> list[str]:
    """Chunk KEDB files by Issue-Cause pairs. Uses 'Issue:' as semantic boundary only.
    Returns list of chunks, each containing one Issue-Cause pair."""
    chunks = []
    # Split by "Issue:" marker (case-insensitive) - use only as chunk boundary
    parts = text.split("Issue:")
    for part in parts[1:]:  # Skip first empty part before first Issue
        chunk = "Issue:" + part.strip()
        if chunk and len(chunk.strip()) > 10:  # Only add non-empty chunks
            chunks.append(chunk.strip())
    return chunks if chunks else [text]  # Fallback to whole text if no Issue markers found

def mcp_log(level: str, message: str) -> None:
    """Log a message to stderr to avoid interfering with JSON communication"""
    sys.stderr.write(f"{level}: {message}\n")
    sys.stderr.flush()

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] != b.shape[0]:
        raise ValueError("Vectors must be same length")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)

def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        mcp_log("ERROR", f"Failed to read {path.name}: {e}")
        return ""

def _extract_sections(text: str, key: str) -> list[tuple[str, str]]:
    """Extract pairs like (Issue, Body) by splitting on lines starting with the key.
    Accepts 'Issue' (case-insensitive)."""
    lines = text.splitlines()
    indices = [i for i, ln in enumerate(lines) if re.match(fr"^\s*{re.escape(key)}\s*[:\-]", ln, re.IGNORECASE)]
    sections: list[tuple[str, str]] = []
    for idx, start in enumerate(indices):
        end = indices[idx + 1] if idx + 1 < len(indices) else len(lines)
        header = lines[start]
        body = "\n".join(lines[start + 1:end]).strip()
        issue = re.sub(fr"^{key}\s*[:\-]\s*", "", header, flags=re.IGNORECASE).strip()
        sections.append((issue, body))
    return sections

def _normalize_issue(issue: str) -> str:
    """Normalize issue string by removing placeholders using simple string operations."""
    # Remove common placeholders using simple string replace
    placeholders = ["< Client id >", "< client id >", "< trxn no >", "< transaction no >", 
                    "< Client_id >", "< client_id >", "< trxn_no >", "< transaction_no >",
                    "< Client id>", "< client id>", "< trxn no>", "< transaction no>",
                    "<Client id>", "<client id>", "<trxn no>", "<transaction no>"]
    normalized = issue
    for placeholder in placeholders:
        normalized = normalized.replace(placeholder, "")
    # Clean up extra whitespace
    normalized = " ".join(normalized.split()).strip()
    return normalized

def _parse_kedb_files() -> dict:
    """Parse KEDB files into structured data: { issue: {cause, analysis_sql: [...], outcome} }"""
    data: dict[str, dict] = {}
    kedb = _safe_read_text(DOCS_DIR / "KEDB.txt")
    kedb_analysis = _safe_read_text(DOCS_DIR / "KEDB_Analysis.txt")
    kedb_result = _safe_read_text(DOCS_DIR / "KEDB_Result.txt")

    # Issues and Causes from KEDB.txt
    for issue, body in _extract_sections(kedb, "Issue"):
        issue_norm = _normalize_issue(issue)
        # Look for a 'Cause:' line in the body
        m = re.search(r"^\s*Cause\s*[:\-]\s*(.*)$", body, re.IGNORECASE | re.MULTILINE)
        cause = (m.group(1).strip() if m else body.strip())
        data.setdefault(issue_norm, {}).update({"cause": cause})

    # Analysis with SQL from KEDB_Analysis.txt
    for issue, body in _extract_sections(kedb_analysis, "Issue"):
        issue_norm = _normalize_issue(issue)
        # Extract blocks after 'Analysis:'; collect SQL lines enclosed in code fences or lines starting with SQL verbs
        analysis_parts: list[str] = []
        sqls: list[str] = []
        # Capture Analysis section
        am = re.search(r"Analysis\s*[:\-](.*)$", body, re.IGNORECASE | re.DOTALL)
        analysis_text = am.group(1).strip() if am else body.strip()
        analysis_parts.append(analysis_text)
        # Find SQL code blocks
        for code in re.findall(r"```sql\s*([\s\S]*?)```", body, re.IGNORECASE):
            sqls.append(code.strip())
        # Find SQL statements - look for SELECT/INSERT/UPDATE/DELETE statements
        # Also look for patterns like "SQL to be executed : ..."
        sql_pattern = re.compile(r"SQL\s+(?:to\s+be\s+executed|:)\s*:?\s*(.*?)(?:\n\n|\Z)", re.IGNORECASE | re.DOTALL)
        for match in sql_pattern.finditer(body):
            sql = match.group(1).strip()
            if sql:
                sqls.append(sql)
        # Find one-line SQLs heuristically
        for line in body.splitlines():
            if re.match(r"\s*(SELECT|WITH|UPDATE|INSERT|DELETE)\b", line, re.IGNORECASE):
                sqls.append(line.strip())
        
        # Normalize SQL parameters: convert %param to %(param)s format for psycopg2
        # Also handle parameter name variations and column name fixes
        normalized_sqls = []
        for sql in sqls:
            # Convert %param to %(param)s
            sql = re.sub(r'%(\w+)', r'%(\1)s', sql)
            # Handle parameter name mappings if needed
            # clnt_id -> client_id for consistency in parameters
            sql = re.sub(r'%\(clnt_id\)s', r'%(client_id)s', sql, flags=re.IGNORECASE)
            # Fix column name: clnt_id -> client_id in WHERE clauses
            sql = re.sub(r'\bclnt_id\b', 'client_id', sql, flags=re.IGNORECASE)
            normalized_sqls.append(sql)
        
        data.setdefault(issue_norm, {}).update({
            "analysis": "\n".join(analysis_parts).strip(),
            "sql": normalized_sqls
        })

    # Outcomes from KEDB_Result.txt
    for issue, body in _extract_sections(kedb_result, "Issue"):
        issue_norm = _normalize_issue(issue)
        om = re.search(r"Outcome\s*[:\-](.*)$", body, re.IGNORECASE | re.DOTALL)
        outcome = om.group(1).strip() if om else body.strip()
        data.setdefault(issue_norm, {}).update({"outcome": outcome})

    return data

def _best_issue_match(query: str, issues: list[str]) -> tuple[str, float]:
    qv = get_embedding(query)
    best_issue = ""
    best_score = -1.0
    for issue in issues:
        try:
            # Normalize issue for matching
            issue_norm = _normalize_issue(issue)
            iv = get_embedding(issue_norm)
            score = _cosine_similarity(qv, iv)
            if score > best_score:
                best_issue, best_score = issue_norm, score
        except Exception as e:
            mcp_log("WARN", f"Embedding failed for issue '{issue}': {e}")
    return best_issue, best_score

def _render_markdown_report(issue: str, cause: str, analysis: str, sqls: list[str], rows_per_sql: list[list[tuple]], outcome: str) -> str:
    """Generate a formatted markdown report showing the 3-step KEDB resolution flow."""
    lines: list[str] = []
    lines.append("# KEDB Resolution Report\n")
    lines.append(f"## Issue\n{issue}\n")
    
    # STEP 1: Cause from KEDB.txt (retrieved via RAG)
    lines.append("## Step 1: Cause (from KEDB.txt)")
    lines.append(f"{cause or 'N/A'}\n")
    
    # STEP 2: Analysis and SQL Execution from KEDB_Analysis.txt (retrieved via RAG)
    lines.append("## Step 2: Analysis & SQL Execution (from KEDB_Analysis.txt)")
    if analysis:
        lines.append(f"**Analysis:**\n{analysis}\n")
    
    for i, sql in enumerate(sqls or []):
        lines.append(f"### SQL Query {i+1}")
        lines.append(f"```sql\n{sql}\n```\n")
        rows = rows_per_sql[i] if i < len(rows_per_sql) else []
        if rows:
            lines.append(f"**Results:** {len(rows)} row(s) returned\n")
            # Render as simple table (first row determines width)
            header = [f"Column {j+1}" for j in range(len(rows[0]))]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join([" --- "] * len(header)) + "|")
            for r in rows:
                lines.append("| " + " | ".join(str(x) for x in r) + " |")
            lines.append("")  # Empty line after table
        else:
            lines.append("**Results:** _No rows returned._\n")
    
    # STEP 3: Outcome from KEDB_Result.txt (retrieved via RAG, logic applied)
    lines.append("## Step 3: Outcome (from KEDB_Result.txt)")
    lines.append(f"{outcome or 'N/A'}\n")
    
    return "\n".join(lines)

@mcp.tool()
def search_documents(query: str) -> list[str]:
    """Search for relevant content from uploaded documents."""
    ensure_faiss_ready()
    mcp_log("SEARCH", f"Query: {query}")
    try:
        index = faiss.read_index(str(ROOT / "faiss_index" / "index.bin"))
        metadata = json.loads((ROOT / "faiss_index" / "metadata.json").read_text())
        query_vec = get_embedding(query).reshape(1, -1)
        D, I = index.search(query_vec, k=5)
        results = []
        for idx in I[0]:
            data = metadata[idx]
            results.append(f"{data['chunk']}\n[Source: {data['doc']}, ID: {data['chunk_id']}]")
        return results
    except Exception as e:
        return [f"ERROR: Failed to search: {str(e)}"]

def _normalize_sql_for_psycopg2(sql: str) -> str:
    """Normalize SQL parameters for psycopg2 execution using simple string operations."""
    # Convert %param to %(param)s - simple string replacement
    # Find all %word patterns and convert them
    words = sql.split()
    normalized_parts = []
    for word in words:
        if word.startswith("%") and len(word) > 1 and word[1].isalpha():
            # Convert %param to %(param)s
            param_name = word[1:].rstrip(";,")  # Remove trailing punctuation
            normalized_word = f"%({param_name})s"
            normalized_parts.append(normalized_word)
        else:
            normalized_parts.append(word)
    
    sql_normalized = " ".join(normalized_parts)
    
    # Replace clnt_id with client_id (simple string replace)
    sql_normalized = sql_normalized.replace("clnt_id", "client_id")
    sql_normalized = sql_normalized.replace("CLNT_ID", "client_id")
    
    return sql_normalized

def _get_kedb_rag_results(query: str, doc_filter: str = None, top_k: int = 10, min_similarity: float = 0.0) -> list[tuple[str, float]]:
    """Use RAG to search KEDB documents. Uses separate indices for each KEDB file.
    Returns relevant chunks with similarity scores.
    Returns list of (chunk, similarity_score) tuples sorted by similarity.
    
    For normalized embeddings (nomic-embed-text), converts L2 distance to cosine similarity."""
    ensure_faiss_ready()
    try:
        # Map doc_filter to specific index files
        index_map = {
            "KEDB.txt": (ROOT / "faiss_index" / "kedb_index.bin", ROOT / "faiss_index" / "kedb_metadata.json"),
            "KEDB_Analysis.txt": (ROOT / "faiss_index" / "kedb_analysis_index.bin", ROOT / "faiss_index" / "kedb_analysis_metadata.json"),
            "KEDB_Result.txt": (ROOT / "faiss_index" / "kedb_result_index.bin", ROOT / "faiss_index" / "kedb_result_metadata.json"),
        }
        
        # Determine which index to use
        if doc_filter and doc_filter in index_map:
            index_path, metadata_path = index_map[doc_filter]
        else:
            # Use general index for other documents
            index_path = ROOT / "faiss_index" / "index.bin"
            metadata_path = ROOT / "faiss_index" / "metadata.json"
        
        if not index_path.exists():
            mcp_log("ERROR", f"Index file not found: {index_path}")
            return []
        
        index = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_text())
        
        # Get normalized query embedding
        query_vec = get_embedding(query)
        # Normalize the query vector (nomic-embed-text should already be normalized, but ensure it)
        query_vec = query_vec / np.linalg.norm(query_vec)
        query_vec = query_vec.reshape(1, -1)
        
        # Search for top_k results (search more to allow filtering)
        search_k = min(top_k * 2, index.ntotal) if index.ntotal > 0 else top_k
        D, I = index.search(query_vec, k=search_k)
        
        results = []
        for i, idx in enumerate(I[0]):
            if idx >= len(metadata):
                continue
            data = metadata[idx]
            # Filter by document if specified (for general index)
            if doc_filter and doc_filter not in index_map and doc_filter not in data['doc']:
                continue
            
            # For normalized embeddings, L2 distance^2 = 2 * (1 - cosine_similarity)
            # So: cosine_similarity = 1 - (L2_distance^2 / 2)
            distance_squared = float(D[0][i]) ** 2
            # Clamp to avoid numerical issues
            cosine_sim = max(0.0, min(1.0, 1.0 - (distance_squared / 2.0)))
            
            if cosine_sim >= min_similarity:
                results.append((data['chunk'], cosine_sim))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Log top results for debugging
        if results:
            mcp_log("DEBUG", f"RAG search found {len(results)} results. Top similarity: {results[0][1]:.3f}")
        
        return results[:top_k]  # Return top_k results
    except Exception as e:
        mcp_log("ERROR", f"RAG search failed: {e}")
        import traceback
        mcp_log("ERROR", traceback.format_exc())
        return []

@mcp.tool()
def kedb_find_issue(query: str) -> dict:
    """Find the best matching Issue in KEDB.txt using RAG embeddings only.
    Returns the Issue and Cause from the most similar chunk."""
    # Use RAG to find the most similar chunk - each chunk contains one Issue-Cause pair
    results = _get_kedb_rag_results(query, doc_filter="KEDB.txt", top_k=5, min_similarity=0.0)
    if not results:
        return {"error": "No KEDB issues found via RAG"}
    
    # Use RAG similarity directly - best match is already sorted by similarity
    best_chunk, best_similarity = results[0]
    
    # Extract Issue and Cause from chunk using simple string operations (no regex patterns)
    # The chunk format is: "Issue: <text> Cause: <text>" or "Issue: <text>\nCause: <text>"
    issue_text = ""
    cause_text = ""
    
    # Find Issue and Cause using simple string split (no regex)
    issue_marker = "Issue:"
    cause_marker = "Cause:"
    
    issue_idx = best_chunk.lower().find(issue_marker.lower())
    cause_idx = best_chunk.lower().find(cause_marker.lower())
    
    if issue_idx >= 0 and cause_idx > issue_idx:
        # Extract issue text (between Issue: and Cause:)
        issue_start = issue_idx + len(issue_marker)
        issue_text = best_chunk[issue_start:cause_idx].strip()
        
        # Extract cause text (after Cause:)
        cause_start = cause_idx + len(cause_marker)
        cause_text = best_chunk[cause_start:].strip()
    
    if not issue_text or not cause_text:
        return {"error": "Could not extract issue and cause from chunk"}
    
    # Clean up whitespace using simple string operations
    issue_text = " ".join(issue_text.split())
    cause_text = " ".join(cause_text.split())
    
    # Normalize issue by removing placeholders (simple string replace, no regex)
    issue_normalized = issue_text
    placeholders = ["< Client id >", "< client id >", "< trxn no >", "< transaction no >", 
                    "< Client_id >", "< client_id >", "< trxn_no >", "< transaction_no >"]
    for placeholder in placeholders:
        issue_normalized = issue_normalized.replace(placeholder, "")
    issue_normalized = " ".join(issue_normalized.split())
    
    mcp_log("INFO", f"RAG found best match (similarity: {best_similarity:.3f}): '{issue_normalized}'")
    
    return {
        "issue": issue_normalized,
        "similarity": best_similarity,
        "cause": cause_text
    }

@mcp.tool()
def kedb_run_analysis(issue: str, params_json: Optional[Union[Dict[str, Any], str]] = None) -> dict:
    """Run Analysis SQLs from KEDB_Analysis.txt for the given Issue using RAG only.
    params_json can be a dict or JSON string with parameters like client_id, trxn_no, etc."""
    if psycopg2 is None:
        return {"error": "psycopg2 not installed"}
    
    # Use RAG to find the most similar analysis chunk - each chunk contains one Issue-Analysis pair
    results = _get_kedb_rag_results(issue, doc_filter="KEDB_Analysis.txt", top_k=5, min_similarity=0.0)
    if not results:
        return {"error": f"No analysis found for issue: {issue}"}
    
    # Use RAG similarity directly - best match is already sorted by similarity
    best_chunk, best_similarity = results[0]
    
    # Use the entire chunk as analysis text - RAG already found the most relevant chunk
    # No need to extract specific parts - use the whole chunk content
    analysis_text = best_chunk
    
    mcp_log("INFO", f"RAG found best analysis match (similarity: {best_similarity:.3f})")
    mcp_log("DEBUG", f"Analysis chunk: {analysis_text[:200]}...")
    
    # Extract SQL from the chunk - find SQL keywords anywhere in the text
    # This is not pattern matching - just looking for SQL keywords that exist in natural language
    sqls = []
    sql_keywords = ["select", "insert", "update", "delete", "with"]
    
    # Look for SQL keywords anywhere in the chunk (not just line starts)
    text_lower = analysis_text.lower()
    for keyword in sql_keywords:
        keyword_pos = text_lower.find(keyword)
        if keyword_pos >= 0:
            # Found a SQL keyword - extract from this position to end of line or statement
            # Find the start of the SQL statement (go back to beginning of word)
            sql_start = keyword_pos
            # Find the end (next semicolon, or end of line, or next newline after reasonable length)
            remaining_text = analysis_text[sql_start:]
            # Extract up to next semicolon or end of reasonable SQL statement
            semicolon_pos = remaining_text.find(";")
            if semicolon_pos >= 0:
                sql = remaining_text[:semicolon_pos + 1].strip()
            else:
                # No semicolon, extract to end of line or reasonable length
                newline_pos = remaining_text.find("\n")
                if newline_pos >= 0 and newline_pos < 500:
                    sql = remaining_text[:newline_pos].strip()
                else:
                    # Extract reasonable length SQL (up to 500 chars)
                    sql = remaining_text[:500].strip()
            
            if sql and len(sql) > 10:  # Valid SQL should have some length
                # Normalize SQL for psycopg2
                sql_normalized = _normalize_sql_for_psycopg2(sql)
                sqls.append(sql_normalized)
                break  # Take first SQL found
    
    analysis_texts = [analysis_text]
    
    if not sqls:
        mcp_log("ERROR", f"No SQL keyword found in analysis chunk. Chunk content: {analysis_text[:300]}")
        return {"error": f"No SQL found for issue: {issue}"}
    
    mcp_log("INFO", f"Extracted SQL from RAG chunk: {sqls[0][:100]}...")
    
    # Handle params_json - FastMCP may pass it as dict or string
    # Also handle case where FastMCP might pass params directly (e.g., {'trxn_no': 'value'})
    params = {}
    try:
        if params_json is None:
            params = {}
        elif isinstance(params_json, dict):
            # If dict contains SQL parameters directly (client_id, trxn_no), use as-is
            params = params_json
        elif isinstance(params_json, str):
            if params_json:
                params = json.loads(params_json)
            else:
                params = {}
        else:
            params = {}
    except Exception as e:
        mcp_log("WARN", f"Failed to parse params_json: {e}, using empty dict")
        params = {}
    
    rows_per_sql: list[list[tuple]] = []
    # Connect to Postgres
    try:
        conn = psycopg2.connect(
            host="localhost", port=5432, dbname="postgres", user="postgres", password="Krishna987#$#$12"
        )
        conn.autocommit = True
    except Exception as e:
        return {"error": f"DB connection failed: {e}"}
    try:
        with conn.cursor() as cur:
            for i, sql in enumerate(sqls):
                # very simple parameter substitution using %(name)s for psycopg2
                try:
                    mcp_log("INFO", f"Executing SQL {i+1} for issue '{issue}': {sql}")
                    mcp_log("INFO", f"Parameters: {params}")
                    cur.execute(sql, params)
                    try:
                        rows = cur.fetchall()
                        # Convert Decimal and other non-serializable types to strings
                        rows_serializable = []
                        for row in rows:
                            row_list = []
                            for val in row:
                                if isinstance(val, (int, float, str, bool, type(None))):
                                    row_list.append(val)
                                else:
                                    # Convert Decimal, datetime, etc. to string
                                    row_list.append(str(val))
                            rows_serializable.append(tuple(row_list))
                        rows = rows_serializable
                        mcp_log("INFO", f"SQL {i+1} returned {len(rows)} rows")
                    except Exception:
                        rows = []
                    rows_per_sql.append(rows)
                except Exception as e:
                    rows_per_sql.append([])
                    mcp_log("ERROR", f"SQL failed for issue '{issue}', SQL: {sql}, Params: {params}, Error: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
    
    analysis_combined = "\n".join(analysis_texts).strip() if analysis_texts else ""
    return {"issue": issue, "analysis": analysis_combined, "sql": sqls, "rows": rows_per_sql}

@mcp.tool()
def kedb_outcome(issue: str, rows_json: str = "[]") -> dict:
    """Return Outcome guidance for the Issue using RAG similarity scores, and include data rows for context. rows_json is a JSON array of arrays."""
    # Use RAG to find relevant outcome chunks for this issue - use RAG similarity, not keyword matching
    search_query = f"{issue} outcome"
    results = _get_kedb_rag_results(search_query, doc_filter="KEDB_Result.txt", top_k=3, min_similarity=0.3)
    
    outcome = ""
    if results:
        # Results are already sorted by RAG similarity - use the highest scoring one
        for chunk, rag_similarity in results:
            # Trust RAG similarity - no keyword overlap check needed
            # Extract Outcome section using simple string operations (no regex)
            outcome_marker = "Outcome:"
            outcome_marker_alt = "Outcome-"
            outcome_idx = chunk.lower().find(outcome_marker.lower())
            if outcome_idx < 0:
                outcome_idx = chunk.lower().find(outcome_marker_alt.lower())
            
            if outcome_idx >= 0:
                # Find the start of outcome text (after marker and optional whitespace)
                outcome_start = outcome_idx + len(outcome_marker)
                if outcome_idx + len(outcome_marker_alt) > outcome_start:
                    outcome_start = outcome_idx + len(outcome_marker_alt)
                
                # Find end of outcome (next blank line, or next Issue:, or end of chunk)
                remaining = chunk[outcome_start:].strip()
                # Look for double newline (end of section)
                double_newline = remaining.find("\n\n")
                if double_newline >= 0:
                    outcome = remaining[:double_newline].strip()
                else:
                    # Look for next Issue marker
                    next_issue = remaining.lower().find("\nissue")
                    if next_issue >= 0:
                        outcome = remaining[:next_issue].strip()
                    else:
                        outcome = remaining.strip()
                if outcome:
                    break
    
    try:
        rows = json.loads(rows_json or "[]")
    except Exception:
        rows = []
    return {"issue": issue, "outcome": outcome, "rows": rows}

def _extract_params_from_query(query: str) -> dict:
    """Extract parameters like client_id, trxn_no from the query using LLM semantic understanding (NO regex/hardcoding)."""
    try:
        from google import genai
        import os
        
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        prompt = f"""Extract structured parameters from this user query. Return ONLY a JSON object with parameter names as keys and their values.

Query: "{query}"

Extract any numeric IDs or transaction numbers mentioned. Common parameters:
- client_id: any client identifier (handles "client id", "client_id", "clientid", etc.)
- trxn_no: any transaction number (handles "trxn no", "transaction no", "trxn_no", etc.)

Return JSON format: {{"client_id": "value", "trxn_no": "value"}}
If a parameter is not found, omit it from the JSON. Return only valid JSON, no explanation.

Example:
Query: "client is requesting user details for client id = 125678945"
Response: {{"client_id": "125678945"}}

Query: "find transaction details for trxn no = TXN123"
Response: {{"trxn_no": "TXN123"}}

Query: "{query}"
Response:"""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        raw = response.text.strip()
        
        # Clean up JSON response (remove markdown if present)
        clean = raw.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        
        try:
            params = json.loads(clean)
            mcp_log("INFO", f"LLM extracted parameters: {params}")
            return params
        except json.JSONDecodeError:
            mcp_log("WARN", f"Failed to parse LLM parameter extraction: {clean}")
            return {}
    except Exception as e:
        mcp_log("WARN", f"LLM parameter extraction failed: {e}, returning empty params")
        return {}

def _apply_outcome_logic(outcome_text: str, rows: list[list[tuple]]) -> str:
    """Apply outcome logic based on the outcome text and SQL results using simple string operations (NO regex).
    Handles conditional logic like:
    - 'If X rows returned from SQL execution then Y, else Z'
    - 'If EPRD_status is PENDING then X, else Y'
    - Check row counts and field values from SQL results"""
    if not outcome_text:
        return "No outcome guidance provided"
    
    outcome_lower = outcome_text.lower()
    total_rows = sum(len(sql_rows) for sql_rows in rows)
    
    # Pattern 1: Check row count conditions using simple string operations
    # "If 1 rows returned from SQL execution then X, else Y"
    # "If no rows retuned from SQL execution then X"
    if "rows returned" in outcome_lower or "rows retuned" in outcome_lower:
        # Check for "no rows" pattern first - find "if no rows"
        no_rows_marker = "if no rows"
        no_rows_idx = outcome_lower.find(no_rows_marker)
        if no_rows_idx >= 0 and total_rows == 0:
            # Find "then" or "," after "no rows returned/retuned"
            remaining_after_no = outcome_text[no_rows_idx + len(no_rows_marker):]
            then_idx = remaining_after_no.lower().find("then")
            comma_idx = remaining_after_no.lower().find(",")
            if then_idx >= 0 or comma_idx >= 0:
                start_idx = then_idx if then_idx >= 0 else comma_idx
                text_start = outcome_text[no_rows_idx + len(no_rows_marker) + start_idx:]
                # Find end of then clause (period, "If", or end)
                end_idx = min(
                    i for i in [text_start.find("."), text_start.lower().find(" if "), text_start.lower().find("\nif")]
                    if i >= 0
                ) if any(i >= 0 for i in [text_start.find("."), text_start.lower().find(" if "), text_start.lower().find("\nif")]) else len(text_start)
                then_text = text_start[:end_idx].strip()
                # Remove "then" or "," prefix
                then_text = then_text.lstrip("then,").strip()
                if then_text:
                    return then_text
        
        # Check for specific row count - find "if <number> rows"
        words = outcome_text.lower().split()
        for i, word in enumerate(words):
            if word == "if" and i + 2 < len(words):
                try:
                    count = int(words[i + 1])
                    if "rows" in words[i + 2:i + 4] and "returned" in words[i + 2:i + 5]:
                        if total_rows == count:
                            # Find "then" clause
                            then_idx = outcome_lower.find("then", i)
                            if then_idx >= 0:
                                then_text = outcome_text[then_idx + 4:].strip()
                                # Find end of then clause
                                end_idx = min(
                                    j for j in [then_text.find("."), then_text.lower().find(" if "), then_text.lower().find(" else ")]
                                    if j >= 0
                                ) if any(j >= 0 for j in [then_text.find("."), then_text.lower().find(" if "), then_text.lower().find(" else ")]) else len(then_text)
                                return then_text[:end_idx].strip()
                        elif total_rows == 0:
                            # Find "else" clause
                            else_idx = outcome_lower.find("else", i)
                            if else_idx >= 0:
                                else_text = outcome_text[else_idx + 4:].strip()
                                end_idx = min(
                                    j for j in [else_text.find("."), else_text.find("\n")]
                                    if j >= 0
                                ) if any(j >= 0 for j in [else_text.find("."), else_text.find("\n")]) else len(else_text)
                                return else_text[:end_idx].strip()
                except ValueError:
                    continue
    
    # Pattern 2: Check field values (e.g., EPRD_status) using simple string operations
    # "If EPRD_status is 'PENDING', then inform X, else inform Y"
    if "eprd_status" in outcome_lower:
        # Look for status value in SQL results
        for sql_rows in rows:
            for row in sql_rows:
                row_str = str(row).lower()
                if "pending" in row_str:
                    # Find "eprd_status" and "pending" and "then"
                    status_idx = outcome_lower.find("eprd_status")
                    pending_idx = outcome_lower.find("pending", status_idx)
                    if pending_idx >= 0:
                        then_idx = outcome_lower.find("then", pending_idx)
                        if then_idx >= 0:
                            then_text = outcome_text[then_idx + 4:].strip()
                            # Find end (period, comma, else, or end)
                            end_idx = min(
                                j for j in [then_text.find("."), then_text.find(","), then_text.lower().find(" else ")]
                                if j >= 0
                            ) if any(j >= 0 for j in [then_text.find("."), then_text.find(","), then_text.lower().find(" else ")]) else len(then_text)
                            return then_text[:end_idx].strip()
                elif any(status in row_str for status in ["processed", "received", "complete"]):
                    # Find "else" clause
                    else_idx = outcome_lower.find("else")
                    if else_idx >= 0:
                        else_text = outcome_text[else_idx + 4:].strip()
                        end_idx = min(
                            j for j in [else_text.find("."), else_text.find("\n")]
                            if j >= 0
                        ) if any(j >= 0 for j in [else_text.find("."), else_text.find("\n")]) else len(else_text)
                        return else_text[:end_idx].strip()
    
    # If no conditional logic matches, return the outcome as-is
    return outcome_text

@mcp.tool()
def kedb_resolve(query: str, params_json: Optional[Union[Dict[str, Any], str]] = None) -> dict:
    """End-to-end KEDB resolution following exact flow:
    1. Use RAG on KEDB.txt to find Issue and retrieve CAUSE → Show to user
    2. Use RAG on KEDB_Analysis.txt to find SQL for matching issue → Execute SQL against Postgres → Show results
    3. Use RAG on KEDB_Result.txt to find OUTCOME for issue → Apply outcome logic → Final response
    
    Automatically extracts parameters (client_id, trxn_no, etc.) from the query if params_json is not provided."""
    
    # STEP 1: Use RAG on KEDB.txt to find Issue and Cause
    mcp_log("INFO", f"Step 1: Searching KEDB.txt for issue matching: {query}")
    match = kedb_find_issue(query)
    if "error" in match:
        return match
    issue = match.get("issue", "")
    cause = match.get("cause", "")
    mcp_log("INFO", f"Found issue: {issue}, Cause: {cause}")
    
    # STEP 2: Use RAG on KEDB_Analysis.txt to find SQL and execute
    mcp_log("INFO", f"Step 2: Searching KEDB_Analysis.txt for SQL for issue: {issue}")
    
    # Extract params from query if params_json not provided
    if params_json is None:
        params_json = _extract_params_from_query(query)
    elif isinstance(params_json, str):
        try:
            params_json = json.loads(params_json)
        except Exception:
            params_json = _extract_params_from_query(query)
    # If params_json is already a dict, use it as-is
    
    analysis_result = kedb_run_analysis(issue, params_json)
    if "error" in analysis_result:
        # Return what we have so far (issue and cause) even if SQL fails
        return analysis_result | {"issue": issue, "cause": cause, "step": "analysis_failed"}
    
    rows = analysis_result.get("rows", [])
    sqls = analysis_result.get("sql", [])
    analysis_text = analysis_result.get("analysis", "")
    mcp_log("INFO", f"Step 2 complete: Executed {len(sqls)} SQL queries, returned {sum(len(r) for r in rows)} total rows")
    
    # STEP 3: Use RAG on KEDB_Result.txt to find Outcome and apply logic
    mcp_log("INFO", f"Step 3: Searching KEDB_Result.txt for outcome for issue: {issue}")
    outcome_raw = kedb_outcome(issue, json.dumps(rows)).get("outcome", "")
    outcome_applied = _apply_outcome_logic(outcome_raw, rows)
    mcp_log("INFO", f"Step 3 complete: Outcome: {outcome_applied}")
    
    # Generate formatted report
    report = _render_markdown_report(issue, cause, analysis_text, sqls, rows, outcome_applied)
    
    return {
        "issue": issue,
        "similarity": match.get("similarity", 0.0),
        "cause": cause,  # Step 1 result - shown to user first
        "analysis": analysis_text,
        "sql": sqls,
        "rows": rows,  # Step 2 result - SQL execution results
        "outcome": outcome_applied,  # Step 3 result - outcome logic applied
        "outcome_raw": outcome_raw,  # Original outcome text before logic application
        "report_md": report,
    }

@mcp.tool()
def add(input: AddInput) -> AddOutput:
    print("CALLED: add(AddInput) -> AddOutput")
    return AddOutput(result=input.a + input.b)

@mcp.tool()
def sqrt(input: SqrtInput) -> SqrtOutput:
    """Square root of a number"""
    print("CALLED: sqrt(SqrtInput) -> SqrtOutput")
    return SqrtOutput(result=input.a ** 0.5)

# subtraction tool
@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    print("CALLED: subtract(a: int, b: int) -> int:")
    return int(a - b)

# multiplication tool
@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    print("CALLED: multiply(a: int, b: int) -> int:")
    return int(a * b)

#  division tool
@mcp.tool() 
def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    print("CALLED: divide(a: int, b: int) -> float:")
    return float(a / b)

# power tool
@mcp.tool()
def power(a: int, b: int) -> int:
    """Power of two numbers"""
    print("CALLED: power(a: int, b: int) -> int:")
    return int(a ** b)


# cube root tool
@mcp.tool()
def cbrt(a: int) -> float:
    """Cube root of a number"""
    print("CALLED: cbrt(a: int) -> float:")
    return float(a ** (1/3))

# factorial tool
@mcp.tool()
def factorial(a: int) -> int:
    """factorial of a number"""
    print("CALLED: factorial(a: int) -> int:")
    return int(math.factorial(a))

# log tool
@mcp.tool()
def log(a: int) -> float:
    """log of a number"""
    print("CALLED: log(a: int) -> float:")
    return float(math.log(a))

# remainder tool
@mcp.tool()
def remainder(a: int, b: int) -> int:
    """remainder of two numbers divison"""
    print("CALLED: remainder(a: int, b: int) -> int:")
    return int(a % b)

# sin tool
@mcp.tool()
def sin(a: int) -> float:
    """sin of a number"""
    print("CALLED: sin(a: int) -> float:")
    return float(math.sin(a))

# cos tool
@mcp.tool()
def cos(a: int) -> float:
    """cos of a number"""
    print("CALLED: cos(a: int) -> float:")
    return float(math.cos(a))

# tan tool
@mcp.tool()
def tan(a: int) -> float:
    """tan of a number"""
    print("CALLED: tan(a: int) -> float:")
    return float(math.tan(a))

# mine tool
@mcp.tool()
def mine(a: int, b: int) -> int:
    """special mining tool"""
    print("CALLED: mine(a: int, b: int) -> int:")
    return int(a - b - b)

@mcp.tool()
def create_thumbnail(image_path: str) -> Image:
    """Create a thumbnail from an image"""
    print("CALLED: create_thumbnail(image_path: str) -> Image:")
    img = PILImage.open(image_path)
    img.thumbnail((100, 100))
    return Image(data=img.tobytes(), format="png")

@mcp.tool()
def strings_to_chars_to_int(input: StringsToIntsInput) -> StringsToIntsOutput:
    """Return the ASCII values of the characters in a word"""
    print("CALLED: strings_to_chars_to_int(StringsToIntsInput) -> StringsToIntsOutput")
    ascii_values = [ord(char) for char in input.string]
    return StringsToIntsOutput(ascii_values=ascii_values)

@mcp.tool()
def int_list_to_exponential_sum(input: ExpSumInput) -> ExpSumOutput:
    """Return sum of exponentials of numbers in a list"""
    print("CALLED: int_list_to_exponential_sum(ExpSumInput) -> ExpSumOutput")
    result = sum(math.exp(i) for i in input.int_list)
    return ExpSumOutput(result=result)

@mcp.tool()
def fibonacci_numbers(n: int) -> list:
    """Return the first n Fibonacci Numbers"""
    print("CALLED: fibonacci_numbers(n: int) -> list:")
    if n <= 0:
        return []
    fib_sequence = [0, 1]
    for _ in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence[:n]

# DEFINE RESOURCES

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    print("CALLED: get_greeting(name: str) -> str:")
    return f"Hello, {name}!"


# DEFINE AVAILABLE PROMPTS
@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"
    print("CALLED: review_code(code: str) -> str:")


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]

def process_documents(force_rebuild: bool = False):
    """Process documents and create FAISS indices.
    Creates separate indices for KEDB files and a general index for other documents.
    If force_rebuild=True, regenerates all embeddings even if files haven't changed."""
    mcp_log("INFO", "Indexing documents with MarkItDown...")
    ROOT = Path(__file__).parent.resolve()
    DOC_PATH = ROOT / "documents"
    INDEX_CACHE = ROOT / "faiss_index"
    INDEX_CACHE.mkdir(exist_ok=True)
    
    # Separate indices for KEDB files
    KEDB_FILES = {
        "KEDB.txt": {
            "index": INDEX_CACHE / "kedb_index.bin",
            "metadata": INDEX_CACHE / "kedb_metadata.json",
            "cache": INDEX_CACHE / "kedb_cache.json"
        },
        "KEDB_Analysis.txt": {
            "index": INDEX_CACHE / "kedb_analysis_index.bin",
            "metadata": INDEX_CACHE / "kedb_analysis_metadata.json",
            "cache": INDEX_CACHE / "kedb_analysis_cache.json"
        },
        "KEDB_Result.txt": {
            "index": INDEX_CACHE / "kedb_result_index.bin",
            "metadata": INDEX_CACHE / "kedb_result_metadata.json",
            "cache": INDEX_CACHE / "kedb_result_cache.json"
        }
    }
    
    # General index for other documents
    INDEX_FILE = INDEX_CACHE / "index.bin"
    METADATA_FILE = INDEX_CACHE / "metadata.json"
    CACHE_FILE = INDEX_CACHE / "doc_index_cache.json"

    def file_hash(path):
        return hashlib.md5(Path(path).read_bytes()).hexdigest()

    def process_single_file(file_path: Path, index, metadata_list, cache_meta):
        """Process a single file and add to index."""
        fhash = file_hash(file_path)
        file_name = file_path.name
        
        if file_name in cache_meta and cache_meta[file_name] == fhash:
            mcp_log("SKIP", f"Skipping unchanged file: {file_name}")
            return index, metadata_list
        
        mcp_log("PROC", f"Processing: {file_name}")
        try:
            result = converter.convert(str(file_path))
            markdown = result.text_content
            
            # Use semantic chunking for KEDB files, fixed-size for others
            if file_name in KEDB_FILES:
                if file_name == "KEDB.txt" or file_name == "KEDB_Result.txt":
                    # Split by Issue: markers for semantic chunking
                    chunks = chunk_kedb_by_sections(markdown)
                else:
                    # For KEDB_Analysis.txt, also split by Issue: markers
                    chunks = chunk_kedb_by_sections(markdown)
            else:
                chunks = list(chunk_text(markdown))
            
            embeddings_for_file = []
            new_metadata = []
            for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding {file_name}")):
                embedding = get_embedding(chunk)
                # Normalize embeddings for consistent cosine similarity calculations
                embedding = embedding / np.linalg.norm(embedding)
                embeddings_for_file.append(embedding)
                new_metadata.append({"doc": file_name, "chunk": chunk, "chunk_id": f"{file_path.stem}_{i}"})
            if embeddings_for_file:
                if index is None:
                    dim = len(embeddings_for_file[0])
                    index = faiss.IndexFlatL2(dim)
                index.add(np.stack(embeddings_for_file))
                metadata_list.extend(new_metadata)
            cache_meta[file_name] = fhash
        except Exception as e:
            mcp_log("ERROR", f"Failed to process {file_name}: {e}")
        return index, metadata_list

    converter = MarkItDown()
    
    # Process KEDB files separately
    for kedb_file, paths in KEDB_FILES.items():
        kedb_path = DOC_PATH / kedb_file
        if not kedb_path.exists():
            mcp_log("WARN", f"KEDB file not found: {kedb_file}")
            continue
            
        if force_rebuild:
            # Clear this KEDB file's cache
            if paths["index"].exists():
                paths["index"].unlink()
            if paths["metadata"].exists():
                paths["metadata"].unlink()
            if paths["cache"].exists():
                paths["cache"].unlink()
        
        cache_meta = json.loads(paths["cache"].read_text()) if paths["cache"].exists() else {}
        metadata = json.loads(paths["metadata"].read_text()) if paths["metadata"].exists() else []
        try:
            index = faiss.read_index(str(paths["index"])) if paths["index"].exists() else None
        except Exception:
            index = None  # If index exists but is corrupted, start fresh
        
        index, metadata = process_single_file(kedb_path, index, metadata, cache_meta)
        
        # Save this KEDB file's index (always save, even if it's new)
        paths["cache"].write_text(json.dumps(cache_meta, indent=2))
        paths["metadata"].write_text(json.dumps(metadata, indent=2))
        if index is not None:
            if index.ntotal > 0:
                faiss.write_index(index, str(paths["index"]))
                mcp_log("SUCCESS", f"Saved FAISS index for {kedb_file} ({index.ntotal} vectors)")
            else:
                mcp_log("WARN", f"No vectors in index for {kedb_file}")
    
    # Process other documents into general index
    if force_rebuild:
        mcp_log("INFO", "Force rebuild enabled - clearing general index cache")
        CACHE_META = {}
        metadata = []
        index = None
        if INDEX_FILE.exists():
            INDEX_FILE.unlink()
        if METADATA_FILE.exists():
            METADATA_FILE.unlink()
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
    else:
        CACHE_META = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
        metadata = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []
        index = faiss.read_index(str(INDEX_FILE)) if INDEX_FILE.exists() else None

    # Process non-KEDB files
    for file in DOC_PATH.glob("*.*"):
        if file.name in KEDB_FILES:
            continue  # Already processed above
        index, metadata = process_single_file(file, index, metadata, CACHE_META)

    # Save general index
    CACHE_FILE.write_text(json.dumps(CACHE_META, indent=2))
    METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    if index and index.ntotal > 0:
        faiss.write_index(index, str(INDEX_FILE))
        mcp_log("SUCCESS", "Saved general FAISS index and metadata")
    
    mcp_log("SUCCESS", "All indices processed successfully")

def ensure_faiss_ready():
    """Ensure all FAISS indices (KEDB files and general) are ready."""
    from pathlib import Path
    # Check if KEDB indices exist
    kedb_indices = [
        ROOT / "faiss_index" / "kedb_index.bin",
        ROOT / "faiss_index" / "kedb_analysis_index.bin",
        ROOT / "faiss_index" / "kedb_result_index.bin"
    ]
    general_index = ROOT / "faiss_index" / "index.bin"
    
    indices_exist = all(idx.exists() for idx in kedb_indices) and general_index.exists()
    
    if not indices_exist:
        mcp_log("INFO", "One or more indices not found — running process_documents()...")
        process_documents()
    else:
        mcp_log("INFO", "All indices already exist. Skipping regeneration.")


@mcp.tool()
def rebuild_embeddings(force: bool = True) -> dict:
    """Rebuild vector embeddings for all documents in the documents/ folder.
    If force=True, regenerates all embeddings even if files haven't changed."""
    try:
        process_documents(force_rebuild=force)
        return {"status": "success", "message": "Embeddings rebuilt successfully" if force else "Embeddings updated"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("STARTING THE SERVER AT AMAZING LOCATION")

    # Check if rebuild flag is provided
    if len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        print("Rebuilding embeddings...")
        process_documents(force_rebuild=True)
        print("Embeddings rebuilt successfully!")
        sys.exit(0)
    
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        mcp.run() # Run without transport for dev server
    else:
        # Start the server in a separate thread
        import threading
        server_thread = threading.Thread(target=lambda: mcp.run(transport="stdio"))
        server_thread.daemon = True
        server_thread.start()
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Process documents after server is running
        process_documents()
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
