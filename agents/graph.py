import time
import json
import os
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.router import supervisor_router
from agents.nodes import tool_execution_node, context_builder_node, llm_node

def define_graph():
    """
    Defines the nodes, edges, and transitions for the LangGraph agentic workflow.
    """
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("router", supervisor_router)
    workflow.add_node("tools", tool_execution_node)
    workflow.add_node("context", context_builder_node)
    workflow.add_node("llm", llm_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Conditional transitions
    def decide_next_step(state: AgentState):
        tool_calls = state.get("current_tool_execution", [])
        if tool_calls:
            return "tools"
        else:
            return "context"
            
    workflow.add_conditional_edges(
        "router",
        decide_next_step,
        {
            "tools": "tools",
            "context": "context"
        }
    )
    
    # Static transitions
    workflow.add_edge("tools", "router")  # Loop back after running tools
    workflow.add_edge("context", "llm")
    workflow.add_edge("llm", END)
    
    return workflow.compile()

# Compile the graph
hospital_agent_graph = define_graph()

def run_agent(query: str, history: List[Dict[str, str]], api_key: str) -> Dict[str, Any]:
    """
    Executes the LangGraph agent loop with observability printing.
    """
    start_time = time.time()
    
    print("\n" + "="*60)
    print(f"[AGENT START] Doctor Query: {query}")
    print("="*60)
    
    initial_state: AgentState = {
        "query": query,
        "history": history,
        "retrieved_hospital_context": [],
        "retrieved_patient_context": [],
        "similar_patient_results": [],
        "current_patient_info": {},
        "tool_outputs": [],
        "final_context": "",
        "final_response": "",
        "sources_used": [],
        "current_tool_execution": [],
        "api_key": api_key
    }
    
    try:
        final_state = hospital_agent_graph.invoke(initial_state)
    except Exception as e:
        print(f"[AGENT EXCEPTION] {e}")
        return {
            "final_response": f"Failed to execute clinical agent graph: {str(e)}",
            "sources": [],
            "tool_runs": [],
            "execution_time": 0.0
        }
        
    execution_time = time.time() - start_time
    tool_runs = final_state.get("tool_outputs", [])
    
    # Observability Logging
    print(f"\n[AGENT OBSERVABILITY SUMMARY]")
    print(f"- User Query: '{query}'")
    print(f"- Tools Selected: {', '.join([r['tool'] for r in tool_runs]) if tool_runs else 'None'}")
    print(f"- Execution Order:")
    for idx, r in enumerate(tool_runs):
        print(f"  {idx+1}. {r['tool']} with args: {json.dumps(r['args'])}")
    print(f"- Final Context Size: {len(final_state.get('final_context', ''))} characters")
    print(f"- Sources Cited: {final_state.get('sources_used', [])}")
    print(f"- Execution Time: {execution_time:.2f} seconds")
    print("="*60 + "\n")
    
    # Clean tool runs summary for Streamlit UI and gather explainability details
    formatted_runs = []
    tools_used = []
    hospital_docs = []
    patient_docs = []
    chunks_count = 0
    doc_details = []
    
    for r in tool_runs:
        t_name = r["tool"]
        tools_used.append(t_name)
        result_str = str(r["result"])
        
        # Parse outputs for RAG tools to extract explainability metrics
        if t_name == "hospital_knowledge_search":
            try:
                items = json.loads(result_str)
                if isinstance(items, list):
                    chunks_count += len(items)
                    for item in items:
                        src = item.get("source", "Unknown")
                        src_name = os.path.basename(src)
                        page = item.get("page")
                        page_val = f"Page {page}" if page else "N/A"
                        dist = item.get("distance", 0.0)
                        
                        hospital_docs.append(src_name)
                        doc_details.append({
                            "type": "Hospital SOP",
                            "name": src_name,
                            "page": page_val,
                            "score": f"{dist:.4f} (L2 dist)"
                        })
            except Exception:
                pass
        elif t_name == "patient_history_search":
            try:
                items = json.loads(result_str)
                if isinstance(items, list):
                    chunks_count += len(items)
                    for item in items:
                        pid = item.get("patient_id", "Unknown")
                        src_file = item.get("source", "Unknown")
                        dist = item.get("distance", 0.0)
                        
                        patient_docs.append(f"{pid}/{src_file}")
                        doc_details.append({
                            "type": f"Patient Record ({pid})",
                            "name": src_file,
                            "page": "N/A",
                            "score": f"{dist:.4f} (L2 dist)"
                        })
            except Exception:
                pass
        elif t_name == "similar_case_search":
            try:
                items = json.loads(result_str)
                if isinstance(items, list):
                    for item in items:
                        pid = item.get("patient_id", "Unknown")
                        dist = item.get("distance", 0.0)
                        patient_docs.append(f"Similar Case: {pid}")
                        doc_details.append({
                            "type": "Similar Case",
                            "name": f"Patient {pid} profile",
                            "page": "N/A",
                            "score": f"{dist:.4f} (L2 dist)"
                        })
            except Exception:
                pass
                
        if len(result_str) > 200:
            result_str = result_str[:200] + "..."
        formatted_runs.append({
            "tool": t_name,
            "args": r["args"],
            "result_summary": result_str
        })
        
    explainability_metrics = {
        "tools_used": sorted(list(set(tools_used))),
        "hospital_docs": sorted(list(set(hospital_docs))),
        "patient_docs": sorted(list(set(patient_docs))),
        "retrieved_chunks_count": chunks_count,
        "doc_details": doc_details,
        "memory_used": "Yes" if len(history) > 0 else "No"
    }
        
    return {
        "final_response": final_state.get("final_response", ""),
        "sources": final_state.get("sources_used", []),
        "tool_runs": formatted_runs,
        "execution_time": execution_time,
        "explainability": explainability_metrics
    }
