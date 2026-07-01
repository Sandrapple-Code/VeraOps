import time
import json
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
    
    # Clean tool runs summary for Streamlit UI
    formatted_runs = []
    for r in tool_runs:
        result_str = str(r["result"])
        if len(result_str) > 200:
            result_str = result_str[:200] + "..."
        formatted_runs.append({
            "tool": r["tool"],
            "args": r["args"],
            "result_summary": result_str
        })
        
    return {
        "final_response": final_state.get("final_response", ""),
        "sources": final_state.get("sources_used", []),
        "tool_runs": formatted_runs,
        "execution_time": execution_time
    }
