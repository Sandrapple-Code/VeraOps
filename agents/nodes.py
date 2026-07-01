import os
import json
from typing import Dict, Any, List
from agents.state import AgentState
from llm.groq_client import get_groq_client

# Import existing LangChain tools
from tools.patient_lookup import patient_lookup
from tools.hospital_knowledge import hospital_knowledge_search
from tools.patient_history import patient_history_search
from tools.similar_case import similar_case_search
from tools.bed_availability import bed_availability_lookup
from tools.register_patient import register_patient
from tools.update_patient import modify_patient_record

# Map tool names to tool objects
TOOLS_MAP = {
    "patient_lookup": patient_lookup,
    "hospital_knowledge_search": hospital_knowledge_search,
    "patient_history_search": patient_history_search,
    "similar_case_search": similar_case_search,
    "bed_availability_lookup": bed_availability_lookup,
    "register_patient": register_patient,
    "modify_patient_record": modify_patient_record
}

def tool_execution_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes all scheduled tools in state['current_tool_execution'] sequentially,
    collects their outputs, and tracks source materials.
    """
    tool_calls = state.get("current_tool_execution", [])
    outputs = []
    sources = list(state.get("sources_used", []))
    
    for call in tool_calls:
        tool_name = call.get("tool")
        args = call.get("args", {})
        
        if tool_name in TOOLS_MAP:
            tool_obj = TOOLS_MAP[tool_name]
            try:
                # Execute the LangChain tool
                result = tool_obj.invoke(args)
                outputs.append({
                    "tool": tool_name,
                    "args": args,
                    "result": result
                })
                
                # Dynamic source logging based on tool type
                if tool_name == "hospital_knowledge_search":
                    try:
                        items = json.loads(result)
                        for item in items:
                            if "source" in item:
                                sources.append(f"SOP Doc: {os.path.basename(item['source'])}")
                    except Exception:
                        pass
                elif tool_name in ("patient_lookup", "patient_history_search", "similar_case_search"):
                    patient_id = args.get("patient_id") or args.get("identifier")
                    if patient_id:
                        sources.append(f"Patient Record: {patient_id}")
                elif tool_name == "bed_availability_lookup":
                    sources.append("SQLite Wards Bed Allocation")
            except Exception as e:
                outputs.append({
                    "tool": tool_name,
                    "args": args,
                    "result": f"Error during execution: {str(e)}"
                })
        else:
            outputs.append({
                "tool": tool_name,
                "args": args,
                "result": f"Error: Tool '{tool_name}' is not registered."
            })
            
    # Remove duplicate sources
    unique_sources = sorted(list(set(sources)))
    
    return {
        "tool_outputs": outputs,
        "current_tool_execution": [], # Clear task list once executed
        "sources_used": unique_sources
    }

def context_builder_node(state: AgentState) -> Dict[str, Any]:
    """
    Deduplicates and organizes all tool execution outputs into clean sections:
    Hospital Knowledge, Patient Information, Historical Notes, and Similar Cases.
    """
    tool_outputs = state.get("tool_outputs", [])
    
    hospital_knowledge = []
    patient_info = []
    historical_notes = []
    similar_cases = []
    
    for run in tool_outputs:
        name = run["tool"]
        res = run["result"]
        
        if name == "hospital_knowledge_search":
            try:
                items = json.loads(res)
                for item in items:
                    source_name = os.path.basename(item.get("source", "Unknown"))
                    hospital_knowledge.append(f"[{source_name}] {item.get('text', '').strip()}")
            except Exception:
                hospital_knowledge.append(res)
                
        elif name == "patient_lookup":
            patient_info.append(res)
            
        elif name == "patient_history_search":
            try:
                items = json.loads(res)
                for item in items:
                    historical_notes.append(f"[Patient {item.get('patient_id')}] {item.get('text', '').strip()}")
            except Exception:
                historical_notes.append(res)
                
        elif name == "similar_case_search":
            try:
                items = json.loads(res)
                for item in items:
                    similar_cases.append(f"Patient ID: {item.get('patient_id')} (Distance: {item.get('distance')})\nSummary: {item.get('summary', '').strip()}")
            except Exception:
                similar_cases.append(res)
                
        elif name == "bed_availability_lookup":
            hospital_knowledge.append(f"Wards Occupancy:\n{res}")
            
        elif name in ("register_patient", "modify_patient_record"):
            patient_info.append(res)
            
    # Deduplicate text segments
    hospital_knowledge = sorted(list(set(hospital_knowledge)))
    patient_info = sorted(list(set(patient_info)))
    historical_notes = sorted(list(set(historical_notes)))
    similar_cases = sorted(list(set(similar_cases)))
    
    # Structure the combined text
    sections = []
    if hospital_knowledge:
        sections.append("### Hospital Knowledge & Guidelines\n" + "\n\n".join(hospital_knowledge))
    if patient_info:
        sections.append("### Patient Database Records\n" + "\n\n".join(patient_info))
    if historical_notes:
        sections.append("### Patient Medical Histories (FAISS)\n" + "\n\n".join(historical_notes))
    if similar_cases:
        sections.append("### Similar Clinical Cases\n" + "\n\n".join(similar_cases))
        
    final_context = "\n\n---\n\n".join(sections)
    if not final_context.strip():
        final_context = "No relevant context was retrieved by tools."
        
    return {
        "final_context": final_context
    }

def llm_node(state: AgentState) -> Dict[str, Any]:
    """
    Renders system instruction and retrieved context, queries the Groq client,
    and returns a structured Markdown response with reasoning and sources.
    """
    api_key = state.get("api_key")
    if not api_key:
        return {"final_response": "Error: Groq API Key is missing."}
        
    query = state.get("query", "")
    history = state.get("history", [])
    context = state.get("final_context", "")
    sources = state.get("sources_used", [])
    
    # Format conversation history
    history_str = ""
    for msg in history:
        history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
    system_prompt = f"""You are VeraOps, an Agentic AI Hospital Assistant. 
Analyze the user's clinical query and generate a precise, professional response based on the Context and Conversation History below.

CONTEXT:
{context}

RESPONSE STRUCTURE RULES:
Your answer MUST contain the following 5 specific markdown headings in order:
1. ### CLINICAL ANSWER
   Provide a direct, clear answer to the user's inquiry.
2. ### CLINICAL REASONING
   Explain the clinical reasoning, SOP references, or calculations leading to your answer.
3. ### RELEVANT PATIENT DATA
   Summarize active patient statuses, symptoms, or demographics related to this query.
4. ### HOSPITAL REFERENCES
   List the guidelines or policies referenced.
5. ### SOURCES USED
   List all specific database records, patient folders, or SOP documents cited.
"""

    user_prompt = f"""Conversation History:
{history_str}
Doctor Query: {query}

Please formulate the clinical response."""

    try:
        from services.config_service import get_setting
        client = get_groq_client(api_key)
        model = get_setting("model_selection", "llama-3.3-70b-versatile")
        temperature = get_setting("temperature", 0.1)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        response_text = completion.choices[0].message.content
        
        # Format the sources at the bottom if LLM omitted them
        if "### SOURCES USED" not in response_text:
            source_list_str = "\n".join([f"- {s}" for s in sources]) if sources else "- None"
            response_text += f"\n\n### SOURCES USED\n{source_list_str}"
            
        return {
            "final_response": response_text
        }
    except Exception as e:
        return {
            "final_response": f"Error invoking supervisor model: {str(e)}"
        }
