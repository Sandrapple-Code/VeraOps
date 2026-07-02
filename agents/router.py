import json
from typing import Dict, Any, List
from agents.state import AgentState
from llm.groq_client import get_groq_client

def supervisor_router(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor router node that analyzes the user query, conversation history,
    and previous tool run outputs to schedule the next tools to execute, or 
    transition to output compilation.
    """
    api_key = state.get("api_key")
    if not api_key:
        return {
            "current_tool_execution": [],
            "final_response": "Error: Groq API Key is missing. Please provide it in the sidebar."
        }
        
    query = state.get("query", "")
    history = state.get("history", [])
    tool_outputs = state.get("tool_outputs", [])
    
    # Format query and conversation context
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    tool_outputs_str = ""
    if tool_outputs:
        tool_outputs_str = "\n".join([
            f"- Tool: {t['tool']}\n  Args: {json.dumps(t['args'])}\n  Result: {t['result']}"
            for t in tool_outputs
        ])

    system_prompt = """You are the Supervisor Router for VeraOps, an Agentic Hospital Assistant.
Your responsibility is to decide which tools need to be executed next to gather sufficient context 
to answer the doctor's query, or decide that enough info has been gathered.

AVAILABLE TOOLS:
1. `patient_lookup`: Retrieve structured patient details from SQLite. Takes {"identifier": "<id or name>"}.
2. `hospital_knowledge_search`: Search hospital SOPs/ICU guidelines. Takes {"query": "<search terms>"}.
3. `patient_history_search`: Retrieve raw medical history text segments from patient files. Takes {"query": "<search terms>"}.
4. `similar_case_search`: Find similar patient records & SQLite profiles. Takes {"query": "<diagnosis/symptoms>"}.
5. `bed_availability_lookup`: Check hospital ward occupancy. Takes no arguments.
6. `register_patient`: Create a patient. Takes {"patient_data_json": "<json string of all attributes>"}.
7. `modify_patient_record`: Update a patient. Takes {"patient_id": "<id>", "updates_json": "<json string of updates>"}.

ROUTING PROTOCOL:
- If a query mentions a patient by name (e.g. "Rahul Sharma"), first run `patient_lookup` to find their Patient ID (e.g., 'P001'). 
- Once you have the Patient ID in the "Previously Executed Tool Outputs", run subsequent queries (like `patient_history_search` or `similar_case_search`) using the specific Patient ID or matched symptoms.
- If the User Query is a follow-up or does not explicitly mention a patient name or Patient ID, check the Conversation History to identify if a patient (name or ID) was previously discussed. If so, automatically use that patient's name or Patient ID for tool calls (like `patient_lookup`, `patient_history_search`, etc.).
- If you need hospital policies, SOPs, or ICU guidelines, run `hospital_knowledge_search`.
- If the question can be answered completely using the outputs already gathered in the "Previously Executed Tool Outputs" section, do NOT call any more tools. Transition to finish by returning an empty "tool_calls" list.

Return ONLY a raw JSON block with the following schema (no conversational text or markdown codeblocks outside the JSON):
{
  "reasoning": "Brief explanation of your decision",
  "tool_calls": [
    {
      "tool": "name_of_tool",
      "args": { "arg_name": "arg_val" }
    }
  ]
}
"""

    user_prompt = f"""User Query: {query}

Conversation History:
{history_str}

Previously Executed Tool Outputs:
{tool_outputs_str}

Analyze the state and determine the next action."""

    try:
        from services.config_service import get_setting
        client = get_groq_client(api_key)
        model = get_setting("model_selection", "llama-3.3-70b-versatile")
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        response_text = completion.choices[0].message.content
        data = json.loads(response_text)
        
        tool_calls = data.get("tool_calls", [])
        return {
            "current_tool_execution": tool_calls
        }
    except Exception as e:
        print(f"[Router Exception] {e}")
        # On error, terminate and move to the LLM node
        return {
            "current_tool_execution": []
        }
