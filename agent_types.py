from typing import Annotated, List, TypedDict, Dict, Optional, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field 

class PlanEditor(BaseModel):
    edited_plan: List[str] = Field(..., description="Revised plan list")

class Plan(BaseModel):
    """Plan of sequential steps to be followed."""
    steps: List[str] = Field(..., description="Ordered list of steps to follow.")
    input_file_path: Optional[Dict[str,str]] = Field(None, description="Dictionary where keys are file paths and values are their descriptions.")

class SelectedTool(BaseModel):
    """Selected tools name for the current task."""
    tools: List[str] = Field(..., description="List of selected tool names. Return ['None'] if no tools are suitable.")

class Code(BaseModel):
    """Schema for code solutions, including explanation, imports, and code."""
    prefix: str = Field(..., description="Description of the problem and approach")
    imports: str = Field(..., description="Code import statements.")
    code: str = Field(..., description="Main body of the code, excluding imports.")

class Reflection(BaseModel):
    """Reflection details capturing encountered issues and recommended fixes."""
    error: str = Field(..., description="Description of the error encountered")
    suggestion: str = Field(..., description="Suggested fix or improvement")
    
class AgentState(TypedDict):
    # Conversation history and messages and Input
    conversation_history: Annotated[Sequence[BaseMessage], add_messages]    # History of all conversation messages
    messages: Annotated[Sequence[BaseMessage], add_messages]                # Inner messages of Analysis workflow
    user_prompt: str                                                        # The latest user input prompt
    
    # Tooling and Input File Info
    available_tools: Dict[str, str]                                         # Dictionary of available tool names and descriptions
    input_file_path: str                                                    # path of your file

    # Planning and Task Management                                  
    plan: List[str]                                                         # Planned list of task steps
    current_task: Optional[str]                                             # Current task being processed
    current_task_index: int                                                 # Index of the current task step in the plan list
    selected_tool: Optional[List[str]]                                      # Tool(s) selected for the current task

    # Routing and Flow Control Flags                                    
    conductor_status: str                                                   # Flag that decide which agent to use (frontdesk_agent, plan_generator_agent or plan_editor_agent)
    error: Optional[str]                                                    # Binary flag for control flow to indicate whether test error was tripped
    replan_triggered: bool                                     
    # Execution and troubleshooting                                 
    code_generation: Optional[Code]                                         # Generated code solution object
    all_generated_code: Optional[str]                                       # Full code history for all generated steps
    stdout_output: Optional[str]                                            # All captured stdout output from code execution
    iterations: Optional[int]                                               # Number of iterations or retries attempted
