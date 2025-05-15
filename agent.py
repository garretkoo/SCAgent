from langgraph.graph import StateGraph, END, START
from langchain_core.runnables import RunnableLambda
from workflow_nodes import *
from agent_types import AgentState



class Agent:
    def __init__(self):
        """
        Activate the Multi-Agentic Workflow
        """
        
        # Define the agent workflow
        # Initialize a StateGraph with AgentState to define the agent workflow
        workflow = StateGraph(AgentState)
        
        # Define agent nodes (steps in the workflow)
        workflow.add_node("conductor_agent", RunnableLambda(conductor_node))
        workflow.add_node("frontdesk_agent", RunnableLambda(frontdesk_node))            # Handles user entry and decides whether to activate multi-agentic workflow
        workflow.add_node("plan_editor_agent", RunnableLambda(plan_editor_node))
        workflow.add_node("planner_agent", RunnableLambda(planner_node))                # Generates a high-level plan of tasks
        workflow.add_node("task_retriever", RunnableLambda(get_next_task))                    # Retrieves the next task in the plan
        workflow.add_node("tool_selector_agent_one", RunnableLambda(tool_selector_node_one))    # Selects appropriate tools for the task
        workflow.add_node("tool_selector_node_two", RunnableLambda(tool_selector_node_two))    # Selects appropriate tools for the task       
        workflow.add_node("code_generator_agent", RunnableLambda(code_generator_node))  # Generates code for the task
        workflow.add_node("code_checker", RunnableLambda(code_check))                     # Tests run generated code and checks for errors
        workflow.add_node("reflect_agent", RunnableLambda(reflect_node))                # Reflects on errors and provides suggestions
        workflow.add_node("completion_checker", lambda x: x)                                    # Dummy node to evaluate code test completion
        workflow.add_node("index_updater", RunnableLambda(update_task_index))            # Increments the task index to proceed
        workflow.add_node("replan_agent", RunnableLambda(replan_node))                  # Replan the task after repeated failure
        workflow.add_node("reporter_agent", RunnableLambda(reporter_node))              # Report the entire process and outputs

        # Connect the nodes, begin from START
        workflow.add_edge(START, "conductor_agent")
        
        # Decide which agent to choose, else activate the main analysis pipeline
        workflow.add_conditional_edges(
            "conductor_agent",
            conductor_router,
            {
                "frontdesk_agent": "frontdesk_agent",
                "plan_generator_agent": "tool_selector_node_two",
                "plan_editor_agent": "plan_editor_agent", 
                "analysis_agent": "task_retriever",
            }
        )
        
        workflow.add_edge("frontdesk_agent", END)
        workflow.add_edge("tool_selector_node_two","planner_agent")
        workflow.add_edge("planner_agent", END)
        workflow.add_edge("plan_editor_agent", END)
        workflow.add_edge("task_retriever", "tool_selector_agent_one")
    
        # Main Analysis Pipeline Activated        
        workflow.add_edge("task_retriever", "tool_selector_agent_one")
        workflow.add_edge("tool_selector_agent_one", "code_generator_agent")
        workflow.add_edge("code_generator_agent", "code_checker")

        # Check if code execution succeeded or needs rework
        workflow.add_conditional_edges(
            "code_checker",
            decide_to_finish,
            {
                "end": "completion_checker",         # No error encounted, proceed to next task
                "replan": "replan_agent",    # Encounter repeated error, replan
                "reflect": "reflect_agent",  # Encounter error, retry
            },
        )

        workflow.add_edge("reflect_agent", "code_generator_agent")
        workflow.add_edge("replan_agent", "task_retriever")

        # Decide to proceed to next task or finalize report
        workflow.add_conditional_edges(
            "completion_checker",
            should_continue,
            {
                True: "index_updater",        # More tasks to complete
                False: "reporter_agent",     # All tasks completed, generate report
            },
        )

        workflow.add_edge("index_updater", "task_retriever")
        workflow.add_edge("reporter_agent", END)

        # Compile the full workflow graph
        self.app = workflow.compile()