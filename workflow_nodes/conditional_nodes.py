from agent_types import AgentState
import logging


def conductor_router(state:AgentState):
    logging.info("---Initiate conductor_router CONDITIONAL NODE---")
    conductor_status = state["conductor_status"]
    generated_plan = state["plan"]
    print(f"conductor_status: {conductor_status}")
    print(f"generated_plan: {generated_plan}")
    
    if conductor_status == "frontdesk_agent":
        logging.info("Route to frontdesk_agent")
        return "frontdesk_agent"
    elif conductor_status == "plan_generator_agent":
        logging.info("Route to plan_generator_agent")
        return "plan_generator_agent"
    elif conductor_status == "plan_editor_agent":
        logging.info("Route to plan_editor_agent")
        return "plan_editor_agent"
    elif conductor_status == "analysis_agent" and generated_plan:
        logging.info("Route to analysis_agent")
        return "analysis_agent"
    elif conductor_status == "analysis_agent" and not generated_plan:
        logging.info("Plan is empty, plan is required to activate analysis_agent. Route back to plan_generator_agent")
        return "plan_generator_agent"
    else:
        raise ValueError(f"Routing: Unknown plan_status: {conductor_status}, ending workflow.")


def should_continue(state: AgentState):
    logging.info("---Initiate should_continue CONDITIONAL NODE---")
    if state["plan"] and state["current_task_index"] == len(state["plan"]) - 1: 
        logging.info("All Plan completed")
        return False  
    else:
        logging.info("All Plan not completed yet")
        return True 
    
def decide_to_finish(state: AgentState):
    """
    Determines whether to end code generation loop, or Reflect, or Replan.
    """
    logging.info("---Initiate decide_to_finish CONDITIONAL NODE---")
    error = state["error"]
    iterations = state["iterations"]
    max_iterations = 6 

    if error == "no":
        logging.info("---DECISION: FINISH!!!---")
        return "end"
    elif iterations >= max_iterations:
        logging.info("---DECISION: RE-PLAN---")
        return "replan"
    else:
        logging.info("---DECISION: RE-TRY SOLUTION---")
        return "reflect"

