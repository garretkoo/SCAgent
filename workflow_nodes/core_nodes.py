import os
import logging
import shutil
import subprocess
from langchain_core.messages import HumanMessage, AIMessage
from agent_types import AgentState

# This node is responsible to assign current task for current loop.
def get_next_task(state: AgentState):
    print("---Initiate get_next_task---")
    logging.info("---Initiate get_next_task---")
    
    plan_list = state["plan"]
    
    if plan_list:  
        current_task_index = state["current_task_index"]
        
        if current_task_index < len(plan_list):
            current_task = plan_list[current_task_index]
            print(f"Current task index: {current_task_index}")
            print(f"Current task: {current_task}")
            logging.info(f"Current task index: {current_task_index}")
            logging.info(f"Current task: {current_task}")
            
            new_message = HumanMessage(content=f"Task {current_task_index+1}: {current_task}")
            
            # First loop - append the plan list message and new task message
            if not state.get("messages") and current_task_index < len(plan_list):
                plan_list_message = HumanMessage(content=f"Here are the list of tasks to be achieved: {plan_list}")
                return {
                    "current_task": current_task,
                    "messages": [plan_list_message, new_message],
                    "iterations": 0,
                    "code_generation": "",
                    "error": "no",
                }
            
            # Subsequent loop reset everything after success/replan of one task.
            elif current_task_index < len(plan_list):
                return {
                    "current_task": current_task,
                    "messages": [new_message],
                    "iterations": 0,
                    "code_generation": "",
                    "error": "no"  
                }
        else:
            raise IndexError(f"Current task index {current_task_index} is out of bounds for the plan list.")
    
    else:
        raise ValueError("Plan list is empty.")

# This node is responsible to update the current task index at the end of each loop.
def update_task_index(state: AgentState):
    print("---Initiate update_task_index---")
    logging.info("---Initiate update_task_index---")
    return {"current_task_index": state["current_task_index"] + 1, "iterations": 0}

# This node is responsible to check the code. It proceed to next task if correct, else return to the code generation step.
def code_check(state: AgentState):
    """
    Check code        
    """
    print("---Initiate Code Checker---")
    logging.info("---Initiate Code Checker---")
    
    code_solution = state["code_generation"]
    all_generated_code = state["all_generated_code"]
    current_task = state["current_task"]
    current_task_index = state["current_task_index"]
    std_output_all = state.get("stdout_output","")
    imports = code_solution.imports
    code_block = code_solution.code

    # Test run 
    temp_dir = "temp_scripts"
    os.makedirs(temp_dir, exist_ok=True)
    script_filename = "test_script.py"
    script_path = os.path.join(temp_dir, script_filename)
    
    with open(script_path, "w") as f:
        # f.write(all_generated_code + "\n" + imports + "\n" + code_block)
        f.write(imports + "\n" + code_block)
    
    try:
        result = subprocess.run(
            ["python", script_filename],
            cwd=temp_dir,  
            capture_output=True,
            text=True,
            timeout=600,
            check=True
        )
        
        # If no error occurs in "subprocess.run", the code below will be executed.
        
        # Build message content from output
        output_msg = f"Code executed successfully for task '{current_task}'."
        
        # Save the output message
        if result.stdout.strip():
            std_output_all += f"\n{result.stdout.strip()}\n"
            
        print("---CODE BLOCK CHECK: SUCCESS---")
        logging.info("---CODE BLOCK CHECK: SUCCESS---")
        
        output_message = AIMessage(content=output_msg)
        
        logging.info(f"success message: {output_message.content}")
        # Update compiled code for next steps
        compile_generated_code = all_generated_code + f"\n#Next Task: {current_task}\n" + imports + "\n" + code_block
        return {
            "messages": [output_message],
            "error": "no",
            "all_generated_code": compile_generated_code,
            "stdout_output": std_output_all,
        }
            
    except subprocess.CalledProcessError as e:
        print("---CODE BLOCK CHECK: FAILED (CalledProcessError)---")
        print(f"current task: {current_task}")
        print(f"current task index: {state['current_task_index']}")

        logging.error("---CODE BLOCK CHECK: FAILED (CalledProcessError)---")
        logging.error(f"current task: {current_task}")
        logging.error(f"current task index: {state['current_task_index']}")
        
        error_message = AIMessage(content=f"Code execution failed with error:\n\nStdout: {e.stdout.strip()}\n\nStderr: {e.stderr.strip()}")
        print(f"Error message:\n\n{error_message.content}")
        logging.error(f"Error message:\n\n{error_message.content}")
        
        return {
            "messages": [error_message],
            "error": "yes",
        }
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED (Unexpected Exception)---")
        print(f"current task: {current_task}")
        print(f"current task index: {current_task_index}")
        
        logging.error("---CODE BLOCK CHECK: FAILED (Unexpected Exception)---")
        logging.error(f"current task: {current_task}")
        logging.error(f"current task index: {current_task_index}")
        
        error_message = AIMessage(content=f"An unexpected error occurred during code execution:\n{str(e)}")
        print(f"Error message:\n\t{error_message.content}")
        logging.error(f"Error message:\n\t{error_message.content}")
        
        return {
            "messages": [error_message],
            "error": "yes",
        }
    finally:
    # Always clean up the directory
        shutil.rmtree(temp_dir)

# Theoretically not a "Node" but act like a node
def tool_doc_retrieval(tool_name):
    doc_path = os.path.join("tool_documentation", f"{tool_name}.txt")
    document_full_path = os.path.abspath(doc_path) 
    if os.path.isfile(document_full_path):
        with open(document_full_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return "" # Return empty string if no such file exist for selected tool.