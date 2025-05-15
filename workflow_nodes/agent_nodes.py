import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from agent_types import AgentState, Plan, SelectedTool, Code, Reflection, PlanEditor
from .core_nodes import tool_doc_retrieval

# Define System Prompts for all LLM Agents & Agent Node Function
# ==================== Conductor Agent ====================
conductor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a routing agent. Based on the user's input, decide which of the following agents should handle the request.
            
            Agents and their purposes:
            
            1. frontdesk_agent – Use for greetings, general question, general scRNA-seq questions, or requests that DON'T clearly involve execution of data analysis and plotting. 
                • Output: Conversational response.
            
            2. plan_generator_agent – Use when the user wants to:
                - Perform an analysis
                - Perform a new analysis,
                - Generate or interpret a figure not covered by the current plan,
                - Analyze or visualize data files (e.g. .h5ad, .csv, .pdf) not yet analyzed,
                - Explore a new step or sub-analysis outside the existing plan.
                • Output: New list of sequenced tasks (a new plan).
            
            3. plan_editor_agent – Use when the user wants to refine, modify, or discuss changes to an existing analysis plan.
                • Output: Edited list of tasks.
            
            4. analysis_agent – Use only when:
                - A plan already exists,
                - The user has confirmed, approved, or agreed to proceed with executing the plan.
                • Output: Results or interpretations from the execution of the plan.

            If the user intent is unclear, default to **plan_generator_agent**.
            
            Respond ONLY with one of the following agent names: frontdesk_agent, plan_generator_agent, plan_editor_agent, or analysis_agent.
            
            Always look at the conversation history for context before routing.
            Here is the conversation historu: {conversation_history}
            
            """
        ),
        ("user","{user_prompt}")
    ]
)

conductor_agent = conductor_prompt | ChatOpenAI(model = "gpt-4o-mini", temperature = 0)

def conductor_node(state: AgentState):
    logging.info("---Initiate Conductor Agent---")
    print("---Initiate Conductor Agent---")
    
    user_prompt = state["user_prompt"]
    conversation_history = state.get("conversation_history","No History")
    logging.info(f"user_prompt: {user_prompt}")
    
    conductor_result = conductor_agent.invoke({"user_prompt":user_prompt,"conversation_history":conversation_history})
    
    conductor_content = conductor_result.content
    logging.info(f"conductor_result:\n{conductor_content}")
    print(f"conductor_result: {conductor_content}")
    
    return {"conductor_status": conductor_content}

# ==================== FrontDesk Agent ====================

frontdesk_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are SCAgent, a Front Desk Agent for a single-cell analysis AI workspace. 
            
            Respond naturally to casual greetings, general daily question, general AI/bioinformatics questions, or basic non-technical inquiries based on user_prompt.
            
            Here is conversation history for context: {conversation_history}
            
            """
        ),
        ("user", "{user_prompt}")
    ]
)

frontdesk_agent = frontdesk_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)

def frontdesk_node(state: AgentState):
    logging.info("---Initiate FrontDesk Agent---")
    print("---Initiate FrontDesk Agent---")
    user_prompt = state["user_prompt"]
    conversation_history = state["conversation_history"]
    
    # Invoke the front desk LLM with current message history
    frontdesk_result = frontdesk_agent.invoke({"user_prompt": user_prompt, "conversation_history":conversation_history})
    
    frontdesk_content = frontdesk_result.content
    logging.info(f"frontdesk_content:\n{frontdesk_content}")
    print(f"frontdesk_content: {frontdesk_content}")
    
    return {"conversation_history": frontdesk_result}

# ==================== Plan Editor Agent ====================
plan_editor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         You are a helpful assistant specialized in editing multi-step plans based on user feedback.
         The plan to revise is:
         {plan}
 
         Your task is to revise the plan according to the user's instructions and return only a valid Python list of revised steps. 
         Do not include any explanation, comments, or numbering—just the Python list.          
         """
        ),
        ("user","Here is user feedback: {user_prompt}"),
    ]
)
plan_editor_agent = plan_editor_prompt | ChatOpenAI(model = "gpt-4.1-mini", temperature = 0).with_structured_output(PlanEditor)

def plan_editor_node(state:AgentState):
    
    print("---Initiate Plan Editor Agent---")
    logging.info("---Initiate Plan Editor Agent---")
    
    user_prompt = state["user_prompt"]
    plan = state["plan"]
    
    if not plan:
        raise ValueError(f"Plan list is empty!")
    elif plan:
        plan_editor_result = plan_editor_agent.invoke({"plan":plan, "user_prompt":user_prompt})
        edited_plan = plan_editor_result.edited_plan
        print(f"edited_plan:\n{edited_plan}")
        logging.info(f"edited_plan:\n{edited_plan}")
        formatted_steps = "\n".join([f"{idx+1}. {step}" for idx, step in enumerate(edited_plan)])
        new_message = AIMessage(content = f"This is the revised plan, let me know if you still need any changes:\n\n{formatted_steps}")
        logging.info(f"new_message:\n{new_message.content}")
        return {"plan":edited_plan, "conversation_history": new_message}

# ==================== Planner Agent ====================
planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         You are an expert Planner Agent in bioinformatics and scRNA-seq analysis. 
         
         Your role is to break down the user's request into a logically ordered sequence of major subtasks involved in single-cell RNA-seq analysis. 
         These subtasks should represent the standard stages of analysis.
         
         The `selected_tool` is a tool suitable for user request. The `tool_context` contain tool-specific parameters/instructions, use it as context during planning.
         
         **Requirements for each Subtask:**
         1. Each subtask should reflect a core step in scRNA-seq analysis.
         2. Follow a scientifically valid sequence that reflects standard workflows.
         3. Omit redundant or irrelevant steps.
         4. The final task should represent the ultimate output of the analysis, such as producing annotated visualizations or summaries.
         5. The final subtask should produce interpretable outputs (e.g., annotated data or visualizations).
         
         **Input File Path Extraction:**
         - If the user mentions input file(s), extract the file path(s) and description(s) into a dictionary called `input_file_path`.
         - Ensure `input_file_path` is formatted as a dictionary where:
           - Each **key** is the full path to a file (e.g., "/path/to/file.h5ad")
           - Each **value** is a short human-readable description of what that file contains
         - If no file is mentioned, set `input_file_path` to `null`.
         
         **Output Format:**
         Return:
         - A clearly ordered list of subtasks in the workflow.
         - A correctly structured `input_file_path` dictionary as defined above.         
         """
        ), 
        ("user", "user prompt: {user_prompt}. selected_tool: {selected_tool}. tool_context: {tool_context}"),
    ]
)

planner_agent = planner_prompt | ChatOpenAI(model="gpt-4.1", temperature=0).with_structured_output(Plan)
def planner_node(state: AgentState):
    print("---Initiate Planner Agent---")
    logging.info("---Initiate Planner Agent---")

    user_prompt = state["user_prompt"]
    selected_tool: List[str] | None = state["selected_tool"]
    logging.info(f"selected_tool: {selected_tool}")
    
    if selected_tool[0] != "None":
        tool_context = tool_doc_retrieval(selected_tool)
    else: 
        tool_context = ""
         
    planner_result = planner_agent.invoke({"user_prompt": user_prompt, "selected_tool":selected_tool, "tool_context":tool_context})
    generated_plans = planner_result.steps
    input_file_path = planner_result.input_file_path
    print(f"planner result: {generated_plans}")
    logging.info(f"planner result:\n{generated_plans}")
    print(f"extracted input_file_path: {input_file_path}")
    logging.info(f"extracted input_file_path: {input_file_path}")
    
    formatted_steps = "\n".join([f"{idx+1}. {step}" for idx, step in enumerate(generated_plans)])
    all_plans_message = AIMessage(content=f"Here is the planned sequence of tasks. Let me know if you need any changes:\n\n{formatted_steps}")
    logging.info(f"all_plans_message:\n{all_plans_message.content}")
    return {"plan": generated_plans, "conversation_history": all_plans_message, "input_file_path": input_file_path}


# ==================== Tool Selector Agent ====================
tool_selector_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         You are a Tool Selector Agent specializing in bioinformatics and single-cell RNA-seq analysis.
         Select the most relevant tool(s) from the provided list for the user's task. Output a list of tool names, or ["None"] if none are suitable.
         Do **not** create new tools or hallucinate capabilities not in the provided list.

         Available Tool List: {tools_dict}
         Current subtask:
         """
        ), ("user", "{current_task}"),
    ]
)
tool_selector_agent = tool_selector_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(SelectedTool)

def tool_selector_node_one(state: AgentState):
    print("---Initiate Tool Selector Agent---")
    logging.info("---Initiate Tool Selector Agent---")
    current_subtask = state["current_task"]
    tool_dictionary = state["available_tools"]
    
    print(f"current task: {current_subtask}")
    logging.info(f"current task: {current_subtask}")

    task_formatted = f"""
    This the list for overall plan: {state["plan"]} \n
    You are tasked with selecting tool for this step:{current_subtask}.
    """
    tool_selector_result = tool_selector_agent.invoke({"current_task": task_formatted,"tools_dict": tool_dictionary})
    if tool_selector_result.tools[0] != "None":
        selected_tool_result = tool_selector_result.tools[0] # Only one tool is selected
        logging.info(f"Tool selector result: {selected_tool_result}\n")
        print(f"Tool selector result: {selected_tool_result}\n")
    else:
        selected_tool_result = ["None"]
        print("Tool selector did not select any tool.\n")
        logging.info("Tool selector did not select any tool.\n")
    return {"selected_tool": selected_tool_result}

def tool_selector_node_two(state: AgentState):
    print("---Initiate Tool Selector Agent---")
    logging.info("---Initiate Tool Selector Agent---")

    current_subtask = state["user_prompt"]
    task_formatted = f"""You are tasked with selecting tool based on this task: {current_subtask}"""

    tool_dictionary = state["available_tools"]
    
    print(f"current task: {current_subtask}")
    logging.info(f"current task: {current_subtask}")

    tool_selector_result = tool_selector_agent.invoke({"current_task": task_formatted,"tools_dict": tool_dictionary})
    if tool_selector_result.tools[0] != "None":
        selected_tool_result = tool_selector_result.tools[0] # Only one tool is selected
        logging.info(f"Tool selector result: {selected_tool_result}\n")
        print(f"Tool selector result: {selected_tool_result}\n")
    else:
        selected_tool_result = ["None"]
        print("Tool selector did not select any tool.\n")
        logging.info("Tool selector did not select any tool.\n")
    return {"selected_tool": selected_tool_result}

# ==================== Code Generation Agent ====================
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a Python code generation agent specialized in single-cell RNA-seq using Scanpy.
            Generate a **self-contained**, **immediately executable** code block for a single task in the workflow.

            ### Inputs:
            - `selected_tool`: Tool to use for the task (fallback to Scanpy if not given).
            - `tool_context`: Tool-specific parameters/instructions.
            - `code`: Newly generated Python code that contain errors.
            - `output_messages`: Accumulated code output messages from previous successful tasks. 
            - `current_task`: Task to implement.
            - `input_file_path`: input data file for the *entire workflow*.

            ### Guidelines:
            1. Implement **only** the current task — no placeholders, no mixing unrelated logic.
            2. Use the selected tool if provided; otherwise follow standard Scanpy best practices.
            3. Save outputs to `<base_dir>/../results/`, creating the directory if needed.
            4. Use `try-except` for file I/O. Use `sys.exit(1)` for critical failures.
            5. **Print informative messages** at each major step:
                - What step is being run.
                - What the result/output represents.
                - Where the result was saved (with filename and using absolute path, do not guess).
                - What this step achieves in the overall workflow.
            6. Follow Scanpy workflow standards:
                - Preserve sparsity in `adata.X` unless dense ops are explicitly needed.
                - Validate presence of needed data in `adata` before relying on it.
            7. Use parameters provided in the tool context for the current task (if provided), else, use common parameters relevant to the current task, and **state them clearly in the print messages**.
            8. Use the `output_messages` as context to understand what was done previously. 

            **Restrictions:**
            - No internet access unless explicitly allowed via the tool context.
            - Output must be **raw Python code only**—no markdown formatting, no surrounding text, no explanations, no code fences (```). Just the code for the current task.
            """
        ),
        ("placeholder", "{messages}"),
        ("user", "Code context:\n\n{code}\n\n- selected_tool: {selected_tool}\n- output_messages: {output_messages}\n- tool_context: {tool_context}\n- input_file_path: {input_file_path}\n\nGenerate the Python code block for the current task: {current_task}"),
    ]
)
# 5. Use variables from prior code (e.g., `adata`) but dont redefine them unnecessarily.

code_gen_agent = code_gen_prompt | ChatOpenAI(temperature=1, model="gpt-4.1").with_structured_output(Code)

def code_generator_node(state: AgentState):

    print("---Initiate Code Generator Agent---")
    logging.info("---Initiate Code Generator Agent---")
    # State Access
    current_task = state["current_task"] # Access current task
    current_task_index = state["current_task_index"] # Access current task index
    messages = state["messages"] # Access messages
    input_file_path = state["input_file_path"]
    selected_tool = state["selected_tool"] # Access selected tool
    iterations = state["iterations"]  # Access iterations
    error: str | None = state["error"] # Access error  
    stdout_output = state["stdout_output"]
    prev_code = state["code_generation"] # Access previous generated code
    all_generated_code = state["all_generated_code"] # Access all generated code
    
    # Conversion
    if prev_code != "" and hasattr(prev_code, "imports") and hasattr(prev_code, "code"):
        prev_code = f"{prev_code.imports} \n {prev_code.code}"
    else:
        prev_code = ""
    
    # # generate all code
    # if prev_code != "":
    #     generated_code = f"This previous code block contain an ERROR and must be solved:\n{prev_code}\n" 
    #     # all_code = f"This previous code block contains no error and is for context only:\n{all_generated_code}\nThis previous code block contain an ERROR and must be solved:\n{prev_code}\n" 
    # else:
    #     generated_code = f"No error"
    #     # all_code = f"This previous code block contains no error and is for context only:\n{all_generated_code}\n"

    # tool documentation
    if selected_tool[0] != "None":
        tool_docs = tool_doc_retrieval(selected_tool)
    else: 
        tool_docs = ""
    
    print(f"current task:\n\t{current_task}")
    print(f"current task index:\n\t{current_task_index}")
    print(f"the selected tool:\n\t{selected_tool}")
    print(f"current iterations:\n\t{iterations}")
    print(f"error:\n\t{error}")
    print(f"input_file_path:\n\t{input_file_path}")
    print(f"messages:\n\t{messages}")
    print("\n"+"="*80+"\n")
    print(f"previous code:\n\t{prev_code}") #If this print nothing,it means no error was occur for current task.
    print("\n"+"="*80+"\n")
    print(f"output_messages:\n\n{stdout_output}")
    print(f"Retrieved Tool Docs\n\t{tool_docs[:90]}")
    logging.info(f"current task: {current_task}")
    logging.info(f"current task index: {current_task_index}")
    logging.info(f"selected tool: {selected_tool}")
    logging.info(f"current iterations: {iterations}")
    logging.info(f"error: {error}")
    logging.info(f"input_file_path: {input_file_path}")
    logging.info(f"Retrieved Tool Docs\n\n{tool_docs[:90]}")
    logging.info("\n"+"="*80+"\n")
    logging.info(f"messages:\n\n{messages}")
    logging.info(f"output_messages:\n\n{stdout_output}")
    logging.info("\n"+"="*80+"\n")
    # logging.info(f"code with error:\n\n{generated_code}")
    logging.info("\n"+"="*80+"\n")
        
    # We have been routed back to generation with an error
    if error == "yes":
        messages += [HumanMessage(content="The previous attempt failed. Please review the error and suggestions, then try generating the code again.")]    
 
    # Invoke LLM
    code_solution= code_gen_agent.invoke(
        {
            "messages": messages,
            "current_task": current_task,
            "tool_context": tool_docs, 
            "input_file_path": input_file_path, 
            "selected_tool": selected_tool, 
            "code":prev_code,
            "output_messages": stdout_output,
         }
    )

    # Increment
    new_iterations: int = iterations + 1
    print("\n"+"="*80+"\n")
    print(f"my current prexif:\n\n{code_solution.prefix}\n\nmy final code:\n\n{code_solution.code}\n\nmy final imports:\n\n{code_solution.imports}")
    print("\n"+"="*80+"\n")
    print(f"iterations after invoke:\n\t{new_iterations}")
    logging.info("\n"+"="*80+"\n")
    logging.info(f"my current prexif:\n\n{code_solution.prefix}\n\nmy final code:\n\n{code_solution.code}\n\nmy final imports:\n\n{code_solution.imports}")
    logging.info("\n"+"="*80+"\n")
    logging.info(f"iterations after invoke:\n\t{new_iterations}")
    return {"messages": messages, "code_generation": code_solution, "iterations": new_iterations}   

# ==================== Reflection Agent ====================
code_reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a code reflection agent helping with debugging and improving Python code executed in a sandboxed environment.

            Code snapshot (current faulty code):
            {error_code}

            Previous successful code:
            {all_generated_code}

            Tool context:
            {tool_context}

            Your task is to:
            - Analyze the code and associated error message to identify the **root cause**.
            - Suggest **specific, minimal, and actionable** changes — avoid generic advice.
            - Use stdout logs to trace execution and pinpoint logic failures.
            - Detect misunderstandings of object structure, attributes, or data flow.

            Constraints:
            - The code runs in a sandbox; **temporary files and outputs are cleared** after execution.
            - Suggest passing intermediate results via variables when necessary.
            """
        ),
        ("placeholder", "{messages}")
    ]
)

reflection_agent = code_reflection_prompt | ChatOpenAI(temperature=0, model="gpt-4.1").with_structured_output(Reflection)  
def reflect_node(state: AgentState):
    """
    Reflect on errors and suggest improvements.
    """

    print("---Initiate Reflect Agent---")
    logging.info("---Initiate Reflect Agent---")

    # State
    messages = state["messages"]
    error_code = state["code_generation"]    
    previous_code = state["all_generated_code"]
    selected_tool = state["selected_tool"] 
    replan_triggered = state["replan_triggered"]
    iterations = state["iterations"]
    
    if replan_triggered == True and iterations == 5:
        raise RuntimeError("Repeated Error occur even after Replan. Kindly identify root problem.")
    else:
        # tool documentation
        if selected_tool[0] != "None":
            tool_docs = tool_doc_retrieval(selected_tool)
        else: 
            tool_docs = ""
        
        # Prompt reflection
        # Add reflection
        reflections_result = reflection_agent.invoke(
            {
                "messages": messages, 
                "error_code": error_code,
                "all_generated_code": previous_code,
                "tool_context": tool_docs, 
            }
        )
        reflection_message = AIMessage(content=f"Here are reflections on the error: {reflections_result.error} \n Here are the Suggestions:{reflections_result.suggestion}")
        print(f"reflection message:\n\n{reflection_message.content}")
        logging.info(f"reflection message:\n\n{reflection_message.content}")
        return {"messages": [reflection_message]}

# ==================== Replan Agent ====================
replanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a replanning agent. 
            Your goal is to output a single, clear, and directly actionable revised task that addresses any shortcomings of the "current_task" identified in the "messages". 
            Focus on making the revised task unambiguous and immediately implementable by a code generation agent.
            Consider the original plan and the user's initial request to guide your revision. Do not revise the entire plan or other tasks; focus only on the current subtask.
            """
        ),
        (
            "user",
            """Previous Interactions:
            {messages}
            
            current task:
            {current_task}
            
            original plan:
            {plan}
            
            Based on the previous interactions, provide a clear and precise revised task.
            """
        )
    ]
) 
replan_agent = replanner_prompt | ChatOpenAI(temperature=0, model="gpt-4.1-mini")
def replan_node(state: AgentState):
    """
    Generate a revised task based on the error and suggestions.
    """
    print("---Initiate Replan Agent---")
    logging.info("---Initiate Replan Agent---")

    # State
    messages = state["messages"]
    current_task = state["current_task"]
    plan_list = state["plan"]
    current_task_index = state["current_task_index"]

    # Prompt replan
    replan_result = replan_agent.invoke(
        {
            "messages": messages, 
            "current_task": current_task,
            "plan": plan_list,
        }
    )
    revised_task = replan_result.content
    plan_list[current_task_index] = revised_task
    replan_message = AIMessage(content=f"Here is the revised task: {revised_task} from this original task: {current_task}")
    print(f"New revised task:\n\t{revised_task}")
    logging.info(f"replan_message:\n\n{replan_message.content}")
    return {"replan_triggered": True, "plan": plan_list, "current_task": revised_task, "messages": replan_message}

# ==================== Reporter Agent ====================
reporter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a precise , structured and insightful Report Generator Agent.
            
            Your task is to read the final output message from a program, which may include printed logs, analysis steps, intermediate status updates, result, file paths, and error messages.
            
            Generate a clear, human-readable report that includes:
            1. What was done or achieved.
            2. What the results or outputs were.
            3. Any saved outputs: where they are saved and what they represent.
            4. If possible, briefly interpret what the key outputs mean in a biological or analytical context, without guessing beyond the data. Highlight any patterns, dominant cell types, or notable distributions.
            5. Summarize dominant patterns or findings (e.g., number of clusters, major cell types) if present.
            
            Format the report using bullet points or Markdown for readability. Do not speculate or infer beyond what is present in the message."
            """
        ),
        (
            "user",
            """
            Here is the generated output: {final_output_message}
            """
        )
    ]
)
reporter_agent = reporter_prompt | ChatOpenAI(temperature=0, model="gpt-4o-mini")
def reporter_node(state: AgentState):
    
    compiled_messages = state["messages"]
    all_generated_code = state["all_generated_code"]
    all_stdout = state["stdout_output"]
    
    logging.info(f"Compiled messages:\n\n{compiled_messages}")
    logging.info(f"all generated code:\n\n{all_generated_code}")
    logging.info(f"all_stdout:\n\n{all_stdout}")

    report_result = reporter_agent.invoke({"final_output_message": all_stdout})
    
    summary_content = report_result.content
    print(f"Here is your Report of your output:\n\n{summary_content}\n\n")
    logging.info(f"Here is your Report of your output:\n\n{summary_content}\n\n")
    return {"conversation_history": report_result}
    