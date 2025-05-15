import os
import time
import logging
from datetime import datetime
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import getpass
import streamlit as st

# Set up logging configuration
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Generate log filename with timestamp and unique ID for this run
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"logs/scagent_log_{current_time}.log"
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        format= "%(asctime)s - %(levelname)s -%(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  
        ]
    )
    
    # Create a logger instance
    logger = logging.getLogger("SCAgent")
    logger.info(f"New application instance started with log file: {log_filename}")
    
    return logger

load_dotenv()

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")

# Import agent
from agent import Agent
agent = Agent()
agent_app = agent.app

# Set page config and title
st.set_page_config(page_title="SCAgent – Your Single-Cell Analysis Assistant")
st.markdown("## SCAgent – Your Single-Cell Analysis Assistant")

# Available Tools Definition
tools_dict: dict[str, str] = {
    "CellTypist": "An automated cell type annotation tool for scRNA-seq datasets on the basis of logistic regression classifiers optimised by the stochastic gradient descent algorithm. CellTypist allows for cell prediction using either built-in (with a current focus on immune sub-populations) or custom models, in order to assist in the accurate classification of different cell types and subtypes.", 
    "SAM": "a foundation model for image segmentation. It is designed to segment any object in any image with minimal user input — like a point, box, or free-form mask.",
    "YOLO": "a real-time object detection algorithm that detects and classifies objects in images or videos in a single neural network pass.",
    "PRnet" : "a perturbation-conditioned generative model designed to predict transcriptional responses to novel drug perturbations at both bulk and single-cell levels.",
    "ScType": "a computational tool for the fully automated and rapid identification of cell types from single-cell RNA sequencing data by utilizing specific marker gene combinations."
    }

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "plan" not in st.session_state:
    st.session_state.plan = []
if "input_file_path" not in st.session_state:
    st.session_state.input_file_path = ""
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
if "thinking" not in st.session_state:
    st.session_state.thinking = False
if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0.0
if "stdout_output" not in st.session_state:
    st.session_state.stdout_output = ""
if "logger" not in st.session_state:
    # Initiatlize Logger
    st.session_state.logger = "Initiate Logger"
    logger = setup_logging()
    logger.info("Application started")


# UI Elements
# Function to display the cost and processing time at the bottom right
def display_metrics():
    metrics_display_html = f"""
    <div style='
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent white */
        color: #333; /* Dark text */
        padding: 10px 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        font-size: 1.1em;
        font-weight: bold;
        z-index: 9999; /* Ensure it's on top */
    '>
        Processing Time: <span style='color: #3F51B5;'>{st.session_state.processing_time:.2f}s</span><br>
        Total Cost: <span style='color: #4CAF50;'>${st.session_state.total_cost:.6f}</span>
    </div>
    """
    st.markdown(metrics_display_html, unsafe_allow_html=True)

# Main Chat Interface Logic
# 1. Display messages from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. Display thinking indicator if in thinking state
if st.session_state.thinking:
    with st.chat_message("assistant"):
        st.write("Thinking...")

# Accept user input
if prompt := st.chat_input("Ask about single-cell analysis..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append(HumanMessage(content=f"User prompt: {prompt}"))
    
    # Show user input
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Set thinking state to true and rerun to show thinking indicator
    st.session_state.thinking = True
    st.rerun()

# Process the query if in thinking state
elif st.session_state.thinking:
    start_time = time.time()
    
    with get_openai_callback() as cb:
        agent_result = agent_app.invoke({
            "user_prompt": st.session_state.messages[-1]["content"],
            "conversation_history": st.session_state.conversation_history,
            "iteration": 0,
            "error": "no",
            "all_generated_code": "",
            "available_tools": tools_dict,
            "code_generation": "",
            "current_task_index": 0,
            "replan_triggered": False,
            "plan": st.session_state.plan,
            "input_file_path": st.session_state.input_file_path,
            "stdout_output": st.session_state.stdout_output
        }, {"recursion_limit": 100})
        st.session_state.total_cost += cb.total_cost
    
    # Calculate processing time
    end_time = time.time()
    st.session_state.processing_time = end_time - start_time
    
    # Update session state with results
    assistant_msgs = agent_result["conversation_history"][-1]
    if hasattr(assistant_msgs, "content"):
        reply = assistant_msgs.content
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.conversation_history.append(assistant_msgs)
    
    # Update other state
    st.session_state.plan = agent_result["plan"]
    st.session_state.input_file_path = agent_result["input_file_path"]
    
    # Turn off thinking state
    st.session_state.thinking = False
    
    # Force a rerun to update the UI properly
    st.rerun()

# Call the function to display the metrics at the bottom right
display_metrics()