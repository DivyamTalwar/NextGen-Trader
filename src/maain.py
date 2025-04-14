import streamlit as st
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

from langchain_core.messages import HumanMessage
from graph.state import AgentState
from langgraph.graph import END, StateGraph
from agents.ben_graham import ben_graham_agent
from agents.bill_ackman import bill_ackman_agent
from agents.fundamentals import fundamentals_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.technicals import technical_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.sentiment import sentiment_agent
from agents.warren_buffett import warren_buffett_agent
from agents.valuation import valuation_agent
from utils.analysts import ANALYST_ORDER, get_analyst_nodes
from utils.progress import progress
from llm.models import LLM_ORDER, get_model_info
from utils.visualize import save_graph_as_png

from dotenv import load_dotenv

load_dotenv()

# ----- Enhanced Professional & Eye-Catching UI Styling -----
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    /* Overall body settings */
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #16222A, #3A6073);
        color: #E0E0E0;
        margin: 0;
        padding: 0;
    }
    
    /* Main container styling */
    .reportview-container .main {
        background: transparent;
        padding: 2rem;
    }

    /* Sidebar styling */
    .css-1d391kg {  
        background: linear-gradient(135deg, #2b5876, #4e4376);
        color: #ffffff;
        border: none;
    }
    .css-1d391kg .sidebar-content {
        background: transparent;
    }
    [data-testid="stSidebar"] .css-1d391kg {
        background: linear-gradient(135deg, #2b5876, #4e4376);
    }
    
    /* Title Styling */
    h1 {
        font-weight: 700;
        font-size: 3.2em;
        color: #00C5B3;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.6);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Info block styling */
    .css-1lsmgbg {
        background-color: #222;
        border-left: 8px solid #00C5B3;
        font-size: 1.1em;
    }
    
    /* Buttons styling */
    div.stButton > button {
        background-color: #00C5B3;
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 14px 28px;
        font-size: 1em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #00a39b;
        transform: translateY(-3px);
        box-shadow: 0px 7px 12px rgba(0, 0, 0, 0.5);
    }
    
    /* Input fields styling */
    input[type="text"],
    input[type="number"] {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
        background: #fdfdfd;
        color: #333;
        font-size: 1em;
    }
    
    /* Multiselect and selectbox styling */
    .css-1ex4k6z, .css-1hwfws3 {
        border-radius: 8px;
        padding: 8px;
        font-size: 1em;
    }
    
    /* Simulation Parameters Card Styling */
    .param-card {
        background: rgba(0, 0, 0, 0.5);
        padding: 25px;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.7);
    }
    .param-card h2 {
        color: #00C5B3;
        font-weight: 700;
        margin-bottom: 1rem;
        font-size: 2.2em;
        border-bottom: 2px solid #00C5B3;
        padding-bottom: 0.5rem;
    }
    .param-card ul {
        list-style: disc;
        margin-left: 20px;
    }
    .param-card li {
        margin: 12px 0;
        font-size: 1.2em;
    }
    .param-label {
        color: #00C5B3;
        font-weight: 700;
        margin-right: 0.5rem;
    }
    .param-value {
        font-weight: 600;
    }
    /* Custom bullet list styling for nested lists */
    .nested-list {
        list-style: '▹ '; 
        margin-left: 2rem;
        padding-left: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1>AI-Powered Hedge Fund Simulator</h1>", unsafe_allow_html=True)

# ----- Info Node for Models -----
st.info("Note: Please use GROQ models (llama-3.3-70b-versatile) For Best Results. OpenAI keys are not rechargeable, so consider these free models.")

# Sidebar: Configuration Section
st.sidebar.title("Configuration")

with st.sidebar.expander("Basic Settings", expanded=True):
    initial_cash = st.number_input("Initial Cash", value=100000.0, step=1000.0, format="%.2f")
    margin_requirement = st.number_input("Margin Requirement", value=0.0, step=1.0, format="%.2f")
    # Revert ticker input back to text input
    tickers_input = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, GOOGL")

with st.sidebar.expander("Date Settings", expanded=True):
    start_date_input = st.text_input("Start Date (YYYY-MM-DD)", value="")
    end_date_input = st.text_input("End Date (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"))

with st.sidebar.expander("Advanced Options", expanded=False):
    show_reasoning = st.checkbox("Show Reasoning", value=False)
    show_agent_graph = st.checkbox("Show Agent Graph", value=False)
    
    # Analyst selection using multiselect (dropdown style remains)
    analyst_choices = [value for display, value in ANALYST_ORDER]
    selected_analysts = st.multiselect("Select AI Analysts", options=analyst_choices, default=analyst_choices)

    # Model selection using selectbox
    model_choice = st.selectbox("Select LLM Model", options=[value for display, value, _ in LLM_ORDER])
    model_info = get_model_info(model_choice)
    if model_info:
        model_provider = model_info.provider.value
    else:
        model_provider = "Unknown"
    st.markdown(f"**Selected Model:** {model_choice} ({model_provider})")

# Process tickers input from text field
tickers = [ticker.strip() for ticker in tickers_input.split(",") if ticker.strip()]

if not end_date_input:
    end_date = datetime.now().strftime("%Y-%m-%d")
else:
    try:
        datetime.strptime(end_date_input, "%Y-%m-%d")
        end_date = end_date_input
    except ValueError:
        st.error("End date must be in YYYY-MM-DD format")
        st.stop()

if not start_date_input:
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
else:
    try:
        datetime.strptime(start_date_input, "%Y-%m-%d")
        start_date = start_date_input
    except ValueError:
        st.error("Start date must be in YYYY-MM-DD format")
        st.stop()

# ----- Display Simulation Parameters in a Styled Card (as bullet lists) -----
st.markdown(
    f"""
    <div class="param-card">
      <h2>Simulation Parameters</h2>
      <ul>
         <li>
             <span class="param-label">Tickers:</span>
             <ul class="nested-list">
                 {''.join([f"<li class='param-value'>{ticker}</li>" for ticker in tickers])}
             </ul>
         </li>
         <li><span class="param-label">Start Date:</span> <span class="param-value">{start_date}</span></li>
         <li><span class="param-label">End Date:</span> <span class="param-value">{end_date}</span></li>
         <li><span class="param-label">Initial Cash:</span> <span class="param-value">${initial_cash:,.2f}</span></li>
         <li><span class="param-label">Margin Requirement:</span> <span class="param-value">{margin_requirement}</span></li>
         <li>
             <span class="param-label">Selected Analysts:</span>
             <ul class="nested-list">
                 {''.join([f"<li class='param-value'>{analyst}</li>" for analyst in selected_analysts])}
             </ul>
         </li>
         <li><span class="param-label">LLM Model:</span> <span class="param-value">{model_choice} ({model_provider})</span></li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Build the portfolio dictionary
portfolio = {
    "cash": initial_cash,
    "margin_requirement": margin_requirement,
    "positions": {
        ticker: {
            "long": 0,
            "short": 0,
            "long_cost_basis": 0.0,
            "short_cost_basis": 0.0,
        } for ticker in tickers
    },
    "realized_gains": {
        ticker: {
            "long": 0.0,
            "short": 0.0,
        } for ticker in tickers
    },
}

# Create workflow from selected analysts
def create_workflow(selected_analysts=None):
    workflow = StateGraph(AgentState)
    # Define start node
    workflow.add_node("start_node", lambda state: state)
    analyst_nodes = get_analyst_nodes()
    if selected_analysts is None or not selected_analysts:
        selected_analysts = list(analyst_nodes.keys())
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)
    # Always add risk management and portfolio management nodes
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")
    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)
    workflow.set_entry_point("start_node")
    return workflow

workflow = create_workflow(selected_analysts)
app = workflow.compile()

if show_agent_graph:
    file_path = "agent_graph.png"
    save_graph_as_png(app, file_path)
    st.image(file_path, caption="Agent Graph", use_column_width=True)

# Run simulation button
if st.button("Run Hedge Fund Simulation"):
    with st.spinner("Running simulation..."):
        progress.start()
        try:
            final_state = app.invoke({
                "messages": [HumanMessage(content="Make trading decisions based on the provided data.")],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_choice,
                    "model_provider": model_provider,
                },
            })
            decisions = json.loads(final_state["messages"][-1].content)
            analyst_signals = final_state["data"]["analyst_signals"]

            st.markdown("### Trading Decisions")
            st.json(decisions)
            st.markdown("### Analyst Signals") 
            st.json(analyst_signals)
        except Exception as e:
            st.error(f"Error during simulation: {e}")
        finally:
            progress.stop()
