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

# Custom CSS for styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
        background-color: #0e1117;
        color: #ffffff;
    }

    .stApp {
        background-color: #0e1117;
    }

    h1 {
        color: #f0ab51;
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
        font-weight: 700;
        letter-spacing: 1px;
        text-shadow: 2px 2px 4px #000000;
    }

    h3 {
        color: #f0ab51;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .stButton>button {
        color: #ffffff;
        background-color: #f0ab51;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 700;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #d48b36;
    }

    .stTextInput>div>div>input {
        background-color: #1e2532;
        color: #ffffff;
        border-radius: 5px;
        border: 1px solid #4d596b;
        padding: 10px;
    }

    .stNumberInput>div>div>div>input {
        background-color: #1e2532;
        color: #ffffff;
        border-radius: 5px;
        border: 1px solid #4d596b;
        padding: 10px;
    }

    .stMultiSelect>div>div>div {
        background-color: #1e2532;
        color: #ffffff;
        border-radius: 5px;
        border: 1px solid #4d596b;
        padding: 10px;
    }

    .stSelectbox>div>div>div {
        background-color: #1e2532;
        color: #ffffff;
        border-radius: 5px;
        border: 1px solid #4d596b;
        padding: 10px;
    }

    .stCheckbox>label {
        color: #ffffff;
    }

    .stSidebar {
        background-color: #1e2532;
        color: #ffffff;
    }

    .stSidebar h2 {
        color: #f0ab51;
    }

    .css-1d391kg {
        background-color: #1e2532;
    }

    .css-1d391kg p, .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5 {
        color: #ffffff;
    }

    .stExpander {
        border: 1px solid #4d596b;
        border-radius: 5px;
        margin-bottom: 15px;
    }

    .stExpander>div[data-testid="stVerticalBlock"] {
        background-color: #1e2532;
        color: #ffffff;
    }

    details[open] summary:before {
        content: "▼";
        color: #f0ab51;
    }

    details summary:before {
        content: "▶";
        color: #f0ab51;
        margin-right: 7px;
    }

    .stInfo {
        background-color: #1e2532;
        border-left: 5px solid #f0ab51;
        color: #ffffff;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }

    .stJson {
        background-color: #1e2532;
        color: #ffffff;
        border-radius: 5px;
        padding: 20px;
        margin-bottom: 20px;
        overflow-x: auto;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .fade-in {
        animation: fadeIn 1s ease-in-out;
    }

    .parameter-card {
        background-color: rgba(30, 37, 50, 0.8);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    .parameter-card h3 {
        color: #f0ab51;
        border-bottom: 2px solid #f0ab51;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }

    .parameter-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
    }

    .parameter-label {
        color: #ffffff;
        font-weight: bold;
    }

    .parameter-value {
        color: #bbbbbb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with Fade-In Animation
st.markdown("<h1 class='fade-in'>AI-Powered Hedge Fund Simulator</h1>", unsafe_allow_html=True)

# Info Node for Models
st.info("Note: Please use GROQ models (llama-3.3-70b-versatile) For Best Results. OpenAI keys are not rechargeable, so consider these free models.")

# Sidebar: Configuration Section
with st.sidebar:
    st.title("Configuration")

    with st.expander("Basic Settings", expanded=True):
        initial_cash = st.number_input("Initial Cash", value=100000.0, step=1000.0, format="%.2f")
        margin_requirement = st.number_input("Margin Requirement", value=0.0, step=1.0, format="%.2f")
        tickers_input = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, GOOGL")

    with st.expander("Date Settings", expanded=True):
        start_date_input = st.text_input("Start Date (YYYY-MM-DD)", value="")
        end_date_input = st.text_input("End Date (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"))

    with st.expander("Advanced Options", expanded=False):
        show_reasoning = st.checkbox("Show Reasoning", value=False)
        show_agent_graph = st.checkbox("Show Agent Graph", value=False)

    analyst_choices = [value for display, value in ANALYST_ORDER]
    selected_analysts = st.multiselect("Select AI Analysts", options=analyst_choices, default=analyst_choices)

    model_choice = st.selectbox("Select LLM Model", options=[value for display, value, _ in LLM_ORDER])
    model_info = get_model_info(model_choice)
    model_provider = model_info.provider.value if model_info else "Unknown"
    st.markdown(f"**Selected Model:** {model_choice} ({model_provider})")

# Process tickers input
tickers = [ticker.strip() for ticker in tickers_input.split(",") if ticker.strip()]

# Date validation and processing
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

# Simulation Parameters Card
st.markdown("""
<div class="parameter-card fade-in">
    <h3>Simulation Parameters</h3>
</div>
""", unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="parameter-card fade-in">
            <div class="parameter-item">
                <span class="parameter-label">Tickers:</span>
                <span class="parameter-value">{', '.join(tickers)}</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">Start Date:</span>
                <span class="parameter-value">{start_date}</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">Initial Cash:</span>
                <span class="parameter-value">${initial_cash:,.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="parameter-card fade-in">
            <div class="parameter-item">
                <span class="parameter-label">End Date:</span>
                <span class="parameter-value">{end_date}</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">Margin Requirement:</span>
                <span class="parameter-value">{margin_requirement}</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">LLM Model:</span>
                <span class="parameter-value">{model_choice} ({model_provider})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(f"""
<div class="parameter-card fade-in">
    <div class="parameter-item">
        <span class="parameter-label">Selected Analysts:</span>
        <span class="parameter-value">{', '.join(selected_analysts)}</span>
    </div>
</div>
""", unsafe_allow_html=True)

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
    workflow.add_node("start_node", lambda state: state)
    analyst_nodes = get_analyst_nodes()

    if selected_analysts is None or not selected_analysts:
        selected_analysts = list(analyst_nodes.keys())

    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

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

            st.markdown("<h3>Trading Decisions</h3>", unsafe_allow_html=True)
            st.json(decisions)

            st.markdown("<h3>Analyst Signals</h3>", unsafe_allow_html=True)
            st.json(analyst_signals)

        except Exception as e:
            st.error(f"Error during simulation: {e}")
        finally:
            progress.stop()
