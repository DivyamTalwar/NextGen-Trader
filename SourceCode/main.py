import streamlit as st
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from langchain_core.messages import HumanMessage
from graph.state import AgentState
from langgraph.graph import END, StateGraph
from agents.portfolio_manager import portfolio_management_agent
from agents.risk_manager import risk_management_agent
from utils.analysts import ANALYST_ORDER, get_analyst_nodes
from utils.progress import progress
from llm.models import LLM_ORDER, get_model_info
from utils.visualize import save_graph_as_png
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px


load_dotenv()


# Custom CSS for enhanced UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    /* General styling */
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
        padding: 30px 0;
        margin-bottom: 40px;
        font-weight: 700;
        letter-spacing: 1px;
        text-shadow: 3px 3px 5px #000000;
        animation: fadeIn 2s ease-in-out;
    }

    h3 {
        color: #f0ab51;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 15px;
        border-bottom: 2px solid #f0ab51;
        padding-bottom: 10px;
    }

    .stButton>button {
        color: #ffffff;
        background-color: #f0ab51;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: 700;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    .stButton>button:hover {
        background-color: #d48b36;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.4);
    }

    .stTextInput>div>div>input {
        background-color: #1e2532;
        color: #ffffff;
        border-radius: 8px;
        border: 1px solid #4d596b;
        padding: 12px;
        font-size: 16px;
        transition: border-color 0.3s ease;
    }

    .stNumberInput>div>div>div>input {
        background-color: #1e2532;
        color: #ffffff;
        border-radius: 8px;
        border: 1px solid #4d596b;
        padding: 12px;
        font-size: 16px;
    }

    .stMultiSelect>div>div>div {
        background-color: #1e2532;
        color: #ffffff;
        border-radius: 8px;
        border: 1px solid #4d596b;
        padding: 12px;
        font-size: 16px;
    }

    .stSelectbox>div>div>div {
        background-color: #1e2532;
        color: #ffffff;
        border-radius: 8px;
        border: 1px solid #4d596b;
        padding: 12px;
        font-size: 16px;
    }

    .stCheckbox>label {
        color: #ffffff;
        font-size: 16px;
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
        border-radius: 8px;
        margin-bottom: 20px;
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
        padding: 18px;
        margin-bottom: 25px;
        border-radius: 8px;
        font-size: 16px;
    }

    .stJson {
        background-color: #1e2532;
        color: #ffffff;
        border-radius: 8px;
        padding: 25px;
        margin-bottom: 30px;
        overflow-x: auto;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .fade-in {
        animation: fadeIn 2s ease-in-out;
    }

    @keyframes slideIn {
        from { transform: translateY(-50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    .slide-in {
        animation: slideIn 1s ease-out;
    }

    /* Parameter Card */
    .parameter-card {
        background-color: rgba(30, 37, 50, 0.8);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 30px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .parameter-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.5);
    }

    .parameter-card h3 {
        color: #f0ab51;
        border-bottom: 3px solid #f0ab51;
        padding-bottom: 12px;
        margin-bottom: 20px;
        font-size: 24px;
    }

    .parameter-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
        font-size: 18px;
    }

    .parameter-label {
        color: #ffffff;
        font-weight: bold;
    }

    .parameter-value {
        color: #bbbbbb;
    }

    /* Trading Decisions and Analyst Signals */
    .output-section {
        background-color: rgba(30, 37, 50, 0.8);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 30px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.4);
        animation: slideIn 1s ease-out;
    }

    .output-section h3 {
        color: #f0ab51;
        border-bottom: 3px solid #f0ab51;
        padding-bottom: 12px;
        margin-bottom: 20px;
        font-size: 24px;
    }

    /* Plotly Chart Styling */
    .plotly-chart {
        margin-bottom: 20px;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* Recommendation Animation */
    .recommendation {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        animation: recommend 1s infinite alternate;
    }

    .recommendation.buy {
        background-color: #28a745;
        color: white;
    }

    .recommendation.sell {
        background-color: #dc3545;
        color: white;
    }

    @keyframes recommend {
        0% { transform: scale(1); }
        100% { transform: scale(1.1); }
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
    selected_analysts = st.multiselect("Select AI Analysts", options=analyst_choices, default=["warren_buffett", "ben_graham", "rakesh_jhunjhunwala", "sentiment_analyst"])

    model_choice_name = st.selectbox("Select LLM Model", options=[model[1] for model in LLM_ORDER], index=0)
    
    # Find the selected model's info from LLM_ORDER
    selected_model = next((model for model in LLM_ORDER if model[1] == model_choice_name), None)
    
    if selected_model:
        model_choice_display, model_choice, model_provider_str = selected_model
        model_info = get_model_info(model_choice, model_provider_str)
        model_provider = model_info.provider.value if model_info else "Unknown"
    else:
        model_choice = ""
        model_provider = "Unknown"
        st.error("Could not find the selected model information.")

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
                <span class="parameter-value">{model_choice_display} ({model_provider})</span>
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
    "total_cash": initial_cash,
    "margin_requirement": margin_requirement,
    "positions": {
        ticker: {
            "long": 0,
            "short": 0,
            "long_cost_basis": 0.0,
            "short_cost_basis": 0.0,
        } for ticker in tickers
    },
}

def run_hedge_fund(
    tickers,
    portfolio,
    start_date,
    end_date,
    selected_analysts,
    show_reasoning,
    model_choice,
    model_provider,
    show_agent_graph,
):
    """
    Runs the hedge fund simulation.
    """
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
        st.image(file_path, caption="Agent Graph", use_container_width=True)

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

            st.markdown("<div class='output-section'>", unsafe_allow_html=True)
            st.markdown("<h3>Trading Decisions</h3>", unsafe_allow_html=True)
            st.json(decisions)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='output-section'>", unsafe_allow_html=True)
            st.markdown("<h3>Analyst Signals</h3>", unsafe_allow_html=True)
            st.json(analyst_signals)
            st.markdown("</div>", unsafe_allow_html=True)

            for ticker, decision in decisions.items():
                action = decision.get("action")
                if action in ["buy", "sell"]:
                    display_recommendation(ticker, action)

        except Exception as e:
            st.error(f"Error during simulation: {e}")
        finally:
            progress.stop()

def display_recommendation(ticker, action):
    if action == "buy":
        st.markdown(f"""
        <div class="recommendation buy">
            BUY {ticker}! 🚀
        </div>
        """, unsafe_allow_html=True)
    elif action == "sell":
        st.markdown(f"""
        <div class="recommendation sell">
            SELL {ticker}! 📉
        </div>
        """, unsafe_allow_html=True)

if st.button("Run Hedge Fund Simulation"):
    run_hedge_fund(
        tickers,
        portfolio,
        start_date,
        end_date,
        selected_analysts,
        show_reasoning,
        model_choice,
        model_provider,
        show_agent_graph,
    )
