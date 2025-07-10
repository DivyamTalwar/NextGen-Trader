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

# Page configuration
st.set_page_config(
    page_title="AI-Powered Hedge Fund Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a professional and modern UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@800&family=Chakra+Petch:wght@500&family=Rajdhani:wght@700&family=Inter:wght@500&display=swap');

    /* --- 1. Design Tokens --- */
    :root {
        --bg-dark: #0F0F0F;
        --neon-cyan: #00FFFF;
        --electric-blue: #1E88E5;
        --amber-warning: #FFC107;
        --emerald-success: #43A047;
        --ruby-error: #E53935;
        --text-primary: #F5F5F5;
        --text-secondary: #BDBDBD;
        --font-headline: 'Orbitron', sans-serif;
        --font-body: 'Chakra Petch', sans-serif;
        --spacing-unit: 8px;
        --border-radius: 14px;
    }

    /* --- Base & Typography --- */
    body {
        font-family: var(--font-body);
        background-color: var(--bg-dark);
        color: var(--text-primary);
        font-size: 17px;
        line-height: 1.6;
    }
    .stApp {
        background-color: var(--bg-dark);
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-headline);
        text-transform: uppercase;
        font-weight: 800;
    }
    h3 {
        font-size: 28px;
        color: var(--electric-blue);
        border-bottom: 2px solid rgba(0, 255, 255, 0.3);
        padding-bottom: calc(var(--spacing-unit) * 1.5);
        margin-bottom: calc(var(--spacing-unit) * 3);
    }

    /* --- 2. Header & Branding --- */
    .header-container {
        padding: 20px;
        margin-bottom: 30px;
        border-radius: 20px;
        background: radial-gradient(ellipse at center, rgba(30, 136, 229, 0.15), rgba(15, 15, 15, 0) 70%);
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.1), inset 0 0 15px rgba(30, 136, 229, 0.2);
        animation: header-pulse 5s infinite alternate;
    }
    @keyframes header-pulse {
        from { box-shadow: 0 0 30px rgba(0, 255, 255, 0.1), inset 0 0 15px rgba(30, 136, 229, 0.2); }
        to { box-shadow: 0 0 45px rgba(0, 255, 255, 0.3), inset 0 0 25px rgba(30, 136, 229, 0.4); }
    }
    #neon-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 42px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, var(--neon-cyan), var(--electric-blue), var(--amber-warning));
        -webkit-background-clip: text;
        color: transparent;
        text-shadow:
            0 0 4px rgba(255, 255, 255, 0.3),
            0 0 8px var(--neon-cyan),
            0 0 18px var(--neon-cyan),
            0 0 28px var(--electric-blue),
            0 0 40px var(--electric-blue),
            0 0 60px var(--amber-warning),
            0 0 80px var(--amber-warning);
        animation: pulse-glow 3s infinite alternate;
    }
    @keyframes pulse-glow {
        from {
            text-shadow:
                0 0 4px rgba(255, 255, 255, 0.3),
                0 0 8px var(--neon-cyan),
                0 0 18px var(--neon-cyan),
                0 0 28px var(--electric-blue),
                0 0 40px var(--electric-blue),
                0 0 60px var(--amber-warning),
                0 0 80px var(--amber-warning);
        }
        to {
            text-shadow:
                0 0 6px rgba(255, 255, 255, 0.5),
                0 0 12px var(--neon-cyan),
                0 0 22px var(--neon-cyan),
                0 0 35px var(--electric-blue),
                0 0 50px var(--electric-blue),
                0 0 75px var(--amber-warning),
                0 0 100px var(--amber-warning);
        }
    }

    /* --- 3. Layout & Cards --- */
    .card {
        background: rgba(20, 20, 22, 0.6);
        backdrop-filter: blur(12px);
        border-radius: var(--border-radius);
        padding: calc(var(--spacing-unit) * 3);
        margin-bottom: calc(var(--spacing-unit) * 2);
        position: relative;
        overflow: hidden;
        box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: card-fade-in 0.5s ease-out forwards;
        opacity: 0;
    }
    .card:before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        border-radius: var(--border-radius);
        border: 3px solid transparent;
        background: linear-gradient(135deg, var(--neon-cyan), var(--electric-blue), var(--amber-warning)) border-box;
        -webkit-mask:
            linear-gradient(#fff 0 0) padding-box,
            linear-gradient(#fff 0 0);
        -webkit-mask-composite: destination-out;
        mask-composite: exclude;
    }
    .card.animated-border:before {
        animation: gradient-spin 5s linear infinite;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0px 12px 24px rgba(0, 0, 0, 0.5);
    }
    @keyframes gradient-spin {
        100% { transform: rotate(360deg); }
    }
    @keyframes card-fade-in {
        from { opacity: 0; transform: translateY(30px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* --- 4. Strategic Directives Panel --- */
    .card-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: var(--spacing-unit);
        font-size: 1rem;
    }
    .card-label { color: var(--text-secondary); }
    .card-value { color: var(--text-primary); font-weight: bold; }

    /* --- 5. Trade Execution Protocol --- */
    .badge-sign {
        display: inline-block;
        padding: var(--spacing-unit) calc(var(--spacing-unit) * 2);
        font-size: 1.2em;
        font-weight: bold;
        text-transform: uppercase;
        border-radius: 20px;
        transition: all 0.3s ease;
    }
    .badge-sign.long {
        background: var(--emerald-success);
        box-shadow: 0 0 12px var(--emerald-success);
    }
    .badge-sign.short {
        background: var(--ruby-error);
        box-shadow: 0 0 12px var(--ruby-error);
    }
    .trade-protocol-card {
        background: rgba(30, 136, 229, 0.1); /* Electric blue tint */
        border: 3px solid var(--electric-blue);
        box-shadow: 0 0 25px rgba(30, 136, 229, 0.5);
    }
    .trade-protocol-card .quantity-value {
        color: var(--neon-cyan);
        text-shadow: 0 0 8px var(--neon-cyan);
    }
    .badge-sign:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px currentColor;
    }
    .quantity-value { font-size: 2em; font-weight: bold; }
    .confidence-bar {
        height: 12px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid rgba(0, 255, 255, 0.3);
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--electric-blue), var(--amber-warning));
        box-shadow: 0 0 8px var(--amber-warning);
        animation: fill-bar 1.5s ease-out forwards;
    }
    @keyframes fill-bar {
        from { width: 0%; }
        to { width: var(--confidence-percent, 0%); }
    }
    .reasoning-text {
        background: rgba(0, 0, 0, 0.3);
        padding: calc(var(--spacing-unit) * 2);
        border-radius: var(--spacing-unit);
        color: var(--amber-warning);
        text-shadow: 0 0 5px var(--amber-warning);
        line-height: 1.7;
        max-height: 200px;
        overflow-y: auto;
    }
    details > summary {
        list-style: none;
        color: var(--electric-blue);
        cursor: pointer;
        font-family: var(--font-headline);
        padding: var(--spacing-unit);
        border-radius: var(--spacing-unit);
        transition: background 0.3s ease;
    }
    details > summary::-webkit-details-marker {
        display: none;
    }
    details > summary:hover {
        background: rgba(30, 136, 229, 0.2);
    }

    /* --- 6. Analyst Intelligence Matrix --- */
    .analyst-card h5 {
        color: var(--neon-cyan);
        text-shadow: 0 0 5px var(--neon-cyan);
    }
    .signal-line { font-size: 1.1em; font-weight: bold; }
    .signal-line.bullish { color: var(--emerald-success); text-shadow: 0 0 8px var(--emerald-success); }
    .signal-line.bearish { color: var(--ruby-error); text-shadow: 0 0 8px var(--ruby-error); }
    .signal-line.neutral { color: var(--text-secondary); }

    /* --- 7. Sidebar & Controls --- */
    .stSidebar {
        background: rgba(15, 15, 15, 0.5);
        backdrop-filter: blur(20px);
        border-right: 2px solid var(--neon-cyan);
        box-shadow: 0 0 20px var(--neon-cyan);
    }
    .model-display, .model-display strong {
        color: var(--text-primary) !important;
    }
    .stTextInput input, .stNumberInput input, .stSelectbox > div > div {
        background: rgba(0,0,0,0.4);
        border: 2px solid var(--electric-blue);
        box-shadow: 0 0 8px var(--electric-blue) inset;
        border-radius: var(--border-radius);
        padding: calc(var(--spacing-unit) * 1.5);
        font-size: 1.1em;
    }
    /* Ensure text inside selectbox is visible and has enough space */
    .stSelectbox [data-baseweb="select"] > div {
        color: var(--text-primary) !important;
        width: 100%;
    }
    .stSelectbox {
        width: 100% !important;
    }
    .stMultiSelect .stChip {
        background: var(--neon-cyan);
        color: var(--bg-dark);
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.2s ease;
    }
    .stMultiSelect .stChip:hover {
        box-shadow: 0 0 10px var(--neon-cyan);
        transform: scale(1.05);
    }
    .stButton>button {
        background: var(--electric-blue);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: calc(var(--spacing-unit) * 1.5) calc(var(--spacing-unit) * 3);
        font-family: var(--font-headline);
        text-transform: uppercase;
        box-shadow: 0 0 12px var(--electric-blue);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px var(--electric-blue);
    }

    /* --- 9. Accessibility & Responsiveness --- */
    @media (max-width: 768px) {
        .stApp {
            padding: var(--spacing-unit);
        }
        h3 { font-size: 24px; }
        .card { padding: calc(var(--spacing-unit) * 2); }
        .stSidebar {
            /* On mobile, Streamlit hides the sidebar behind a hamburger menu */
            box-shadow: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- App Header ---
st.markdown('<div class="header-container"><h1 id="neon-title">NextGen Traders – Where Algorithms Meet Alpha</h1></div>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ Configuration")

    with st.expander("💵 Capital & Ticker", expanded=True):
        initial_cash = st.number_input("Initial Cash", value=100000.0, step=1000.0, format="%.2f")
        margin_requirement = st.number_input("Margin Requirement", value=0.20, step=0.1, format="%.2f")
        tickers_input = st.text_input("Tickers (comma-separated)", value="AAPL")

    with st.expander("🗓️ Dates", expanded=True):
        start_date_input = st.text_input("Start Date (YYYY-MM-DD)", value="2025-01-01")
        end_date_input = st.text_input("End Date (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"))

    with st.expander("🧠 Analysis Engines", expanded=True):
        analyst_choices = {display: value for display, value in ANALYST_ORDER}
        selected_analysts_display = st.multiselect(
            "Select AI Analysts",
            options=list(analyst_choices.keys()),
            default=["Warren Buffett", "Sentiment Analyst"],
        )
        selected_analysts = [analyst_choices[display] for display in selected_analysts_display]

        use_custom_api = st.checkbox("Use Custom API Key & Model", value=False)
        
        user_api_key = None
        model_choice = None
        model_provider = None
        model_choice_display = None

        if use_custom_api:
            st.write("---")
            user_api_key = st.text_input("Enter your API Key", type="password", help="Your API key will not be stored.")
            model_provider = st.selectbox(
                "Select Provider",
                options=["OpenAI", "Anthropic", "Gemini"],
                index=0,
                key="provider_select"
            )
            
            if model_provider == "OpenAI":
                model_suggestions = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "Other..."]
            elif model_provider == "Anthropic":
                model_suggestions = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307", "Other..."]
            else: # Gemini
                model_suggestions = ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.0-pro", "Other..."]

            selected_model_option = st.selectbox("Select Model", options=model_suggestions, index=0)
            
            if selected_model_option == "Other...":
                model_choice = st.text_input("Enter Custom Model Name", key="custom_model_name")
            else:
                model_choice = selected_model_option
            
            model_choice_display = model_choice
            st.write("---")
        else:
            model_options = {model[0]: (model[1], model[2]) for model in LLM_ORDER}
            model_display_name = st.selectbox(
                "Select LLM Model",
                options=list(model_options.keys()),
                index=7,  # Default to llama-3.3-70b-versatile
            )
            model_choice, model_provider = model_options[model_display_name]
            model_choice_display = model_display_name

    with st.expander("🖥️ Display Options", expanded=False):
        show_reasoning = st.checkbox("Show Reasoning", value=True)
        show_agent_graph = st.checkbox("Show Agent Graph", value=False)

    st.markdown(f'<div class="model-display">**Selected Model:** {model_choice_display} ({model_provider})</div>', unsafe_allow_html=True)

# --- Main Content ---
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

# Date validation
try:
    end_date = datetime.strptime(end_date_input, "%Y-%m-%d").strftime("%Y-%m-%d") if end_date_input else datetime.now().strftime("%Y-%m-%d")
    start_date = (
        datetime.strptime(start_date_input, "%Y-%m-%d").strftime("%Y-%m-%d")
        if start_date_input
        else (datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=3)).strftime("%Y-%m-%d")
    )
except ValueError:
    st.error("Invalid date format. Please use YYYY-MM-DD.")
    st.stop()

# --- Simulation Parameters Display ---
st.markdown("<h3>Strategic Directives</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

# Calculate elapsed days for the mini-KPI
try:
    elapsed_days = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
except ValueError:
    elapsed_days = "N/A"

with col1:
    st.markdown(
        f"""
        <div class="card animated-border" style="animation-delay: 0.1s;">
            <div class="card-item"><span class="card-label">Target Assets</span> <span class="card-value">{', '.join(tickers)}</span></div>
            <div class="card-item"><span class="card-label">Initial Capital</span> <span class="card-value">${initial_cash:,.2f}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="card animated-border" style="animation-delay: 0.2s;">
            <div class="card-item"><span class="card-label">Date Range</span> <span class="card-value">{start_date} to {end_date}</span></div>
            <div class="card-item"><span class="card-label">Elapsed</span> <span class="card-value">{elapsed_days} days</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="card animated-border" style="animation-delay: 0.3s;">
            <div class="card-item"><span class="card-label">Cognitive Engine</span> <span class="card-value">{model_choice_display}</span></div>
            <div class="card-item"><span class="card-label">Analyst Swarm</span> <span class="card-value">{len(selected_analysts)} selected</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Portfolio Initialization ---
portfolio = {
    "total_cash": initial_cash,
    "margin_requirement": margin_requirement,
    "positions": {
        ticker: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0} for ticker in tickers
    },
}

def create_workflow(selected_analysts=None):
    """Creates the agent workflow graph."""
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


def run_agent_simulation(
    tickers,
    portfolio,
    start_date,
    end_date,
    selected_analysts,
    model_choice,
    model_provider,
    show_reasoning=True,
    user_api_key=None,
):
    """Compiles and runs the agent graph, returning the final decisions."""
    workflow = create_workflow(selected_analysts)
    app = workflow.compile()

    # Enhanced prompt to ensure detailed reasoning as requested.
    prompt = (
        "Make trading decisions based on the provided data. For each ticker, provide a detailed, "
        "in-depth reasoning of at least 4 lines, explaining the key factors and analysis that led to the final action. "
        "Go beyond surface-level comments and justify the trade with specific data points or analyst signals. "
        "The final output must be a single JSON object where each key is a ticker. The value for each ticker should be another JSON object "
        "with 'action' (buy, sell, short, cover, hold), 'quantity' (integer), 'confidence' (float from 0-100), and 'reasoning' (the detailed string)."
    )

    final_state = app.invoke({
        "messages": [HumanMessage(content=prompt)],
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
            "api_key": user_api_key,
        },
    })

    decisions = json.loads(final_state["messages"][-1].content)
    analyst_signals = final_state["data"]["analyst_signals"]

    return {"decisions": decisions, "analyst_signals": analyst_signals}


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
    user_api_key=None,
):
    """Runs the hedge fund simulation and displays results in Streamlit."""
    if show_agent_graph:
        workflow = create_workflow(selected_analysts)
        app = workflow.compile()
        file_path = "agent_graph.png"
        save_graph_as_png(app, file_path)
        st.image(file_path, caption="Agent Graph", width=800)

    with st.spinner("Running simulation... This may take a moment."):
        progress.start()
        try:
            output = run_agent_simulation(
                tickers=tickers,
                portfolio=portfolio,
                start_date=start_date,
                end_date=end_date,
                selected_analysts=selected_analysts,
                model_choice=model_choice,
                model_provider=model_provider,
                show_reasoning=show_reasoning,
                user_api_key=user_api_key,
            )
            decisions = output["decisions"]
            analyst_signals = output["analyst_signals"]

            display_results(decisions, analyst_signals)

        except Exception as e:
            st.error(f"An error occurred during the simulation: {e}")
        finally:
            progress.stop()

def display_results(decisions, analyst_signals):
    """Displays the simulation results in a structured and detailed format."""
    st.markdown("<h3>Trade Execution Protocol</h3>", unsafe_allow_html=True)

    for ticker, decision in decisions.items():
        action = decision.get("action", "hold").upper()
        confidence = decision.get('confidence', 0)
        quantity = decision.get('quantity', 0)
        reasoning = decision.get('reasoning', 'No reasoning provided.')

        action_class = "long" if action in ["BUY", "COVER"] else "short" if action in ["SELL", "SHORT"] else "neutral"

        with st.container():
            st.markdown(
                f"""
                <div class="card trade-protocol-card" style="animation-delay: 0.4s; margin-bottom: 0;">
                    <h4 style="text-align: center;">{ticker}</h4>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin: 20px 0;">
                        <div style="text-align: left;">
                            <span class="card-label">Quantity</span>
                            <div class="quantity-value">{quantity}</div>
                        </div>
                        <div style="text-align: right;">
                            <span class="card-label">Final Action</span>
                            <div><span class="badge-sign {action_class}">{action}</span></div>
                        </div>
                    </div>
                    <div class="card-item">
                        <span class="card-label">Confidence</span>
                        <span class="card-value">{confidence:.1f}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="--confidence-percent: {confidence}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # The expander is now open by default to show the detailed reasoning immediately.
            with st.expander("✅ View Reasoning", expanded=True):
                st.markdown(f'<div class="reasoning-text" style="background: transparent; border: none; padding: 0;">{reasoning}</div>', unsafe_allow_html=True)

        st.markdown("<h4 style='margin-top: 40px;'>Analyst Intelligence Matrix</h4>", unsafe_allow_html=True)
        
        ticker_analyst_signals = []
        for agent_name, signals in analyst_signals.items():
            if isinstance(signals, dict) and ticker in signals:
                ticker_analyst_signals.append((agent_name, signals[ticker]))

        # Create a two-column layout for the analyst cards
        col1, col2 = st.columns(2)
        columns = [col1, col2]

        for i, (agent_name, signal_data) in enumerate(ticker_analyst_signals):
            with columns[i % 2]:
                analyst_display_name = agent_name.replace("_agent", "").replace("_", " ").title()
                signal = signal_data.get("signal", "N/A").lower()
                confidence = signal_data.get("confidence", 0)
                reasoning = signal_data.get("reasoning", "No reasoning provided.")

                icon = "🔼" if signal == "bullish" else "🔽" if signal == "bearish" else "➖"
                
                with st.container():
                    st.markdown(
                        f"""
                        <div class="card analyst-card" style="animation-delay: {0.5 + i*0.1}s; margin-bottom: 0;">
                            <h5>{analyst_display_name}</h5>
                            <div class="card-item">
                                <span class="card-label">Signal</span>
                                <span class="signal-line {signal}">{icon} {signal.capitalize()}</span>
                            </div>
                            <div class="card-item">
                                <span class="card-label">Confidence</span>
                                <span class="card-value">{confidence:.1f}%</span>
                            </div>
                            <div class="confidence-bar" style="height: 8px;">
                               <div class="confidence-fill" style="--confidence-percent: {confidence}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    with st.expander("✅ Full Reasoning"):
                        if isinstance(reasoning, dict):
                            st.json(reasoning)
                        else:
                            st.markdown(f'<div class="reasoning-text" style="background: transparent; border: none; padding: 0;">{reasoning}</div>', unsafe_allow_html=True)

def display_recommendation(ticker, action):
    if action == "buy":
        st.markdown(f'<div class="recommendation buy">BUY {ticker}! 🚀</div>', unsafe_allow_html=True)
    elif action == "sell":
        st.markdown(f'<div class="recommendation sell">SELL {ticker}! 📉</div>', unsafe_allow_html=True)

if st.button("Run Hedge Fund Simulation"):
    if not tickers:
        st.warning("Please enter at least one ticker.")
    else:
        run_hedge_fund(
            tickers=tickers,
            portfolio=portfolio,
            start_date=start_date,
            end_date=end_date,
            selected_analysts=selected_analysts,
            show_reasoning=show_reasoning,
            model_choice=model_choice,
            model_provider=model_provider,
            show_agent_graph=show_agent_graph,
            user_api_key=user_api_key,
        )
