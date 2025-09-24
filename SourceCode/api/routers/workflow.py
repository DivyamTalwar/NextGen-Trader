"""
Workflow Router - Main hedge fund simulation endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
import sys
import os
import time
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.schemas import (
    WorkflowRequest,
    WorkflowResponse,
    TradingDecision,
    Portfolio,
    Position,
    BaseResponse,
    ResponseStatus,
    AnalystSignal
)
from maain import create_workflow, run_agent_simulation
from graph.state import AgentState
from utils.analysts import get_analyst_nodes, ANALYST_CONFIG
from langchain_core.messages import HumanMessage
from utils.visualize import save_graph_as_png

router = APIRouter()


@router.post("/simulate")
async def run_simulation(request: WorkflowRequest):
    """Run the complete hedge fund simulation with all agents"""
    start_time = time.time()
    
    try:
        # Initialize portfolio
        portfolio = {
            "total_cash": request.initial_capital,
            "margin_requirement": request.margin_requirement,
            "positions": {
                ticker: {
                    "long": 0,
                    "short": 0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0
                } for ticker in request.tickers
            }
        }
        
        # Get selected analysts
        selected_analysts = request.selected_analysts
        if not selected_analysts:
            # Use all analysts if none specified
            selected_analysts = list(get_analyst_nodes().keys())
        
        # Run the simulation
        result = run_agent_simulation(
            tickers=request.tickers,
            portfolio=portfolio,
            start_date=request.start_date,
            end_date=request.end_date,
            selected_analysts=selected_analysts,
            model_choice=request.model_name or "llama-3.3-70b-versatile",
            model_provider=request.model_provider or "groq",
            show_reasoning=request.show_reasoning,
            user_api_key=None
        )
        
        # Format decisions
        formatted_decisions = {}
        for ticker, decision in result["decisions"].items():
            formatted_decisions[ticker] = TradingDecision(
                action=decision.get("action", "hold"),
                quantity=decision.get("quantity", 0),
                confidence=decision.get("confidence", 0),
                reasoning=decision.get("reasoning", "")
            )
        
        # Format analyst signals
        formatted_signals = {}
        for agent_name, signals in result["analyst_signals"].items():
            formatted_signals[agent_name] = {}
            for ticker, signal in signals.items():
                if isinstance(signal, dict):
                    formatted_signals[agent_name][ticker] = AnalystSignal(
                        signal=signal.get("signal", "neutral"),
                        confidence=signal.get("confidence", 0),
                        reasoning=signal.get("reasoning", "")
                    )
        
        # Create portfolio state
        portfolio_state = Portfolio(
            total_cash=portfolio["total_cash"],
            margin_requirement=portfolio["margin_requirement"],
            positions={
                ticker: Position(**pos) for ticker, pos in portfolio["positions"].items()
            }
        )
        
        execution_time = time.time() - start_time
        
        response = WorkflowResponse(
            decisions=formatted_decisions,
            analyst_signals=formatted_signals,
            portfolio_state=portfolio_state,
            execution_time=execution_time
        )
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message="Simulation completed successfully",
            data=response.dict()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during simulation: {str(e)}"
        )


@router.post("/simulate-step")
async def run_simulation_step(request: WorkflowRequest):
    """Run a single step of the simulation (useful for debugging)"""
    try:
        # Initialize portfolio
        portfolio = {
            "total_cash": request.initial_capital,
            "margin_requirement": request.margin_requirement,
            "positions": {
                ticker: {
                    "long": 0,
                    "short": 0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0
                } for ticker in request.tickers
            }
        }
        
        # Create the workflow
        workflow = create_workflow(request.selected_analysts)
        app = workflow.compile()
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content="Analyze tickers for trading decisions")],
            "data": {
                "tickers": request.tickers,
                "portfolio": portfolio,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "analyst_signals": {}
            },
            "metadata": {
                "show_reasoning": request.show_reasoning,
                "model_name": request.model_name or "llama-3.3-70b-versatile",
                "model_provider": request.model_provider or "groq",
                "api_key": None
            }
        }
        
        # Get the execution steps
        steps = []
        for step in app.stream(initial_state):
            step_info = {
                "node": list(step.keys())[0] if step else "unknown",
                "timestamp": datetime.now().isoformat()
            }
            
            # Extract key information from the step
            if step:
                node_name = list(step.keys())[0]
                node_data = step[node_name]
                
                if "data" in node_data and "analyst_signals" in node_data["data"]:
                    step_info["signals"] = node_data["data"]["analyst_signals"]
            
            steps.append(step_info)
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message="Simulation step completed",
            data={
                "steps": steps,
                "total_steps": len(steps)
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during simulation step: {str(e)}"
        )


@router.get("/workflow-graph")
async def get_workflow_graph(analysts: Optional[str] = None):
    """Get the workflow graph visualization"""
    try:
        # Parse analysts if provided
        selected_analysts = None
        if analysts:
            selected_analysts = analysts.split(",")
        
        # Create workflow
        workflow = create_workflow(selected_analysts)
        
        # Generate graph image
        graph_path = "workflow_graph.png"
        save_graph_as_png(workflow, graph_path)
        
        # Get nodes and edges info
        nodes = []
        analyst_nodes = get_analyst_nodes()
        
        if selected_analysts:
            for key in selected_analysts:
                if key in analyst_nodes:
                    nodes.append({
                        "id": analyst_nodes[key][0],
                        "name": ANALYST_CONFIG[key]["name"],
                        "type": "analyst"
                    })
        else:
            for key, (node_name, _) in analyst_nodes.items():
                nodes.append({
                    "id": node_name,
                    "name": ANALYST_CONFIG[key]["name"],
                    "type": "analyst"
                })
        
        # Add system nodes
        nodes.extend([
            {"id": "start_node", "name": "Start", "type": "system"},
            {"id": "risk_management_agent", "name": "Risk Manager", "type": "system"},
            {"id": "portfolio_management_agent", "name": "Portfolio Manager", "type": "system"}
        ])
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message="Workflow graph generated",
            data={
                "nodes": nodes,
                "node_count": len(nodes),
                "graph_image": graph_path
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating workflow graph: {str(e)}"
        )


@router.post("/validate")
async def validate_workflow(request: WorkflowRequest):
    """Validate workflow configuration without running it"""
    try:
        errors = []
        warnings = []
        
        # Validate tickers
        if not request.tickers:
            errors.append("No tickers specified")
        elif len(request.tickers) > 10:
            warnings.append(f"Large number of tickers ({len(request.tickers)}) may slow down processing")
        
        # Validate dates
        try:
            start = datetime.strptime(request.start_date, "%Y-%m-%d")
            end = datetime.strptime(request.end_date, "%Y-%m-%d")
            
            if start >= end:
                errors.append("Start date must be before end date")
            
            days_diff = (end - start).days
            if days_diff > 365:
                warnings.append(f"Large date range ({days_diff} days) may result in long processing time")
            elif days_diff < 7:
                warnings.append(f"Small date range ({days_diff} days) may not provide enough data")
        except ValueError as e:
            errors.append(f"Invalid date format: {str(e)}")
        
        # Validate analysts
        if request.selected_analysts:
            analyst_nodes = get_analyst_nodes()
            for analyst in request.selected_analysts:
                if analyst not in analyst_nodes:
                    errors.append(f"Unknown analyst: {analyst}")
        
        # Validate capital
        if request.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        elif request.initial_capital < 10000:
            warnings.append("Low initial capital may limit trading opportunities")
        
        # Validate model
        valid_models = ["llama-3.3-70b-versatile", "gpt-4o", "claude-3-opus", "gemini-pro"]
        if request.model_name and request.model_name not in valid_models:
            warnings.append(f"Unknown model '{request.model_name}', will use default")
        
        is_valid = len(errors) == 0
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS if is_valid else ResponseStatus.ERROR,
            message="Validation completed" if is_valid else "Validation failed",
            data={
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "configuration": {
                    "tickers": request.tickers,
                    "date_range": f"{request.start_date} to {request.end_date}",
                    "analysts_count": len(request.selected_analysts) if request.selected_analysts else "all",
                    "model": request.model_name or "default"
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error validating workflow: {str(e)}"
        )


@router.get("/presets")
async def get_workflow_presets():
    """Get predefined workflow configurations"""
    presets = [
        {
            "name": "Quick Analysis",
            "description": "Fast analysis with core agents only",
            "config": {
                "tickers": ["AAPL", "MSFT"],
                "selected_analysts": ["fundamentals", "technicals", "valuation"],
                "model_name": "llama-3.3-70b-versatile",
                "show_reasoning": True
            }
        },
        {
            "name": "Full Analysis",
            "description": "Complete analysis with all available agents",
            "config": {
                "tickers": ["AAPL", "GOOGL", "AMZN"],
                "selected_analysts": None,  # Use all
                "model_name": "llama-3.3-70b-versatile",
                "show_reasoning": True
            }
        },
        {
            "name": "Value Investing",
            "description": "Focus on value-oriented analysis",
            "config": {
                "tickers": ["BRK.B", "JPM", "WMT"],
                "selected_analysts": ["warren_buffett", "ben_graham", "charlie_munger", "fundamentals", "valuation"],
                "model_name": "llama-3.3-70b-versatile",
                "show_reasoning": True
            }
        },
        {
            "name": "Growth & Innovation",
            "description": "Focus on growth and disruptive companies",
            "config": {
                "tickers": ["TSLA", "NVDA", "PLTR"],
                "selected_analysts": ["cathie_wood", "phil_fisher", "technicals", "sentiment"],
                "model_name": "llama-3.3-70b-versatile",
                "show_reasoning": True
            }
        },
        {
            "name": "Risk-Averse",
            "description": "Conservative approach with focus on risk management",
            "config": {
                "tickers": ["JNJ", "PG", "KO"],
                "selected_analysts": ["risk_manager", "fundamentals", "valuation", "ben_graham"],
                "model_name": "llama-3.3-70b-versatile",
                "show_reasoning": True,
                "margin_requirement": 0.5
            }
        }
    ]
    
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message="Retrieved workflow presets",
        data={"presets": presets}
    )