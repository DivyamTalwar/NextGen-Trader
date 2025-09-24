"""
Agents Router - Endpoints for individual agent analysis
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import time
import sys
import os

# Add parent directory to path to import SourceCode modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.schemas import (
    AgentAnalysisRequest,
    AgentAnalysisResponse,
    BaseResponse,
    ResponseStatus,
    AnalystSignal
)
from graph.state import AgentState
from utils.analysts import get_analyst_nodes, ANALYST_CONFIG
from langchain_core.messages import HumanMessage
import json

router = APIRouter()


def run_single_agent(agent_name: str, agent_func, state: AgentState) -> Dict[str, Any]:
    """Run a single agent and return its analysis"""
    try:
        # Execute the agent function
        result = agent_func(state)
        
        # Extract signals from the result
        if "data" in result and "analyst_signals" in result["data"]:
            signals = result["data"]["analyst_signals"].get(agent_name, {})
        else:
            signals = {}
        
        return signals
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running agent {agent_name}: {str(e)}"
        )


@router.get("/list")
async def list_agents():
    """Get list of all available agents with their descriptions"""
    try:
        agents_info = []
        for key, config in ANALYST_CONFIG.items():
            agents_info.append({
                "key": key,
                "name": config.get("display_name", key),
                "emoji": config.get("emoji", "📊"),
                "description": config.get("description", "Investment analysis agent")
            })
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message="Successfully retrieved agent list",
            data={"agents": agents_info, "total": len(agents_info)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/{agent_key}")
async def analyze_with_agent(agent_key: str, request: AgentAnalysisRequest):
    """Run analysis with a specific agent"""
    start_time = time.time()
    
    # Validate agent exists
    if agent_key not in ANALYST_CONFIG:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_key}' not found. Use /api/agents/list to see available agents."
        )
    
    try:
        # Get the agent function
        analyst_nodes = get_analyst_nodes()
        if agent_key not in analyst_nodes:
            raise HTTPException(
                status_code=404,
                detail=f"Agent function for '{agent_key}' not found"
            )
        
        agent_name, agent_func = analyst_nodes[agent_key]
        
        # Prepare the state for the agent
        portfolio = {
            "total_cash": 100000,
            "margin_requirement": 0.2,
            "positions": {
                ticker: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0}
                for ticker in request.tickers
            }
        }
        
        state = {
            "messages": [HumanMessage(content="Analyze the provided tickers")],
            "data": {
                "tickers": request.tickers,
                "portfolio": portfolio,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "analyst_signals": {}
            },
            "metadata": {
                "show_reasoning": True,
                "model_name": request.model_name or "llama-3.3-70b-versatile",
                "model_provider": request.model_provider or "groq",
                "api_key": None
            }
        }
        
        # Run the agent
        signals = run_single_agent(agent_name, agent_func, state)
        
        # Format the response
        formatted_signals = {}
        for ticker, signal_data in signals.items():
            if isinstance(signal_data, dict):
                formatted_signals[ticker] = AnalystSignal(
                    signal=signal_data.get("signal", "neutral"),
                    confidence=signal_data.get("confidence", 0),
                    reasoning=signal_data.get("reasoning", "")
                )
        
        execution_time = time.time() - start_time
        
        response = AgentAnalysisResponse(
            agent_name=ANALYST_CONFIG[agent_key].get("display_name", agent_key),
            analysis_date=request.end_date,
            signals=formatted_signals,
            execution_time=execution_time
        )
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Analysis completed by {ANALYST_CONFIG[agent_key].get('display_name', agent_key)}",
            data=response.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during agent analysis: {str(e)}"
        )


@router.post("/batch-analyze")
async def batch_analyze(request: AgentAnalysisRequest):
    """Run analysis with multiple agents in parallel"""
    start_time = time.time()
    
    try:
        # Get selected agents
        selected_agents = request.agent_name.split(",") if "," in request.agent_name else [request.agent_name]
        
        # Validate all agents exist
        analyst_nodes = get_analyst_nodes()
        for agent_key in selected_agents:
            if agent_key not in ANALYST_CONFIG:
                raise HTTPException(
                    status_code=404,
                    detail=f"Agent '{agent_key}' not found"
                )
        
        # Prepare the base state
        portfolio = {
            "total_cash": 100000,
            "margin_requirement": 0.2,
            "positions": {
                ticker: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0}
                for ticker in request.tickers
            }
        }
        
        state = {
            "messages": [HumanMessage(content="Analyze the provided tickers")],
            "data": {
                "tickers": request.tickers,
                "portfolio": portfolio,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "analyst_signals": {}
            },
            "metadata": {
                "show_reasoning": True,
                "model_name": request.model_name or "llama-3.3-70b-versatile",
                "model_provider": request.model_provider or "groq",
                "api_key": None
            }
        }
        
        # Run all agents and collect results
        all_signals = {}
        for agent_key in selected_agents:
            agent_name, agent_func = analyst_nodes[agent_key]
            signals = run_single_agent(agent_name, agent_func, state)
            all_signals[ANALYST_CONFIG[agent_key].get("display_name", agent_key)] = signals
        
        execution_time = time.time() - start_time
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Batch analysis completed with {len(selected_agents)} agents",
            data={
                "agents_used": [ANALYST_CONFIG[key].get("display_name", key) for key in selected_agents],
                "analysis_date": request.end_date,
                "signals": all_signals,
                "execution_time": execution_time
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during batch analysis: {str(e)}"
        )


@router.get("/agent/{agent_key}/info")
async def get_agent_info(agent_key: str):
    """Get detailed information about a specific agent"""
    if agent_key not in ANALYST_CONFIG:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_key}' not found"
        )
    
    config = ANALYST_CONFIG[agent_key]
    
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message=f"Retrieved info for {config.get('display_name', agent_key)}",
        data={
            "key": agent_key,
            "name": config.get("display_name", agent_key),
            "emoji": config.get("emoji", "📊"),
            "description": config.get("description", "Investment analysis agent"),
            "philosophy": config.get("philosophy", "Data-driven investment analysis"),
            "metrics_focus": config.get("metrics_focus", ["general analysis"])
        }
    )