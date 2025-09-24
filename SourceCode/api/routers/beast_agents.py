from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Import all beast agents
from agents.warren_buffett import beast_warren_buffett_agent
from agents.quantum_market_dynamics import quantum_market_dynamics_agent  
from agents.neural_sentiment_predictor import neural_sentiment_predictor_agent

# Import enhanced existing agents
from agents.warren_buffett import buffett_agent
from agents.phil_fisher import phil_fisher_agent
from agents.ben_graham import ben_graham_agent
from agents.aswath_damodaran import aswath_damodaran_agent as damodaran_agent
from agents.bill_ackman import bill_ackman_agent
from agents.cathie_wood import cathie_wood_agent
from agents.charlie_munger import charlie_munger_agent
from agents.stanley_druckenmiller import stanley_druckenmiller_agent
from agents.rakesh_jhunjhunwala import rakesh_jhunjhunwala_agent
from agents.fundamentals import fundamentals_analyst_agent
from agents.valuation import valuation_analyst_agent
from agents.technicals import technical_analyst_agent
from agents.sentiment import sentiment_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.portfolio_manager import portfolio_management_agent

# Create router
router = APIRouter()

# Request/Response Models
class AgentRequest(BaseModel):
    """Request model for agent analysis"""
    tickers: List[str] = Field(..., description="List of stock tickers to analyze")
    start_date: str = Field(default=None, description="Start date for analysis (YYYY-MM-DD)")
    end_date: str = Field(default=None, description="End date for analysis (YYYY-MM-DD)")
    portfolio: Optional[Dict[str, Any]] = Field(default=None, description="Current portfolio state")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")
    
class BeastAgentRequest(AgentRequest):
    """Enhanced request for beast agents"""
    analysis_depth: str = Field(default="maximum", description="Analysis depth: quick, standard, deep, maximum")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    include_contrarian: bool = Field(default=True, description="Include contrarian analysis")
    risk_tolerance: str = Field(default="moderate", description="Risk tolerance: conservative, moderate, aggressive")

class AgentResponse(BaseModel):
    """Response model for agent analysis"""
    status: str
    agent: str
    analysis: Dict[str, Any]
    execution_time: float
    timestamp: str
    
class MultiAgentResponse(BaseModel):
    """Response for multiple agent analysis"""
    status: str
    agents_completed: List[str]
    agents_failed: List[str]
    combined_analysis: Dict[str, Any]
    execution_time: float
    timestamp: str

# Agent registry
AGENT_REGISTRY = {
    # Beast Mode Agents
    "beast_warren_buffett": beast_warren_buffett_agent,
    "quantum_market_dynamics": quantum_market_dynamics_agent,
    "neural_sentiment_predictor": neural_sentiment_predictor_agent,
    
    # Classic Agents (Enhanced)
    "warren_buffett": buffett_agent,
    "phil_fisher": phil_fisher_agent,
    "ben_graham": ben_graham_agent,
    "aswath_damodaran": damodaran_agent,
    "bill_ackman": bill_ackman_agent,
    "cathie_wood": cathie_wood_agent,
    "charlie_munger": charlie_munger_agent,
    "stanley_druckenmiller": stanley_druckenmiller_agent,
    "rakesh_jhunjhunwala": rakesh_jhunjhunwala_agent,
    
    # Analysis Agents
    "fundamentals": fundamentals_analyst_agent,
    "valuation": valuation_analyst_agent,
    "technicals": technical_analyst_agent,
    "sentiment": sentiment_analyst_agent,
    "risk_manager": risk_management_agent,
    "portfolio_manager": portfolio_management_agent
}

BEAST_AGENTS = [
    "beast_warren_buffett",
    "quantum_market_dynamics", 
    "neural_sentiment_predictor"
]

def create_agent_state(request: AgentRequest) -> Dict[str, Any]:
    """Create agent state from request"""
    
    # Default dates if not provided
    if not request.end_date:
        request.end_date = datetime.now().strftime("%Y-%m-%d")
    if not request.start_date:
        request.start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Default portfolio if not provided
    if not request.portfolio:
        request.portfolio = {
            "total_cash": 1000000,
            "positions": {}
        }
    
    state = {
        "data": {
            "tickers": request.tickers,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "portfolio": request.portfolio,
            "analyst_signals": {},
            "final_decisions": {}
        },
        "metadata": request.metadata if request.metadata else {"show_reasoning": True},
        "messages": []
    }
    
    return state

async def run_agent_async(agent_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Run agent asynchronously"""
    try:
        agent_func = AGENT_REGISTRY.get(agent_name)
        if not agent_func:
            raise ValueError(f"Agent {agent_name} not found")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, agent_func, state)
        
        return {
            "status": "success",
            "agent": agent_name,
            "result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "agent": agent_name,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Endpoints

@router.get("/agents")
async def list_agents():
    """List all available agents"""
    return {
        "status": "success",
        "agents": {
            "beast_mode": BEAST_AGENTS,
            "classic": [a for a in AGENT_REGISTRY.keys() if a not in BEAST_AGENTS],
            "total": len(AGENT_REGISTRY)
        }
    }

@router.post("/analyze/{agent_name}")
async def analyze_with_agent(
    agent_name: str,
    request: AgentRequest,
    background_tasks: BackgroundTasks
):
    """Run analysis with a specific agent"""
    
    if agent_name not in AGENT_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    
    start_time = datetime.now()
    
    try:
        # Create state
        state = create_agent_state(request)
        
        # Run agent
        result = await run_agent_async(agent_name, state)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Extract analysis
        analysis = {}
        if "data" in result.get("result", {}):
            analyst_signals = result["result"]["data"].get("analyst_signals", {})
            if agent_name in analyst_signals:
                analysis = analyst_signals[agent_name]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return AgentResponse(
            status="success",
            agent=agent_name,
            analysis=analysis,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/beast-mode")
async def analyze_with_beast_agents(request: BeastAgentRequest):
    """Run analysis with all beast mode agents in parallel"""
    
    start_time = datetime.now()
    
    try:
        # Create state
        state = create_agent_state(request)
        
        # Add beast mode metadata
        state["metadata"]["analysis_depth"] = request.analysis_depth
        state["metadata"]["include_contrarian"] = request.include_contrarian
        state["metadata"]["risk_tolerance"] = request.risk_tolerance
        
        # Run all beast agents in parallel
        tasks = []
        for agent_name in BEAST_AGENTS:
            tasks.append(run_agent_async(agent_name, state))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        completed = []
        failed = []
        combined_analysis = {}
        
        for i, result in enumerate(results):
            agent_name = BEAST_AGENTS[i]
            
            if isinstance(result, Exception):
                failed.append(agent_name)
                combined_analysis[agent_name] = {"error": str(result)}
            elif result.get("status") == "error":
                failed.append(agent_name)
                combined_analysis[agent_name] = {"error": result.get("error")}
            else:
                completed.append(agent_name)
                # Extract analysis
                if "result" in result and "data" in result["result"]:
                    analyst_signals = result["result"]["data"].get("analyst_signals", {})
                    if agent_name in analyst_signals:
                        combined_analysis[agent_name] = analyst_signals[agent_name]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return MultiAgentResponse(
            status="success" if not failed else "partial",
            agents_completed=completed,
            agents_failed=failed,
            combined_analysis=combined_analysis,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/all")
async def analyze_with_all_agents(
    request: AgentRequest,
    include_beast: bool = Query(default=True, description="Include beast mode agents"),
    parallel: bool = Query(default=True, description="Run agents in parallel")
):
    """Run analysis with all available agents"""
    
    start_time = datetime.now()
    
    try:
        # Create state
        state = create_agent_state(request)
        
        # Select agents
        if include_beast:
            agents_to_run = list(AGENT_REGISTRY.keys())
        else:
            agents_to_run = [a for a in AGENT_REGISTRY.keys() if a not in BEAST_AGENTS]
        
        if parallel:
            # Run all agents in parallel
            tasks = []
            for agent_name in agents_to_run:
                tasks.append(run_agent_async(agent_name, state))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Run sequentially
            results = []
            for agent_name in agents_to_run:
                result = await run_agent_async(agent_name, state)
                results.append(result)
        
        # Process results
        completed = []
        failed = []
        combined_analysis = {}
        
        for i, result in enumerate(results):
            agent_name = agents_to_run[i]
            
            if isinstance(result, Exception):
                failed.append(agent_name)
                combined_analysis[agent_name] = {"error": str(result)}
            elif result.get("status") == "error":
                failed.append(agent_name)
                combined_analysis[agent_name] = {"error": result.get("error")}
            else:
                completed.append(agent_name)
                # Extract analysis
                if "result" in result and "data" in result["result"]:
                    analyst_signals = result["result"]["data"].get("analyst_signals", {})
                    if agent_name in analyst_signals:
                        combined_analysis[agent_name] = analyst_signals[agent_name]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return MultiAgentResponse(
            status="success" if not failed else "partial",
            agents_completed=completed,
            agents_failed=failed,
            combined_analysis=combined_analysis,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/consensus")
async def get_consensus_analysis(request: AgentRequest):
    """Get consensus analysis from multiple agents"""
    
    start_time = datetime.now()
    
    try:
        # Run all agents
        all_agents_result = await analyze_with_all_agents(request, include_beast=True, parallel=True)
        
        # Calculate consensus
        consensus = calculate_consensus(all_agents_result.combined_analysis)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "status": "success",
            "consensus": consensus,
            "agents_analyzed": all_agents_result.agents_completed,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def calculate_consensus(combined_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate consensus from multiple agent analyses"""
    
    consensus_by_ticker = {}
    
    # Extract all tickers
    all_tickers = set()
    for agent_analysis in combined_analysis.values():
        if isinstance(agent_analysis, dict) and not agent_analysis.get("error"):
            all_tickers.update(agent_analysis.keys())
    
    for ticker in all_tickers:
        signals = []
        confidences = []
        
        for agent_name, agent_analysis in combined_analysis.items():
            if isinstance(agent_analysis, dict) and ticker in agent_analysis:
                ticker_analysis = agent_analysis[ticker]
                
                if isinstance(ticker_analysis, dict):
                    signal = ticker_analysis.get("signal")
                    confidence = ticker_analysis.get("confidence", 0)
                    
                    if signal:
                        signals.append(signal)
                        confidences.append(confidence)
        
        if signals:
            # Map signals to numeric values
            signal_map = {
                "ultra_bullish": 2, "euphoric": 2, "quantum_surge": 2,
                "bullish": 1,
                "neutral": 0,
                "bearish": -1,
                "ultra_bearish": -2, "panic": -2, "quantum_collapse": -2
            }
            
            numeric_signals = [signal_map.get(s, 0) for s in signals]
            
            # Calculate weighted average
            if confidences:
                weighted_sum = sum(s * c for s, c in zip(numeric_signals, confidences))
                total_confidence = sum(confidences)
                consensus_score = weighted_sum / total_confidence if total_confidence > 0 else 0
            else:
                consensus_score = sum(numeric_signals) / len(numeric_signals)
            
            # Determine consensus signal
            if consensus_score > 1.5:
                consensus_signal = "strong_buy"
            elif consensus_score > 0.5:
                consensus_signal = "buy"
            elif consensus_score < -1.5:
                consensus_signal = "strong_sell"
            elif consensus_score < -0.5:
                consensus_signal = "sell"
            else:
                consensus_signal = "hold"
            
            # Calculate agreement level
            signal_std = np.std(numeric_signals) if len(numeric_signals) > 1 else 0
            agreement_level = max(0, 100 - signal_std * 50)
            
            consensus_by_ticker[ticker] = {
                "consensus_signal": consensus_signal,
                "consensus_score": consensus_score,
                "agreement_level": agreement_level,
                "num_agents": len(signals),
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
                "signal_distribution": dict(Counter(signals))
            }
    
    return consensus_by_ticker

@router.get("/health")
async def health_check():
    """Check health of all agents"""
    
    health_status = {}
    
    for agent_name in AGENT_REGISTRY:
        try:
            # Simple check to see if agent is callable
            agent_func = AGENT_REGISTRY[agent_name]
            health_status[agent_name] = {
                "status": "healthy",
                "type": "beast" if agent_name in BEAST_AGENTS else "classic"
            }
        except Exception as e:
            health_status[agent_name] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    all_healthy = all(s["status"] == "healthy" for s in health_status.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "agents": health_status,
        "timestamp": datetime.now().isoformat()
    }

# Import numpy for consensus calculation
import numpy as np
from collections import Counter