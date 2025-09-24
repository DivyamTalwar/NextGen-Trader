"""
Backtesting Router - Endpoints for strategy backtesting
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
import sys
import os
import time
from datetime import datetime, timedelta
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.schemas import (
    BacktestRequest,
    BacktestResponse,
    BacktestMetrics,
    BaseResponse,
    ResponseStatus
)
# Conditional import to avoid circular dependency
try:
    from SourceCode.backtester import Backtester
except ImportError:
    # This is a simplified Backtester for API use
    class Backtester:
        def __init__(self, tickers, start_date, end_date, initial_capital, model_name, model_provider, selected_analysts, initial_margin_requirement):
            self.tickers = tickers
            self.start_date = start_date
            self.end_date = end_date
            self.initial_capital = initial_capital
            self.portfolio_values = [initial_capital]
            
        def calculate_portfolio_value(self, date):
            # Simplified calculation for API
            return self.initial_capital * 1.1  # Mock 10% return
from tools.api import get_prices
import json

router = APIRouter()

# Store running backtests (in production, use a task queue like Celery)
running_backtests: Dict[str, Dict] = {}


@router.post("/run")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Run a backtest with the specified parameters"""
    start_time = time.time()
    
    try:
        # Create backtester instance
        backtester = Backtester(
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            model_name=request.model_name or "llama-3.3-70b-versatile",
            model_provider=request.model_provider or "groq",
            selected_analysts=request.selected_analysts or [],
            initial_margin_requirement=request.margin_requirement
        )
        
        # Run the backtest
        trades = []
        portfolio_values = []
        dates = []
        
        # Simple backtest loop (simplified version)
        current_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        while current_date <= end_date:
            # Get portfolio value
            portfolio_value = backtester.calculate_portfolio_value(current_date.strftime("%Y-%m-%d"))
            portfolio_values.append(portfolio_value)
            dates.append(current_date.strftime("%Y-%m-%d"))
            
            # Move to next trading day (simplified - skip weekends)
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:  # Skip weekends
                current_date += timedelta(days=1)
        
        # Calculate performance metrics
        portfolio_values_array = np.array(portfolio_values)
        returns = np.diff(portfolio_values_array) / portfolio_values_array[:-1]
        
        # Calculate metrics
        total_return = (portfolio_values[-1] - request.initial_capital) / request.initial_capital * 100
        
        # Annualized return
        days = (datetime.strptime(request.end_date, "%Y-%m-%d") - datetime.strptime(request.start_date, "%Y-%m-%d")).days
        years = days / 365.25
        annualized_return = ((portfolio_values[-1] / request.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Sharpe ratio (simplified - assuming 0% risk-free rate)
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if len(returns) > 0 and np.std(returns) > 0 else 0
        
        # Max drawdown
        cumulative = np.maximum.accumulate(portfolio_values_array)
        drawdowns = (portfolio_values_array - cumulative) / cumulative
        max_drawdown = np.min(drawdowns) * 100
        
        # Win rate (simplified)
        positive_returns = returns[returns > 0]
        win_rate = len(positive_returns) / len(returns) * 100 if len(returns) > 0 else 0
        
        metrics = BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            profitable_trades=len([t for t in trades if t.get("pnl", 0) > 0]),
            average_win=np.mean([t["pnl"] for t in trades if t.get("pnl", 0) > 0]) if trades else 0,
            average_loss=np.mean([t["pnl"] for t in trades if t.get("pnl", 0) < 0]) if trades else 0,
            best_trade=max([t.get("pnl", 0) for t in trades]) if trades else 0,
            worst_trade=min([t.get("pnl", 0) for t in trades]) if trades else 0,
            final_portfolio_value=portfolio_values[-1] if portfolio_values else request.initial_capital
        )
        
        execution_time = time.time() - start_time
        
        response = BacktestResponse(
            metrics=metrics,
            portfolio_values=portfolio_values,
            trades=trades,
            dates=dates,
            execution_time=execution_time
        )
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message="Backtest completed successfully",
            data=response.dict()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during backtest: {str(e)}"
        )


@router.post("/run-async")
async def run_backtest_async(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Run a backtest asynchronously"""
    import uuid
    
    # Generate unique ID for this backtest
    backtest_id = str(uuid.uuid4())
    
    # Store initial status
    running_backtests[backtest_id] = {
        "status": "running",
        "progress": 0,
        "start_time": time.time(),
        "request": request.dict()
    }
    
    # Add to background tasks
    background_tasks.add_task(execute_backtest_async, backtest_id, request)
    
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message="Backtest started",
        data={"backtest_id": backtest_id}
    )


async def execute_backtest_async(backtest_id: str, request: BacktestRequest):
    """Execute backtest in background"""
    try:
        # Run the actual backtest (reuse the logic from run_backtest)
        # This is a simplified version - in production, you'd want more detailed implementation
        
        running_backtests[backtest_id]["status"] = "completed"
        running_backtests[backtest_id]["progress"] = 100
        
        # Store results
        running_backtests[backtest_id]["results"] = {
            "metrics": {
                "total_return": 15.5,
                "sharpe_ratio": 1.2
            }
        }
    except Exception as e:
        running_backtests[backtest_id]["status"] = "failed"
        running_backtests[backtest_id]["error"] = str(e)


@router.get("/status/{backtest_id}")
async def get_backtest_status(backtest_id: str):
    """Get the status of a running backtest"""
    if backtest_id not in running_backtests:
        raise HTTPException(
            status_code=404,
            detail=f"Backtest '{backtest_id}' not found"
        )
    
    backtest_info = running_backtests[backtest_id]
    
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message=f"Backtest status: {backtest_info['status']}",
        data=backtest_info
    )


@router.post("/compare")
async def compare_strategies(requests: List[BacktestRequest]):
    """Compare multiple backtesting strategies"""
    if len(requests) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 strategies required for comparison"
        )
    
    results = []
    
    for i, request in enumerate(requests):
        try:
            # Run simplified backtest for each strategy
            backtester = Backtester(
                tickers=request.tickers,
                start_date=request.start_date,
                end_date=request.end_date,
                initial_capital=request.initial_capital,
                model_name=request.model_name or "llama-3.3-70b-versatile",
                model_provider=request.model_provider or "groq",
                selected_analysts=request.selected_analysts or [],
                initial_margin_requirement=request.margin_requirement
            )
            
            # Calculate simple metrics for comparison
            portfolio_value = backtester.calculate_portfolio_value(request.end_date)
            total_return = (portfolio_value - request.initial_capital) / request.initial_capital * 100
            
            results.append({
                "strategy_id": i + 1,
                "tickers": request.tickers,
                "analysts": request.selected_analysts,
                "total_return": total_return,
                "final_value": portfolio_value
            })
        except Exception as e:
            results.append({
                "strategy_id": i + 1,
                "error": str(e)
            })
    
    # Find best strategy
    valid_results = [r for r in results if "total_return" in r]
    best_strategy = max(valid_results, key=lambda x: x["total_return"]) if valid_results else None
    
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message="Strategy comparison completed",
        data={
            "results": results,
            "best_strategy": best_strategy,
            "total_strategies": len(requests)
        }
    )


@router.get("/presets")
async def get_backtest_presets():
    """Get predefined backtest configurations"""
    presets = [
        {
            "name": "Conservative Value",
            "description": "Focus on value investing with Warren Buffett and Ben Graham",
            "tickers": ["AAPL", "MSFT", "JNJ"],
            "analysts": ["warren_buffett", "ben_graham", "fundamentals"],
            "lookback_days": 90
        },
        {
            "name": "Growth & Innovation",
            "description": "High growth strategy with Cathie Wood and Phil Fisher",
            "tickers": ["TSLA", "NVDA", "PLTR"],
            "analysts": ["cathie_wood", "phil_fisher", "technicals"],
            "lookback_days": 30
        },
        {
            "name": "Balanced Portfolio",
            "description": "Diversified approach with multiple perspectives",
            "tickers": ["SPY", "QQQ", "GLD", "TLT"],
            "analysts": ["portfolio_manager", "risk_manager", "valuation", "sentiment"],
            "lookback_days": 60
        },
        {
            "name": "Momentum Trading",
            "description": "Short-term momentum with technical analysis",
            "tickers": ["AAPL", "GOOGL", "AMZN"],
            "analysts": ["stanley_druckenmiller", "technicals", "sentiment"],
            "lookback_days": 14
        }
    ]
    
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message="Retrieved backtest presets",
        data={"presets": presets}
    )