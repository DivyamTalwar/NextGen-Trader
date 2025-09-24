"""
Portfolio Router - Endpoints for portfolio management
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.schemas import (
    Portfolio,
    Position,
    PortfolioUpdate,
    BaseResponse,
    ResponseStatus,
    TradingAction
)
from tools.api import get_prices
from datetime import datetime

router = APIRouter()

# In-memory portfolio storage (in production, use a database)
portfolios: Dict[str, Portfolio] = {}


@router.post("/create")
async def create_portfolio(portfolio_id: str, initial_capital: float = 100000, margin_requirement: float = 0.2):
    """Create a new portfolio"""
    if portfolio_id in portfolios:
        raise HTTPException(
            status_code=400,
            detail=f"Portfolio '{portfolio_id}' already exists"
        )
    
    portfolio = Portfolio(
        total_cash=initial_capital,
        margin_requirement=margin_requirement,
        positions={}
    )
    
    portfolios[portfolio_id] = portfolio
    
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message=f"Portfolio '{portfolio_id}' created successfully",
        data=portfolio.dict()
    )


@router.get("/{portfolio_id}")
async def get_portfolio(portfolio_id: str):
    """Get portfolio details"""
    if portfolio_id not in portfolios:
        raise HTTPException(
            status_code=404,
            detail=f"Portfolio '{portfolio_id}' not found"
        )
    
    portfolio = portfolios[portfolio_id]
    
    # Calculate portfolio metrics
    total_value = portfolio.total_cash
    total_long_value = 0
    total_short_value = 0
    
    for ticker, position in portfolio.positions.items():
        if position.long > 0 or position.short > 0:
            # Get current price (for demo, using last available price)
            try:
                prices = get_prices(ticker, "2024-01-01", datetime.now().strftime("%Y-%m-%d"))
                if prices and len(prices) > 0:
                    current_price = prices[-1]["close"]
                    total_long_value += position.long * current_price
                    total_short_value += position.short * current_price
            except:
                pass
    
    total_value += total_long_value - total_short_value
    
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message=f"Retrieved portfolio '{portfolio_id}'",
        data={
            "portfolio": portfolio.dict(),
            "metrics": {
                "total_value": total_value,
                "cash": portfolio.total_cash,
                "long_value": total_long_value,
                "short_value": total_short_value,
                "margin_used": total_short_value * portfolio.margin_requirement
            }
        }
    )


@router.post("/{portfolio_id}/update")
async def update_portfolio_position(portfolio_id: str, update: PortfolioUpdate):
    """Update a position in the portfolio"""
    if portfolio_id not in portfolios:
        raise HTTPException(
            status_code=404,
            detail=f"Portfolio '{portfolio_id}' not found"
        )
    
    portfolio = portfolios[portfolio_id]
    
    # Initialize position if it doesn't exist
    if update.ticker not in portfolio.positions:
        portfolio.positions[update.ticker] = Position()
    
    position = portfolio.positions[update.ticker]
    
    # Get current price if not provided
    if update.price is None:
        try:
            prices = get_prices(update.ticker, "2024-01-01", datetime.now().strftime("%Y-%m-%d"))
            if prices and len(prices) > 0:
                current_price = prices[-1]["close"]
            else:
                raise HTTPException(status_code=400, detail=f"Could not get price for {update.ticker}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error getting price: {str(e)}")
    else:
        current_price = update.price
    
    # Execute the trade based on action
    cost = update.quantity * current_price
    
    if update.action == TradingAction.BUY:
        if cost > portfolio.total_cash:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient cash. Required: ${cost:.2f}, Available: ${portfolio.total_cash:.2f}"
            )
        
        # Update position
        old_shares = position.long
        old_cost_basis = position.long_cost_basis
        total_shares = old_shares + update.quantity
        
        if total_shares > 0:
            position.long_cost_basis = ((old_cost_basis * old_shares) + cost) / total_shares
        
        position.long += update.quantity
        portfolio.total_cash -= cost
        
    elif update.action == TradingAction.SELL:
        if update.quantity > position.long:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot sell {update.quantity} shares. Only {position.long} available"
            )
        
        position.long -= update.quantity
        portfolio.total_cash += cost
        
        if position.long == 0:
            position.long_cost_basis = 0.0
    
    elif update.action == TradingAction.SHORT:
        margin_required = cost * portfolio.margin_requirement
        if margin_required > portfolio.total_cash:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient margin. Required: ${margin_required:.2f}, Available: ${portfolio.total_cash:.2f}"
            )
        
        # Update short position
        old_shares = position.short
        old_cost_basis = position.short_cost_basis
        total_shares = old_shares + update.quantity
        
        if total_shares > 0:
            position.short_cost_basis = ((old_cost_basis * old_shares) + cost) / total_shares
        
        position.short += update.quantity
        portfolio.total_cash += cost  # Receive cash from short sale
        
    elif update.action == TradingAction.COVER:
        if update.quantity > position.short:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cover {update.quantity} shares. Only {position.short} shorted"
            )
        
        if cost > portfolio.total_cash:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient cash to cover. Required: ${cost:.2f}, Available: ${portfolio.total_cash:.2f}"
            )
        
        position.short -= update.quantity
        portfolio.total_cash -= cost  # Pay to cover short
        
        if position.short == 0:
            position.short_cost_basis = 0.0
    
    # Calculate P&L for the trade
    pnl = 0
    if update.action == TradingAction.SELL and position.long_cost_basis > 0:
        pnl = (current_price - position.long_cost_basis) * update.quantity
    elif update.action == TradingAction.COVER and position.short_cost_basis > 0:
        pnl = (position.short_cost_basis - current_price) * update.quantity
    
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message=f"Successfully executed {update.action.value} order for {update.quantity} shares of {update.ticker}",
        data={
            "ticker": update.ticker,
            "action": update.action.value,
            "quantity": update.quantity,
            "price": current_price,
            "total_value": cost,
            "pnl": pnl,
            "new_position": position.dict(),
            "remaining_cash": portfolio.total_cash
        }
    )


@router.delete("/{portfolio_id}")
async def delete_portfolio(portfolio_id: str):
    """Delete a portfolio"""
    if portfolio_id not in portfolios:
        raise HTTPException(
            status_code=404,
            detail=f"Portfolio '{portfolio_id}' not found"
        )
    
    del portfolios[portfolio_id]
    
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message=f"Portfolio '{portfolio_id}' deleted successfully",
        data=None
    )


@router.get("/{portfolio_id}/positions")
async def get_portfolio_positions(portfolio_id: str):
    """Get all positions in a portfolio with current values"""
    if portfolio_id not in portfolios:
        raise HTTPException(
            status_code=404,
            detail=f"Portfolio '{portfolio_id}' not found"
        )
    
    portfolio = portfolios[portfolio_id]
    positions_with_values = []
    
    for ticker, position in portfolio.positions.items():
        if position.long > 0 or position.short > 0:
            try:
                prices = get_prices(ticker, "2024-01-01", datetime.now().strftime("%Y-%m-%d"))
                current_price = prices[-1]["close"] if prices else 0
                
                long_value = position.long * current_price
                short_value = position.short * current_price
                long_pnl = (current_price - position.long_cost_basis) * position.long if position.long > 0 else 0
                short_pnl = (position.short_cost_basis - current_price) * position.short if position.short > 0 else 0
                
                positions_with_values.append({
                    "ticker": ticker,
                    "long_shares": position.long,
                    "short_shares": position.short,
                    "long_cost_basis": position.long_cost_basis,
                    "short_cost_basis": position.short_cost_basis,
                    "current_price": current_price,
                    "long_value": long_value,
                    "short_value": short_value,
                    "long_pnl": long_pnl,
                    "short_pnl": short_pnl,
                    "total_pnl": long_pnl + short_pnl
                })
            except:
                continue
    
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message=f"Retrieved positions for portfolio '{portfolio_id}'",
        data={
            "positions": positions_with_values,
            "total_positions": len(positions_with_values)
        }
    )


@router.post("/{portfolio_id}/reset")
async def reset_portfolio(portfolio_id: str, initial_capital: float = 100000):
    """Reset a portfolio to initial state"""
    if portfolio_id not in portfolios:
        raise HTTPException(
            status_code=404,
            detail=f"Portfolio '{portfolio_id}' not found"
        )
    
    portfolio = portfolios[portfolio_id]
    portfolio.total_cash = initial_capital
    portfolio.positions = {}
    
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message=f"Portfolio '{portfolio_id}' reset successfully",
        data=portfolio.dict()
    )