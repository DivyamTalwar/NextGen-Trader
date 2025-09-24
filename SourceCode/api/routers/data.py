"""
Data Router - Endpoints for market data and financial information (COMPLETE FIX)
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.schemas import (
    PriceDataRequest,
    PriceData,
    FinancialMetricsRequest,
    FinancialMetrics,
    NewsRequest,
    NewsArticle,
    InsiderTradeRequest,
    InsiderTrade,
    BaseResponse,
    ResponseStatus
)
from tools.api import (
    get_prices,
    get_financial_metrics,
    get_company_news,
    get_insider_trades,
    get_market_cap,
    search_line_items
)

router = APIRouter()


@router.post("/prices")
async def get_price_data(request: PriceDataRequest):
    """Get historical price data for a ticker"""
    try:
        prices = get_prices(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if not prices:
            raise HTTPException(
                status_code=404,
                detail=f"No price data found for {request.ticker} in the specified date range"
            )
        
        # Convert to response format (prices is a list of Price objects with 'time' field)
        price_data = []
        for price in prices:
            price_data.append(PriceData(
                date=price.time,  # Price model uses 'time' field, not 'date'
                open=price.open,
                high=price.high,
                low=price.low,
                close=price.close,
                volume=price.volume,
                adjusted_close=getattr(price, 'adjusted_close', None)
            ))
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Retrieved {len(price_data)} price records for {request.ticker}",
            data={
                "ticker": request.ticker,
                "prices": [p.dict() for p in price_data],
                "count": len(price_data)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching price data: {str(e)}"
        )


@router.get("/prices/{ticker}/latest")
async def get_latest_price(ticker: str):
    """Get the latest price for a ticker"""
    try:
        # Get last 5 days of prices
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        
        prices = get_prices(ticker, start_date, end_date)
        
        if not prices:
            raise HTTPException(
                status_code=404,
                detail=f"No price data found for {ticker}"
            )
        
        latest_price = prices[-1]
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Retrieved latest price for {ticker}",
            data={
                "ticker": ticker,
                "date": latest_price.time,  # Price model uses 'time' field
                "price": latest_price.close,
                "open": latest_price.open,
                "high": latest_price.high,
                "low": latest_price.low,
                "volume": latest_price.volume,
                "change": latest_price.close - latest_price.open,
                "change_percent": ((latest_price.close - latest_price.open) / latest_price.open) * 100 if latest_price.open > 0 else 0
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching latest price: {str(e)}"
        )


@router.get("/financial-metrics/{ticker}")
async def get_financial_data(ticker: str, limit: int = Query(10, ge=1, le=100)):
    """Get financial metrics for a ticker"""
    try:
        # Get current date for end_date
        end_date = datetime.now().strftime("%Y-%m-%d")
        metrics = get_financial_metrics(ticker, end_date=end_date, limit=limit)
        
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail=f"No financial data found for {ticker}"
            )
        
        # Get market cap (requires end_date)
        market_cap = get_market_cap(ticker, end_date)
        
        # Format the latest metrics
        latest_metrics = metrics[0] if metrics else None
        
        if latest_metrics:
            financial_data = FinancialMetrics(
                ticker=ticker,
                market_cap=market_cap,
                pe_ratio=getattr(latest_metrics, 'price_to_earnings_ratio', None),
                pb_ratio=getattr(latest_metrics, 'price_to_book_ratio', None),
                dividend_yield=getattr(latest_metrics, 'dividend_yield', None),
                revenue_growth=getattr(latest_metrics, 'revenue_growth_yoy', None),
                profit_margin=getattr(latest_metrics, 'net_margin', None),
                roe=getattr(latest_metrics, 'return_on_equity', None),
                debt_to_equity=getattr(latest_metrics, 'debt_to_equity', None),
                current_ratio=getattr(latest_metrics, 'current_ratio', None),
                quick_ratio=getattr(latest_metrics, 'quick_ratio', None)
            )
            
            return BaseResponse(
                status=ResponseStatus.SUCCESS,
                message=f"Retrieved financial metrics for {ticker}",
                data={
                    "current": financial_data.dict(),
                    "historical": [m.model_dump() for m in metrics[:limit]]
                }
            )
        else:
            return BaseResponse(
                status=ResponseStatus.SUCCESS,
                message=f"Limited financial metrics for {ticker}",
                data={
                    "ticker": ticker,
                    "market_cap": market_cap
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching financial metrics: {str(e)}"
        )


@router.get("/news/{ticker}")
async def get_news(ticker: str, limit: int = Query(10, ge=1, le=50)):
    """Get latest news for a ticker"""
    try:
        # get_company_news requires end_date parameter
        end_date = datetime.now().strftime("%Y-%m-%d")
        news_items = get_company_news(ticker, end_date=end_date, limit=limit)
        
        if not news_items:
            raise HTTPException(
                status_code=404,
                detail=f"No news found for {ticker}"
            )
        
        # news_items is a list of CompanyNews objects
        # Convert to response format using actual field names
        articles = []
        for news in news_items:
            articles.append(NewsArticle(
                title=news.title,
                url=news.url,
                published_date=news.date,  # CompanyNews uses 'date' field
                source=news.source,
                summary=getattr(news, 'summary', None) or getattr(news, 'sentiment', '')
            ))
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Retrieved {len(articles)} news articles for {ticker}",
            data={
                "ticker": ticker,
                "articles": [a.dict() for a in articles],
                "count": len(articles)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching news: {str(e)}"
        )


@router.get("/insider-trades/{ticker}")
async def get_insider_trading(ticker: str, limit: int = Query(20, ge=1, le=100)):
    """Get insider trading data for a ticker"""
    try:
        # get_insider_trades requires end_date parameter
        end_date = datetime.now().strftime("%Y-%m-%d")
        trades = get_insider_trades(ticker, end_date=end_date, limit=limit)
        
        if not trades:
            raise HTTPException(
                status_code=404,
                detail=f"No insider trades found for {ticker}"
            )
        
        # trades is a list of InsiderTrade objects
        # Convert to response format using actual field names from the model
        insider_trades = []
        for trade in trades:
            # InsiderTrade model fields:
            # name, title, transaction_date, transaction_shares, 
            # transaction_price_per_share, transaction_value
            insider_trades.append(InsiderTrade(
                insider_name=trade.name or "Unknown",
                position=trade.title or "Unknown",
                transaction_date=trade.transaction_date or "",
                transaction_type="Buy" if (trade.transaction_shares and trade.transaction_shares > 0) else "Sell",
                shares=abs(trade.transaction_shares) if trade.transaction_shares else 0,
                price=trade.transaction_price_per_share or 0,
                value=abs(trade.transaction_value) if trade.transaction_value else 0
            ))
        
        # Calculate summary statistics
        total_bought = sum(t.value for t in insider_trades if t.shares > 0)
        total_sold = sum(t.value for t in insider_trades if t.shares < 0 or t.transaction_type == "Sell")
        net_position = total_bought - total_sold
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Retrieved {len(insider_trades)} insider trades for {ticker}",
            data={
                "ticker": ticker,
                "trades": [t.dict() for t in insider_trades],
                "count": len(insider_trades),
                "summary": {
                    "total_bought": total_bought,
                    "total_sold": total_sold,
                    "net_position": net_position,
                    "sentiment": "bullish" if net_position > 0 else "bearish" if net_position < 0 else "neutral"
                }
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching insider trades: {str(e)}"
        )


@router.get("/line-items/{ticker}")
async def get_line_items(
    ticker: str,
    query: str = Query(..., description="Search query for line items"),
    limit: int = Query(10, ge=1, le=50)
):
    """Search for specific financial line items"""
    try:
        # Convert query string to list (split by comma)
        line_items_list = [item.strip() for item in query.split(',')]
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        items = search_line_items(ticker, line_items_list, end_date=end_date, limit=limit)
        
        if not items:
            return BaseResponse(
                status=ResponseStatus.SUCCESS,
                message=f"No line items found for query '{query}'",
                data={"ticker": ticker, "query": query, "items": [], "count": 0}
            )
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Found {len(items)} line items for '{query}'",
            data={
                "ticker": ticker,
                "query": query,
                "items": [item.model_dump() for item in items],
                "count": len(items)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching line items: {str(e)}"
        )


@router.get("/market-cap/{ticker}")
async def get_ticker_market_cap(ticker: str):
    """Get current market capitalization for a ticker"""
    try:
        # Get current market cap
        end_date = datetime.now().strftime("%Y-%m-%d")
        market_cap = get_market_cap(ticker, end_date)
        
        if market_cap is None:
            raise HTTPException(
                status_code=404,
                detail=f"Market cap not found for {ticker}"
            )
        
        # Categorize by market cap size
        if market_cap >= 200_000_000_000:
            category = "Mega Cap"
        elif market_cap >= 10_000_000_000:
            category = "Large Cap"
        elif market_cap >= 2_000_000_000:
            category = "Mid Cap"
        elif market_cap >= 300_000_000:
            category = "Small Cap"
        else:
            category = "Micro Cap"
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Retrieved market cap for {ticker}",
            data={
                "ticker": ticker,
                "market_cap": market_cap,
                "market_cap_formatted": f"${market_cap:,.0f}",
                "category": category
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching market cap: {str(e)}"
        )


@router.post("/batch-quotes")
async def get_batch_quotes(tickers: List[str]):
    """Get latest quotes for multiple tickers"""
    try:
        quotes = []
        errors = []
        
        for ticker in tickers:
            try:
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
                
                prices = get_prices(ticker, start_date, end_date)
                
                if prices:
                    latest = prices[-1]
                    quotes.append({
                        "ticker": ticker,
                        "price": latest.close,
                        "change": latest.close - latest.open,
                        "change_percent": ((latest.close - latest.open) / latest.open) * 100 if latest.open > 0 else 0,
                        "volume": latest.volume,
                        "date": latest.time  # Price model uses 'time' field
                    })
                else:
                    errors.append({"ticker": ticker, "error": "No data available"})
            except Exception as e:
                errors.append({"ticker": ticker, "error": str(e)})
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Retrieved quotes for {len(quotes)} out of {len(tickers)} tickers",
            data={
                "quotes": quotes,
                "errors": errors,
                "success_count": len(quotes),
                "error_count": len(errors)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching batch quotes: {str(e)}"
        )