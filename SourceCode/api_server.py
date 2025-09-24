"""
FastAPI Backend Server for NextGenTrader
This server provides REST API endpoints for the AI-powered hedge fund system
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
import os
import logging
from datetime import datetime

# Import routers
from api.routers import agents, portfolio, backtesting, data, workflow, models, beast_agents

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="NextGenTrader API",
    description="AI-powered autonomous hedge fund system API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "data": None
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API server is running and healthy"""
    return {
        "status": "success",
        "message": "API server is healthy",
        "data": {
            "server_time": datetime.now().isoformat(),
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development")
        }
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "status": "success",
        "message": "Welcome to NextGenTrader API",
        "data": {
            "documentation": "/api/docs",
            "health": "/health",
            "version": "1.0.0"
        }
    }

# Include routers
app.include_router(agents.router, prefix="/api/agents", tags=["Agents"])
app.include_router(beast_agents.router, prefix="/api/beast", tags=["Beast Agents"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(backtesting.router, prefix="/api/backtesting", tags=["Backtesting"])
app.include_router(data.router, prefix="/api/data", tags=["Market Data"])
app.include_router(workflow.router, prefix="/api/workflow", tags=["Workflow"])
app.include_router(models.router, prefix="/api/models", tags=["LLM Models"])

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("NextGenTrader API Server starting up...")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info("All routers loaded successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("NextGenTrader API Server shutting down...")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )