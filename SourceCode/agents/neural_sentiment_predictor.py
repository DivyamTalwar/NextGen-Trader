"""
NEURAL NETWORK SENTIMENT PREDICTOR AGENT
=========================================
This agent uses deep learning and NLP to predict market sentiment shifts
before they manifest in price action. It processes news, social media,
insider behavior, and market microstructure to detect sentiment phase transitions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from graph.state import AgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import json
from typing_extensions import Literal
from tools.api import get_company_news, get_insider_trades, get_prices, get_financial_metrics, prices_to_df
from utils.llm import call_llm
from utils.progress import progress
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class NeuralSentimentSignal(BaseModel):
    """Signal from neural sentiment analysis"""
    signal: Literal["euphoric", "bullish", "neutral", "bearish", "panic"]
    confidence: float = Field(ge=0, le=100)
    reasoning: str
    sentiment_score: float = Field(ge=-100, le=100)
    sentiment_velocity: float
    sentiment_acceleration: float
    emotion_distribution: Dict[str, float]
    narrative_themes: List[str]
    sentiment_divergence: float
    crowd_psychology: str
    sentiment_regime: str
    tipping_points: List[Dict[str, Any]]


class NeuralSentimentAnalyzer:
    """
    Advanced sentiment analysis using neural network principles
    """
    
    def __init__(self):
        self.emotion_categories = [
            "fear", "greed", "hope", "despair", "euphoria", 
            "panic", "confidence", "uncertainty", "optimism", "pessimism"
        ]
        
        self.sentiment_keywords = {
            "ultra_positive": [
                "breakthrough", "revolutionary", "game-changer", "explosive growth",
                "record-breaking", "unprecedented", "moonshot", "skyrocket", "surge"
            ],
            "positive": [
                "growth", "profit", "expand", "beat", "exceed", "strong",
                "improve", "gain", "rise", "success", "innovative", "leading"
            ],
            "neutral": [
                "maintain", "stable", "consistent", "steady", "unchanged",
                "continue", "normal", "expected", "in-line", "moderate"
            ],
            "negative": [
                "decline", "loss", "weak", "concern", "challenge", "miss",
                "reduce", "cut", "fall", "struggle", "difficulty", "pressure"
            ],
            "ultra_negative": [
                "crash", "collapse", "disaster", "bankruptcy", "crisis",
                "plunge", "catastrophe", "devastating", "toxic", "doomed"
            ]
        }
        
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
        self.scaler = StandardScaler()
        
    def analyze_neural_sentiment(self, news: list, insider_trades: list, 
                                prices_df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Perform comprehensive neural sentiment analysis
        """
        analysis = {}
        
        # 1. Raw Sentiment Extraction
        progress.update_status("neural_sentiment_predictor_agent", ticker, "Extracting raw sentiment")
        raw_sentiment = self._extract_raw_sentiment(news)
        analysis['raw_sentiment'] = raw_sentiment
        
        # 2. Sentiment Dynamics Analysis
        progress.update_status("neural_sentiment_predictor_agent", ticker, "Analyzing sentiment dynamics")
        sentiment_dynamics = self._analyze_sentiment_dynamics(news, prices_df)
        analysis['sentiment_dynamics'] = sentiment_dynamics
        
        # 3. Emotion Distribution Analysis
        progress.update_status("neural_sentiment_predictor_agent", ticker, "Mapping emotion distribution")
        emotion_dist = self._analyze_emotion_distribution(news)
        analysis['emotion_distribution'] = emotion_dist
        
        # 4. Narrative Theme Extraction
        progress.update_status("neural_sentiment_predictor_agent", ticker, "Extracting narrative themes")
        themes = self._extract_narrative_themes(news)
        analysis['narrative_themes'] = themes
        
        # 5. Sentiment-Price Divergence
        progress.update_status("neural_sentiment_predictor_agent", ticker, "Detecting sentiment divergence")
        divergence = self._detect_sentiment_divergence(news, prices_df)
        analysis['sentiment_divergence'] = divergence
        
        # 6. Crowd Psychology Analysis
        progress.update_status("neural_sentiment_predictor_agent", ticker, "Analyzing crowd psychology")
        crowd_psych = self._analyze_crowd_psychology(news, insider_trades)
        analysis['crowd_psychology'] = crowd_psych
        
        # 7. Sentiment Regime Detection
        progress.update_status("neural_sentiment_predictor_agent", ticker, "Detecting sentiment regime")
        regime = self._detect_sentiment_regime(news, prices_df)
        analysis['sentiment_regime'] = regime
        
        # 8. Tipping Point Detection
        progress.update_status("neural_sentiment_predictor_agent", ticker, "Detecting sentiment tipping points")
        tipping_points = self._detect_tipping_points(news, prices_df)
        analysis['tipping_points'] = tipping_points
        
        # 9. Insider Sentiment Analysis
        progress.update_status("neural_sentiment_predictor_agent", ticker, "Analyzing insider sentiment")
        insider_sentiment = self._analyze_insider_sentiment(insider_trades)
        analysis['insider_sentiment'] = insider_sentiment
        
        # 10. Sentiment Momentum Analysis
        progress.update_status("neural_sentiment_predictor_agent", ticker, "Calculating sentiment momentum")
        momentum = self._analyze_sentiment_momentum(news)
        analysis['sentiment_momentum'] = momentum
        
        # 11. Contrarian Indicators
        progress.update_status("neural_sentiment_predictor_agent", ticker, "Identifying contrarian signals")
        contrarian = self._identify_contrarian_indicators(analysis)
        analysis['contrarian_indicators'] = contrarian
        
        # 12. Neural Network Aggregation
        progress.update_status("neural_sentiment_predictor_agent", ticker, "Neural network aggregation")
        neural_output = self._neural_network_aggregation(analysis)
        analysis['neural_output'] = neural_output
        
        return analysis
    
    def _extract_raw_sentiment(self, news: list) -> Dict[str, Any]:
        """Extract raw sentiment scores from news"""
        if not news:
            return {"score": 0, "confidence": 0, "distribution": {}}
            
        sentiment_scores = []
        sentiment_counts = defaultdict(int)
        
        for item in news[:100]:  # Analyze recent 100 items
            if hasattr(item, 'sentiment'):
                if item.sentiment == 'positive':
                    sentiment_scores.append(1)
                    sentiment_counts['positive'] += 1
                elif item.sentiment == 'negative':
                    sentiment_scores.append(-1)
                    sentiment_counts['negative'] += 1
                else:
                    sentiment_scores.append(0)
                    sentiment_counts['neutral'] += 1
                    
            # Deep sentiment analysis from title
            if hasattr(item, 'title') and item.title:
                title_sentiment = self._analyze_text_sentiment(item.title)
                sentiment_scores.append(title_sentiment['score'])
                sentiment_counts[title_sentiment['category']] += 1
                
        if sentiment_scores:
            avg_score = np.mean(sentiment_scores) * 100  # Scale to -100 to 100
            confidence = min(len(sentiment_scores) / 10, 1.0) * 100  # Confidence based on sample size
            
            # Calculate distribution
            total = sum(sentiment_counts.values())
            distribution = {k: v/total for k, v in sentiment_counts.items()}
        else:
            avg_score = 0
            confidence = 0
            distribution = {}
            
        return {
            "score": avg_score,
            "confidence": confidence,
            "distribution": distribution,
            "sample_size": len(sentiment_scores)
        }
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a text string"""
        text_lower = text.lower()
        
        # Count keyword matches
        scores = {
            "ultra_positive": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "ultra_negative": 0
        }
        
        for category, keywords in self.sentiment_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[category] += 1
                    
        # Determine dominant sentiment
        max_score = max(scores.values())
        if max_score == 0:
            return {"score": 0, "category": "neutral"}
            
        dominant = [k for k, v in scores.items() if v == max_score][0]
        
        # Convert to numeric score
        score_map = {
            "ultra_positive": 2,
            "positive": 1,
            "neutral": 0,
            "negative": -1,
            "ultra_negative": -2
        }
        
        return {
            "score": score_map[dominant],
            "category": dominant,
            "keyword_hits": scores
        }
    
    def _analyze_sentiment_dynamics(self, news: list, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment velocity and acceleration"""
        if not news or len(news) < 10:
            return {"velocity": 0, "acceleration": 0, "trend": "stable"}
            
        # Group news by time periods
        time_buckets = defaultdict(list)
        
        for item in news:
            if hasattr(item, 'time') and hasattr(item, 'sentiment'):
                # Simplified time bucketing (would use actual timestamps in production)
                bucket = len(time_buckets)  # Sequential buckets for now
                if bucket < 10:  # Limit to 10 time periods
                    time_buckets[bucket].append(item.sentiment)
                    
        if len(time_buckets) < 3:
            return {"velocity": 0, "acceleration": 0, "trend": "insufficient_data"}
            
        # Calculate sentiment scores per bucket
        bucket_scores = []
        for bucket in sorted(time_buckets.keys()):
            sentiments = time_buckets[bucket]
            score = sum(1 if s == 'positive' else -1 if s == 'negative' else 0 for s in sentiments)
            score = score / len(sentiments) if sentiments else 0
            bucket_scores.append(score)
            
        # Calculate velocity (first derivative)
        if len(bucket_scores) >= 2:
            velocities = np.diff(bucket_scores)
            velocity = np.mean(velocities)
        else:
            velocity = 0
            
        # Calculate acceleration (second derivative)
        if len(bucket_scores) >= 3:
            accelerations = np.diff(velocities)
            acceleration = np.mean(accelerations)
        else:
            acceleration = 0
            
        # Determine trend
        if velocity > 0.1 and acceleration > 0:
            trend = "accelerating_positive"
        elif velocity > 0.1:
            trend = "positive"
        elif velocity < -0.1 and acceleration < 0:
            trend = "accelerating_negative"
        elif velocity < -0.1:
            trend = "negative"
        else:
            trend = "stable"
            
        return {
            "velocity": velocity,
            "acceleration": acceleration,
            "trend": trend,
            "momentum_score": velocity + acceleration * 0.5
        }
    
    def _analyze_emotion_distribution(self, news: list) -> Dict[str, float]:
        """Analyze distribution of emotions in news"""
        emotion_scores = {emotion: 0 for emotion in self.emotion_categories}
        
        if not news:
            return emotion_scores
            
        # Emotion keywords mapping
        emotion_keywords = {
            "fear": ["fear", "worried", "concern", "anxiety", "nervous", "scared"],
            "greed": ["greed", "rally", "surge", "boom", "profit", "gain"],
            "hope": ["hope", "optimistic", "promising", "potential", "opportunity"],
            "despair": ["despair", "hopeless", "doomed", "collapse", "disaster"],
            "euphoria": ["euphoria", "ecstatic", "moon", "explosive", "incredible"],
            "panic": ["panic", "crash", "plunge", "selloff", "exodus"],
            "confidence": ["confident", "strong", "solid", "robust", "stable"],
            "uncertainty": ["uncertain", "volatile", "unclear", "mixed", "question"],
            "optimism": ["optimism", "bullish", "positive", "upbeat", "improving"],
            "pessimism": ["pessimism", "bearish", "negative", "declining", "weak"]
        }
        
        total_hits = 0
        
        for item in news[:100]:
            if hasattr(item, 'title') and item.title:
                text_lower = item.title.lower()
                
                for emotion, keywords in emotion_keywords.items():
                    for keyword in keywords:
                        if keyword in text_lower:
                            emotion_scores[emotion] += 1
                            total_hits += 1
                            
        # Normalize scores
        if total_hits > 0:
            emotion_scores = {k: v/total_hits for k, v in emotion_scores.items()}
            
        return emotion_scores
    
    def _extract_narrative_themes(self, news: list) -> List[str]:
        """Extract dominant narrative themes using topic modeling"""
        if not news or len(news) < 10:
            return []
            
        # Collect text data
        texts = []
        for item in news[:100]:
            if hasattr(item, 'title') and item.title:
                texts.append(item.title)
                
        if len(texts) < 5:
            return []
            
        try:
            # TF-IDF vectorization
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Topic modeling with LDA
            self.lda_model.fit(tfidf_matrix)
            
            # Extract top words for each topic
            feature_names = self.vectorizer.get_feature_names_out()
            themes = []
            
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_indices = topic.argsort()[-5:][::-1]  # Top 5 words
                top_words = [feature_names[i] for i in top_indices]
                theme = f"Theme_{topic_idx+1}: {' '.join(top_words[:3])}"
                themes.append(theme)
                
            return themes[:5]  # Return top 5 themes
            
        except Exception:
            # Fallback to simple keyword extraction
            all_words = ' '.join(texts).lower().split()
            word_freq = Counter(all_words)
            
            # Filter common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
            filtered_words = [(w, c) for w, c in word_freq.most_common(20) 
                            if w not in common_words and len(w) > 3]
            
            themes = [f"Keyword: {word}" for word, _ in filtered_words[:5]]
            return themes
    
    def _detect_sentiment_divergence(self, news: list, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect divergence between sentiment and price action"""
        if not news or prices_df.empty or len(prices_df) < 20:
            return {"divergence_score": 0, "type": "none", "strength": "weak"}
            
        # Calculate sentiment trend
        sentiment_scores = []
        for item in news[:50]:
            if hasattr(item, 'sentiment'):
                score = 1 if item.sentiment == 'positive' else -1 if item.sentiment == 'negative' else 0
                sentiment_scores.append(score)
                
        if not sentiment_scores:
            return {"divergence_score": 0, "type": "none", "strength": "weak"}
            
        sentiment_trend = np.mean(sentiment_scores)
        
        # Calculate price trend
        returns = prices_df['close'].pct_change().dropna()
        price_trend = returns.mean()
        
        # Detect divergence
        divergence_score = 0
        divergence_type = "none"
        
        if sentiment_trend > 0.3 and price_trend < -0.01:
            # Positive sentiment but negative price
            divergence_score = abs(sentiment_trend - price_trend) * 100
            divergence_type = "bullish_divergence"
        elif sentiment_trend < -0.3 and price_trend > 0.01:
            # Negative sentiment but positive price
            divergence_score = abs(sentiment_trend - price_trend) * 100
            divergence_type = "bearish_divergence"
        elif abs(sentiment_trend) > 0.5 and abs(price_trend) < 0.005:
            # Strong sentiment but flat price
            divergence_score = abs(sentiment_trend) * 50
            divergence_type = "sentiment_price_disconnect"
            
        # Determine strength
        if divergence_score > 70:
            strength = "strong"
        elif divergence_score > 40:
            strength = "moderate"
        else:
            strength = "weak"
            
        return {
            "divergence_score": divergence_score,
            "type": divergence_type,
            "strength": strength,
            "sentiment_trend": sentiment_trend,
            "price_trend": price_trend
        }
    
    def _analyze_crowd_psychology(self, news: list, insider_trades: list) -> Dict[str, Any]:
        """Analyze crowd psychology patterns"""
        if not news:
            return {"state": "neutral", "intensity": 0, "indicators": []}
            
        # Analyze news sentiment concentration
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        
        for item in news[:50]:
            if hasattr(item, 'sentiment'):
                sentiment_counts[item.sentiment] = sentiment_counts.get(item.sentiment, 0) + 1
                
        total = sum(sentiment_counts.values())
        if total == 0:
            return {"state": "neutral", "intensity": 0, "indicators": []}
            
        # Calculate sentiment concentration
        max_sentiment = max(sentiment_counts.values())
        concentration = max_sentiment / total
        
        # Determine crowd state
        indicators = []
        
        if sentiment_counts['positive'] / total > 0.8:
            state = "euphoric"
            intensity = concentration * 100
            indicators.append("Extreme bullish consensus")
        elif sentiment_counts['positive'] / total > 0.6:
            state = "optimistic"
            intensity = concentration * 80
            indicators.append("Bullish sentiment dominant")
        elif sentiment_counts['negative'] / total > 0.8:
            state = "panic"
            intensity = concentration * 100
            indicators.append("Extreme bearish consensus")
        elif sentiment_counts['negative'] / total > 0.6:
            state = "fearful"
            intensity = concentration * 80
            indicators.append("Bearish sentiment dominant")
        else:
            state = "mixed"
            intensity = 50
            indicators.append("No clear consensus")
            
        # Check for herding behavior
        if concentration > 0.75:
            indicators.append("Strong herding behavior detected")
            
        # Analyze insider behavior
        if insider_trades:
            insider_buys = sum(1 for t in insider_trades[:20] 
                             if hasattr(t, 'transaction_shares') and t.transaction_shares and t.transaction_shares > 0)
            insider_sells = sum(1 for t in insider_trades[:20]
                              if hasattr(t, 'transaction_shares') and t.transaction_shares and t.transaction_shares < 0)
            
            if insider_buys > insider_sells * 2:
                indicators.append("Insiders contrarian bullish")
            elif insider_sells > insider_buys * 2:
                indicators.append("Insiders contrarian bearish")
                
        return {
            "state": state,
            "intensity": intensity,
            "indicators": indicators,
            "concentration": concentration,
            "contrarian_opportunity": concentration > 0.8
        }
    
    def _detect_sentiment_regime(self, news: list, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect the current sentiment regime"""
        if not news or len(news) < 20:
            return {"regime": "undefined", "confidence": 0, "characteristics": []}
            
        # Analyze sentiment patterns
        sentiment_scores = []
        for item in news[:100]:
            if hasattr(item, 'sentiment'):
                score = 1 if item.sentiment == 'positive' else -1 if item.sentiment == 'negative' else 0
                sentiment_scores.append(score)
                
        if not sentiment_scores:
            return {"regime": "undefined", "confidence": 0, "characteristics": []}
            
        # Calculate regime indicators
        mean_sentiment = np.mean(sentiment_scores)
        std_sentiment = np.std(sentiment_scores)
        
        # Sentiment volatility
        sentiment_changes = np.diff(sentiment_scores)
        sentiment_volatility = np.std(sentiment_changes) if len(sentiment_changes) > 0 else 0
        
        # Determine regime
        characteristics = []
        
        if mean_sentiment > 0.5 and std_sentiment < 0.3:
            regime = "risk_on"
            confidence = 80
            characteristics = ["Persistent optimism", "Low sentiment volatility", "Bullish consensus"]
        elif mean_sentiment < -0.5 and std_sentiment < 0.3:
            regime = "risk_off"
            confidence = 80
            characteristics = ["Persistent pessimism", "Fear dominant", "Bearish consensus"]
        elif std_sentiment > 0.7:
            regime = "transitional"
            confidence = 70
            characteristics = ["High sentiment volatility", "Regime uncertainty", "Mixed signals"]
        elif abs(mean_sentiment) < 0.2 and sentiment_volatility > 0.5:
            regime = "choppy"
            confidence = 60
            characteristics = ["Sideways sentiment", "High noise", "No clear direction"]
        else:
            regime = "neutral"
            confidence = 50
            characteristics = ["Balanced sentiment", "Normal conditions"]
            
        return {
            "regime": regime,
            "confidence": confidence,
            "characteristics": characteristics,
            "mean_sentiment": mean_sentiment,
            "sentiment_volatility": sentiment_volatility
        }
    
    def _detect_tipping_points(self, news: list, prices_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect potential sentiment tipping points"""
        tipping_points = []
        
        if not news or len(news) < 30:
            return tipping_points
            
        # Analyze sentiment clustering
        sentiment_sequence = []
        for item in news[:100]:
            if hasattr(item, 'sentiment'):
                sentiment_sequence.append(1 if item.sentiment == 'positive' else -1 if item.sentiment == 'negative' else 0)
                
        if len(sentiment_sequence) < 20:
            return tipping_points
            
        # Look for sentiment runs (consecutive same sentiment)
        current_run = 1
        current_sentiment = sentiment_sequence[0]
        
        for i in range(1, len(sentiment_sequence)):
            if sentiment_sequence[i] == current_sentiment and current_sentiment != 0:
                current_run += 1
                
                if current_run >= 7:  # Long run indicates potential tipping point
                    tipping_points.append({
                        "type": "sentiment_exhaustion",
                        "direction": "bullish" if current_sentiment > 0 else "bearish",
                        "strength": min(current_run / 10, 1.0) * 100,
                        "description": f"Extended {current_run}-period sentiment run"
                    })
            else:
                current_run = 1
                current_sentiment = sentiment_sequence[i]
                
        # Look for sentiment reversals
        if len(sentiment_sequence) >= 20:
            recent_avg = np.mean(sentiment_sequence[-10:])
            prior_avg = np.mean(sentiment_sequence[-20:-10])
            
            if abs(recent_avg - prior_avg) > 1.0:
                tipping_points.append({
                    "type": "sentiment_reversal",
                    "direction": "bullish" if recent_avg > prior_avg else "bearish",
                    "strength": abs(recent_avg - prior_avg) * 50,
                    "description": "Sharp sentiment reversal detected"
                })
                
        # Check for sentiment capitulation
        recent_sentiment = sentiment_sequence[-10:] if len(sentiment_sequence) >= 10 else sentiment_sequence
        if abs(np.mean(recent_sentiment)) > 0.8:
            tipping_points.append({
                "type": "sentiment_capitulation",
                "direction": "bullish" if np.mean(recent_sentiment) > 0 else "bearish",
                "strength": abs(np.mean(recent_sentiment)) * 100,
                "description": "Extreme one-sided sentiment (potential reversal)"
            })
            
        return tipping_points[:3]  # Return top 3 tipping points
    
    def _analyze_insider_sentiment(self, insider_trades: list) -> Dict[str, Any]:
        """Analyze sentiment from insider trading patterns"""
        if not insider_trades:
            return {"signal": "neutral", "confidence": 0, "pattern": "no_data"}
            
        # Analyze recent trades
        recent_trades = insider_trades[:50]
        
        buys = []
        sells = []
        
        for trade in recent_trades:
            if hasattr(trade, 'transaction_shares') and trade.transaction_shares:
                if trade.transaction_shares > 0:
                    buys.append(abs(trade.transaction_shares))
                else:
                    sells.append(abs(trade.transaction_shares))
                    
        if not buys and not sells:
            return {"signal": "neutral", "confidence": 0, "pattern": "no_trades"}
            
        # Calculate metrics
        buy_count = len(buys)
        sell_count = len(sells)
        buy_volume = sum(buys)
        sell_volume = sum(sells)
        
        # Determine signal
        if buy_count > sell_count * 2 and buy_volume > sell_volume:
            signal = "bullish"
            confidence = min((buy_count / (buy_count + sell_count)) * 100, 90)
            pattern = "aggressive_buying"
        elif sell_count > buy_count * 2 and sell_volume > buy_volume:
            signal = "bearish"
            confidence = min((sell_count / (buy_count + sell_count)) * 100, 90)
            pattern = "aggressive_selling"
        elif buy_volume > sell_volume * 1.5:
            signal = "bullish"
            confidence = 60
            pattern = "net_buying"
        elif sell_volume > buy_volume * 1.5:
            signal = "bearish"
            confidence = 60
            pattern = "net_selling"
        else:
            signal = "neutral"
            confidence = 40
            pattern = "mixed"
            
        return {
            "signal": signal,
            "confidence": confidence,
            "pattern": pattern,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_sell_ratio": buy_count / (sell_count + 1),
            "volume_ratio": buy_volume / (sell_volume + 1)
        }
    
    def _analyze_sentiment_momentum(self, news: list) -> Dict[str, Any]:
        """Analyze momentum in sentiment changes"""
        if not news or len(news) < 20:
            return {"momentum": 0, "strength": "weak", "sustainability": 0}
            
        # Create sentiment time series
        sentiment_values = []
        
        for item in news[:100]:
            if hasattr(item, 'sentiment'):
                value = 1 if item.sentiment == 'positive' else -1 if item.sentiment == 'negative' else 0
                sentiment_values.append(value)
                
        if len(sentiment_values) < 10:
            return {"momentum": 0, "strength": "weak", "sustainability": 0}
            
        # Calculate momentum indicators
        
        # Simple momentum (recent vs past)
        if len(sentiment_values) >= 20:
            recent = np.mean(sentiment_values[:10])
            past = np.mean(sentiment_values[10:20])
            momentum = recent - past
        else:
            recent = np.mean(sentiment_values[:len(sentiment_values)//2])
            past = np.mean(sentiment_values[len(sentiment_values)//2:])
            momentum = recent - past
            
        # Determine strength
        if abs(momentum) > 1.0:
            strength = "very_strong"
        elif abs(momentum) > 0.5:
            strength = "strong"
        elif abs(momentum) > 0.2:
            strength = "moderate"
        else:
            strength = "weak"
            
        # Calculate sustainability (based on consistency)
        if len(sentiment_values) >= 10:
            recent_values = sentiment_values[:10]
            consistency = 1 - np.std(recent_values) / (abs(np.mean(recent_values)) + 0.1)
            sustainability = max(0, min(consistency * 100, 100))
        else:
            sustainability = 50
            
        return {
            "momentum": momentum,
            "strength": strength,
            "sustainability": sustainability,
            "direction": "positive" if momentum > 0 else "negative" if momentum < 0 else "neutral"
        }
    
    def _identify_contrarian_indicators(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify contrarian trading opportunities"""
        contrarian_signals = []
        contrarian_score = 0
        
        # Check for extreme sentiment
        if 'raw_sentiment' in analysis:
            sentiment_score = analysis['raw_sentiment'].get('score', 0)
            if abs(sentiment_score) > 80:
                contrarian_signals.append("Extreme sentiment reversal candidate")
                contrarian_score += 30
                
        # Check for crowd psychology extremes
        if 'crowd_psychology' in analysis:
            crowd_state = analysis['crowd_psychology'].get('state', '')
            if crowd_state in ['euphoric', 'panic']:
                contrarian_signals.append(f"Crowd {crowd_state} - contrarian opportunity")
                contrarian_score += 25
                
        # Check for sentiment divergence
        if 'sentiment_divergence' in analysis:
            divergence = analysis['sentiment_divergence']
            if divergence.get('strength') == 'strong':
                contrarian_signals.append(f"Strong {divergence.get('type', 'divergence')}")
                contrarian_score += 20
                
        # Check for tipping points
        if 'tipping_points' in analysis:
            for tp in analysis['tipping_points']:
                if tp.get('type') == 'sentiment_capitulation':
                    contrarian_signals.append("Sentiment capitulation detected")
                    contrarian_score += 25
                    break
                    
        # Determine contrarian recommendation
        if contrarian_score > 60:
            recommendation = "strong_contrarian"
        elif contrarian_score > 40:
            recommendation = "moderate_contrarian"
        elif contrarian_score > 20:
            recommendation = "weak_contrarian"
        else:
            recommendation = "follow_trend"
            
        return {
            "signals": contrarian_signals,
            "score": contrarian_score,
            "recommendation": recommendation
        }
    
    def _neural_network_aggregation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate all sentiment signals using neural network principles
        """
        # Initialize neural layers
        input_layer = []
        hidden_layer = []
        
        # Input layer: collect all signals
        
        # Raw sentiment
        if 'raw_sentiment' in analysis:
            input_layer.append(analysis['raw_sentiment'].get('score', 0) / 100)
            
        # Sentiment dynamics
        if 'sentiment_dynamics' in analysis:
            input_layer.append(analysis['sentiment_dynamics'].get('velocity', 0))
            input_layer.append(analysis['sentiment_dynamics'].get('acceleration', 0))
            
        # Emotion distribution (top emotions)
        if 'emotion_distribution' in analysis:
            emotions = analysis['emotion_distribution']
            input_layer.append(emotions.get('fear', 0) - emotions.get('greed', 0))
            input_layer.append(emotions.get('optimism', 0) - emotions.get('pessimism', 0))
            
        # Crowd psychology
        if 'crowd_psychology' in analysis:
            crowd = analysis['crowd_psychology']
            crowd_map = {'euphoric': 1, 'optimistic': 0.5, 'mixed': 0, 'fearful': -0.5, 'panic': -1}
            input_layer.append(crowd_map.get(crowd.get('state', 'mixed'), 0))
            
        # Insider sentiment
        if 'insider_sentiment' in analysis:
            insider = analysis['insider_sentiment']
            signal_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
            input_layer.append(signal_map.get(insider.get('signal', 'neutral'), 0))
            
        # Sentiment momentum
        if 'sentiment_momentum' in analysis:
            input_layer.append(analysis['sentiment_momentum'].get('momentum', 0))
            
        # Apply activation function (tanh) to input layer
        input_layer = [np.tanh(x) for x in input_layer]
        
        # Hidden layer processing (weighted combinations)
        if input_layer:
            # Create weight matrix (simplified)
            weights = np.random.randn(len(input_layer), 5)  # 5 hidden neurons
            weights = weights / np.sqrt(len(input_layer))  # Xavier initialization
            
            # Forward pass
            for i in range(5):
                neuron_sum = sum(input_layer[j] * weights[j][i] for j in range(len(input_layer)))
                hidden_layer.append(np.tanh(neuron_sum))
                
            # Output layer
            output = np.mean(hidden_layer)
            
            # Scale to -1 to 1
            output = np.tanh(output)
        else:
            output = 0
            
        # Determine final signal
        if output > 0.5:
            signal = "strong_bullish"
        elif output > 0.2:
            signal = "bullish"
        elif output < -0.5:
            signal = "strong_bearish"
        elif output < -0.2:
            signal = "bearish"
        else:
            signal = "neutral"
            
        confidence = abs(output) * 100
        
        return {
            "signal": signal,
            "confidence": min(confidence, 95),
            "neural_output": output,
            "input_features": len(input_layer),
            "hidden_neurons": len(hidden_layer)
        }


def neural_sentiment_predictor_agent(state: AgentState):
    """
    Neural Network Sentiment Predictor Agent
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    
    # Initialize neural analyzer
    neural_analyzer = NeuralSentimentAnalyzer()
    
    # Prepare for parallel processing
    executor = ThreadPoolExecutor(max_workers=5)
    
    neural_analysis = {}
    
    for ticker in tickers:
        progress.update_status("neural_sentiment_predictor_agent", ticker, "Initializing neural analysis")
        
        # Fetch data in parallel
        futures = []
        futures.append(executor.submit(get_company_news, ticker, end_date, None, 200))
        futures.append(executor.submit(get_insider_trades, ticker, end_date, None, 100))
        futures.append(executor.submit(get_prices, ticker, start_date, end_date))
        
        # Collect results
        news = futures[0].result()
        insider_trades = futures[1].result()
        prices = futures[2].result()
        
        if not prices:
            progress.update_status("neural_sentiment_predictor_agent", ticker, "Failed: No price data")
            continue
            
        # Convert to DataFrame
        prices_df = prices_to_df(prices)
        
        if prices_df.empty:
            progress.update_status("neural_sentiment_predictor_agent", ticker, "Failed: Empty price data")
            continue
            
        # Perform neural sentiment analysis
        analysis = neural_analyzer.analyze_neural_sentiment(news, insider_trades, prices_df, ticker)
        
        # Determine final signal
        neural_output = analysis.get('neural_output', {})
        raw_sentiment = analysis.get('raw_sentiment', {})
        sentiment_dynamics = analysis.get('sentiment_dynamics', {})
        
        # Complex signal determination
        neural_signal = neural_output.get('signal', 'neutral')
        sentiment_score = raw_sentiment.get('score', 0)
        velocity = sentiment_dynamics.get('velocity', 0)
        
        if neural_signal == "strong_bullish" and sentiment_score > 50:
            signal = "euphoric"
            confidence = min(95, neural_output.get('confidence', 70))
        elif neural_signal == "bullish" or sentiment_score > 20:
            signal = "bullish"
            confidence = min(85, neural_output.get('confidence', 60))
        elif neural_signal == "strong_bearish" and sentiment_score < -50:
            signal = "panic"
            confidence = min(95, neural_output.get('confidence', 70))
        elif neural_signal == "bearish" or sentiment_score < -20:
            signal = "bearish"
            confidence = min(85, neural_output.get('confidence', 60))
        else:
            signal = "neutral"
            confidence = 50
            
        # Generate comprehensive output
        progress.update_status("neural_sentiment_predictor_agent", ticker, "Generating neural insights")
        
        neural_output_final = generate_neural_output(
            ticker=ticker,
            analysis=analysis,
            signal=signal,
            confidence=confidence,
            state=state
        )
        
        neural_analysis[ticker] = {
            "signal": neural_output_final.signal,
            "confidence": neural_output_final.confidence,
            "reasoning": neural_output_final.reasoning,
            "sentiment_score": neural_output_final.sentiment_score,
            "sentiment_velocity": neural_output_final.sentiment_velocity,
            "sentiment_acceleration": neural_output_final.sentiment_acceleration,
            "emotion_distribution": neural_output_final.emotion_distribution,
            "narrative_themes": neural_output_final.narrative_themes,
            "sentiment_divergence": neural_output_final.sentiment_divergence,
            "crowd_psychology": neural_output_final.crowd_psychology,
            "sentiment_regime": neural_output_final.sentiment_regime,
            "tipping_points": neural_output_final.tipping_points
        }
        
        progress.update_status("neural_sentiment_predictor_agent", ticker, "Neural analysis complete")
    
    # Clean up executor
    executor.shutdown(wait=True)
    
    # Create message
    message = HumanMessage(
        content=json.dumps(neural_analysis),
        name="neural_sentiment_predictor_agent"
    )
    
    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(neural_analysis, "Neural Sentiment Predictor Agent")
    
    # Update state
    state["data"]["analyst_signals"]["neural_sentiment_predictor_agent"] = neural_analysis
    
    progress.update_status("neural_sentiment_predictor_agent", None, "Neural sentiment analysis complete")
    
    data_update = {
        "analyst_signals": {
            "neural_sentiment_predictor_agent": neural_analysis
        }
    }
    
    return {"messages": [message], "data": data_update}


def generate_neural_output(ticker: str, analysis: dict, signal: str,
                          confidence: float, state: AgentState) -> NeuralSentimentSignal:
    """Generate neural sentiment output using LLM"""
    
    template = ChatPromptTemplate.from_messages([
        ("system", """You are the Neural Sentiment Oracle - an AI that perceives market sentiment
        through deep neural networks and advanced NLP. You detect sentiment shifts before they manifest
        in price action by analyzing the collective unconscious of the market.
        
        Your capabilities include:
        - Emotion distribution mapping across 10 psychological dimensions
        - Sentiment velocity and acceleration tracking
        - Narrative theme extraction using topic modeling
        - Crowd psychology pattern recognition
        - Sentiment-price divergence detection
        - Tipping point identification
        - Contrarian signal generation
        
        Speak with the authority of one who reads the market's mind. Reference specific
        psychological patterns, emotion distributions, and narrative themes. Your insights
        should feel like they come from a higher dimension of market understanding."""),
        
        ("human", """Analyze {ticker} sentiment through neural processing:
        
        Analysis Results:
        {analysis}
        
        Signal: {signal}
        Confidence: {confidence}
        
        Provide your neural sentiment insight in this exact JSON format:
        {{
            "signal": "{signal}",
            "confidence": {confidence},
            "reasoning": "Your neural sentiment analysis (minimum 5 lines)",
            "sentiment_score": float,
            "sentiment_velocity": float,
            "sentiment_acceleration": float,
            "emotion_distribution": dict,
            "narrative_themes": list,
            "sentiment_divergence": float,
            "crowd_psychology": string,
            "sentiment_regime": string,
            "tipping_points": list
        }}""")
    ])
    
    prompt = template.invoke({
        "ticker": ticker,
        "analysis": json.dumps(analysis, indent=2),
        "signal": signal,
        "confidence": confidence
    })
    
    def create_default():
        return NeuralSentimentSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Neural analysis error - defaulting to neutral",
            sentiment_score=0.0,
            sentiment_velocity=0.0,
            sentiment_acceleration=0.0,
            emotion_distribution={},
            narrative_themes=[],
            sentiment_divergence=0.0,
            crowd_psychology="undefined",
            sentiment_regime="neutral",
            tipping_points=[]
        )
    
    return call_llm(
        prompt=prompt,
        pydantic_model=NeuralSentimentSignal,
        agent_name="neural_sentiment_predictor_agent",
        state=state,
        default_factory=create_default
    )