class Cache:
    """in-memory storage system for API responses to avoid repeated network calls for the same data"""

    def __init__(self):
        self._prices_cache: dict[str, list[dict[str, any]]] = {}
        self._financial_metrics_cache: dict[str, list[dict[str, any]]] = {}
        self._line_items_cache: dict[str, list[dict[str, any]]] = {}
        self._insider_trades_cache: dict[str, list[dict[str, any]]] = {}
        self._company_news_cache: dict[str, list[dict[str, any]]] = {}
        
        
    """This function merges new data into existing cached data while avoiding duplicates based on a specific unique field 
        (called key_field) It tells the unique identifies field (like time report period etc).
        "|" Stands For OR so either arguments can be this OR that """
    def _merge_data(self, existing: list[dict] | None, new_data: list[dict], key_field: str) -> list[dict]:

        if not existing: #if exising is none meaning no prev record so just return the new data as it is
            return new_data

        # This creates a set of values for the key_field from all existing records
        existing_keys = {item[key_field] for item in existing}

        """This goes through every item in new_data and checks if its key_field value is already in the set existing_keys.
            If it's not a duplicate, we add it to merged."""
        merged = existing.copy()
        merged.extend([item for item in new_data if item[key_field] not in existing_keys])
        return merged #Returns Combined Records(Old + New)

    """If The Price for that particular ticker already exists then it will return those values else will return None"""
    def get_prices(self, ticker: str) -> list[dict[str, any]] | None:
        """Get cached price data if available."""
        return self._prices_cache.get(ticker)
    
    """Cache Price Data For The Given Ticker"""
    def set_prices(self, ticker: str, data: list[dict[str, any]]):
        """Append new price data to cache."""
        self._prices_cache[ticker] = self._merge_data(self._prices_cache.get(ticker), data, key_field="time")

    """Get cached financial metrics if available."""
    def get_financial_metrics(self, ticker: str) -> list[dict[str, any]]:
        return self._financial_metrics_cache.get(ticker)

    """Append new financial metrics to cache."""
    def set_financial_metrics(self, ticker: str, data: list[dict[str, any]]):
        self._financial_metrics_cache[ticker] = self._merge_data(self._financial_metrics_cache.get(ticker), data, key_field="report_period")
    
    """Get cached line items if available."""
    def get_line_items(self, ticker: str) -> list[dict[str, any]] | None:
        
        return self._line_items_cache.get(ticker)

    """Append new line items to cache."""
    def set_line_items(self, ticker: str, data: list[dict[str, any]]):
        
        self._line_items_cache[ticker] = self._merge_data(self._line_items_cache.get(ticker), data, key_field="report_period")

    """Get cached insider trades if available."""
    def get_insider_trades(self, ticker: str) -> list[dict[str, any]] | None:
        
        return self._insider_trades_cache.get(ticker)

    """Append new insider trades to cache."""
    def set_insider_trades(self, ticker: str, data: list[dict[str, any]]):
        
        self._insider_trades_cache[ticker] = self._merge_data(self._insider_trades_cache.get(ticker), data, key_field="filing_date")  # Could also use transaction_date if preferred

    """Get cached company news if available."""
    def get_company_news(self, ticker: str) -> list[dict[str, any]] | None:
        return self._company_news_cache.get(ticker)

    """Append new company news to cache."""
    def set_company_news(self, ticker: str, data: list[dict[str, any]]):
        
        self._company_news_cache[ticker] = self._merge_data(self._company_news_cache.get(ticker), data, key_field="date")



_cache = Cache()

def get_cache() -> Cache:
    """Get the global cache instance."""
    return _cache
