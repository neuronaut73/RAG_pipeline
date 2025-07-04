#!/usr/bin/env python3
"""
setup_validator.py - Centralized Setup Validation Utility

Provides validation functions to ensure the RAG pipeline only uses confirmed setups
from setups.csv and proper date filtering for historical data.
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SetupValidator:
    """
    Centralized setup validation for the RAG pipeline
    
    Ensures that only confirmed setups from setups.csv are used
    and provides date validation for historical data filtering.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the setup validator
        
        Args:
            data_dir: Directory containing the setups.csv file
        """
        self.data_dir = Path(data_dir)
        self.setups_df = None
        self.confirmed_setup_ids = set()
        self.setup_dates = {}
        
        self.load_confirmed_setups()
    
    def load_confirmed_setups(self) -> None:
        """Load confirmed setups from setups.csv"""
        setups_file = self.data_dir / "setups.csv"
        
        if not setups_file.exists():
            raise FileNotFoundError(f"setups.csv not found at {setups_file}")
        
        try:
            self.setups_df = pd.read_csv(setups_file)
            logger.info(f"Loaded {len(self.setups_df)} confirmed setups from setups.csv")
            
            # Extract setup IDs and dates
            for _, row in self.setups_df.iterrows():
                setup_id = row.get('setup_id', '')
                setup_date = row.get('spike_timestamp', '')
                
                if setup_id and setup_date:
                    self.confirmed_setup_ids.add(setup_id)
                    # Parse date - handle different formats
                    try:
                        if isinstance(setup_date, str):
                            # Try different date formats
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                                try:
                                    parsed_date = datetime.strptime(setup_date, fmt)
                                    self.setup_dates[setup_id] = parsed_date
                                    break
                                except ValueError:
                                    continue
                        else:
                            self.setup_dates[setup_id] = pd.to_datetime(setup_date)
                    except Exception as e:
                        logger.warning(f"Could not parse date for setup {setup_id}: {setup_date}")
                        continue
            
            logger.info(f"Confirmed setup IDs: {sorted(self.confirmed_setup_ids)}")
            logger.info(f"Setup dates parsed: {len(self.setup_dates)}")
            
        except Exception as e:
            logger.error(f"Error loading confirmed setups: {e}")
            raise
    
    def is_confirmed_setup(self, setup_id: str) -> bool:
        """
        Check if a setup_id is in the confirmed setups
        
        Args:
            setup_id: The setup ID to validate
            
        Returns:
            True if the setup is confirmed, False otherwise
        """
        return setup_id in self.confirmed_setup_ids
    
    def get_confirmed_setup_ids(self) -> Set[str]:
        """
        Get all confirmed setup IDs
        
        Returns:
            Set of confirmed setup IDs
        """
        return self.confirmed_setup_ids.copy()
    
    def get_setup_date(self, setup_id: str) -> Optional[datetime]:
        """
        Get the setup date for a given setup_id
        
        Args:
            setup_id: The setup ID
            
        Returns:
            datetime object of the setup date, or None if not found
        """
        return self.setup_dates.get(setup_id)
    
    def is_historical_data(self, setup_id: str, data_date: str) -> bool:
        """
        Check if data is historical (before the setup date)
        
        Args:
            setup_id: The setup ID
            data_date: The date of the data (string format)
            
        Returns:
            True if data is historical (before setup date), False otherwise
        """
        if not self.is_confirmed_setup(setup_id):
            return False
        
        setup_date = self.get_setup_date(setup_id)
        if not setup_date:
            return False
        
        try:
            # Parse data date - handle different formats
            if isinstance(data_date, str):
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S.%f']:
                    try:
                        parsed_data_date = datetime.strptime(data_date, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # Try pandas parsing as fallback
                    parsed_data_date = pd.to_datetime(data_date)
            else:
                parsed_data_date = pd.to_datetime(data_date)
            
            # Data is historical if it's before the setup date
            return parsed_data_date < setup_date
            
        except Exception as e:
            logger.warning(f"Could not parse data date {data_date}: {e}")
            return False
    
    def filter_labels_by_confirmed_setups(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter labels DataFrame to only include confirmed setups
        
        Args:
            labels_df: DataFrame containing labels with setup_id column
            
        Returns:
            Filtered DataFrame with only confirmed setups
        """
        if 'setup_id' not in labels_df.columns:
            logger.warning("labels DataFrame does not contain setup_id column")
            return labels_df
        
        initial_count = len(labels_df)
        filtered_df = labels_df[labels_df['setup_id'].isin(self.confirmed_setup_ids)]
        final_count = len(filtered_df)
        
        logger.info(f"Filtered labels: {initial_count} -> {final_count} records "
                   f"(removed {initial_count - final_count} non-confirmed setups)")
        
        return filtered_df
    
    def filter_fundamentals_by_date(self, fundamentals_df: pd.DataFrame, 
                                   setup_id: str, date_column: str = 'period_end') -> pd.DataFrame:
        """
        Filter fundamentals data to only include historical data before setup date
        
        Args:
            fundamentals_df: DataFrame containing fundamentals data
            setup_id: The setup ID to filter for
            date_column: Column name containing the date information
            
        Returns:
            Filtered DataFrame with only historical data
        """
        if not self.is_confirmed_setup(setup_id):
            logger.warning(f"Setup {setup_id} is not confirmed")
            return pd.DataFrame()
        
        if date_column not in fundamentals_df.columns:
            logger.warning(f"Date column {date_column} not found in fundamentals data")
            return fundamentals_df
        
        setup_date = self.get_setup_date(setup_id)
        if not setup_date:
            logger.warning(f"No setup date found for {setup_id}")
            return fundamentals_df
        
        initial_count = len(fundamentals_df)
        
        # Filter to only include data before setup date
        try:
            fundamentals_df[date_column] = pd.to_datetime(fundamentals_df[date_column])
            filtered_df = fundamentals_df[fundamentals_df[date_column] < setup_date]
            final_count = len(filtered_df)
            
            logger.info(f"Filtered fundamentals for {setup_id}: {initial_count} -> {final_count} records "
                       f"(removed {initial_count - final_count} non-historical records)")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering fundamentals by date: {e}")
            return fundamentals_df
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about confirmed setups
        
        Returns:
            Dictionary containing summary statistics
        """
        if self.setups_df is None:
            return {}
        
        stats = {
            'total_confirmed_setups': len(self.confirmed_setup_ids),
            'confirmed_setup_ids': sorted(list(self.confirmed_setup_ids)),
            'unique_tickers': list(self.setups_df['lse_ticker'].unique()) if 'lse_ticker' in self.setups_df.columns else [],
            'date_range': {
                'earliest': self.setups_df['spike_timestamp'].min() if 'spike_timestamp' in self.setups_df.columns else None,
                'latest': self.setups_df['spike_timestamp'].max() if 'spike_timestamp' in self.setups_df.columns else None
            }
        }
        
        return stats


def get_setup_validator(data_dir: str = "data") -> SetupValidator:
    """
    Factory function to get a setup validator instance
    
    Args:
        data_dir: Directory containing the setups.csv file
        
    Returns:
        SetupValidator instance
    """
    return SetupValidator(data_dir=data_dir)


# Global instance for convenience
_validator_instance = None

def get_confirmed_setup_ids(data_dir: str = "data") -> Set[str]:
    """
    Convenience function to get confirmed setup IDs
    
    Args:
        data_dir: Directory containing the setups.csv file
        
    Returns:
        Set of confirmed setup IDs
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = SetupValidator(data_dir=data_dir)
    return _validator_instance.get_confirmed_setup_ids()


def is_confirmed_setup(setup_id: str, data_dir: str = "data") -> bool:
    """
    Convenience function to check if a setup is confirmed
    
    Args:
        setup_id: The setup ID to validate
        data_dir: Directory containing the setups.csv file
        
    Returns:
        True if the setup is confirmed, False otherwise
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = SetupValidator(data_dir=data_dir)
    return _validator_instance.is_confirmed_setup(setup_id)


if __name__ == "__main__":
    # Test the setup validator - use correct data directory path
    validator = SetupValidator(data_dir="../data")
    stats = validator.get_summary_stats()
    
    print("Setup Validator Summary:")
    print(f"Total confirmed setups: {stats['total_confirmed_setups']}")
    print(f"Confirmed setup IDs: {stats['confirmed_setup_ids']}")
    print(f"Unique tickers: {stats['unique_tickers']}")
    print(f"Date range: {stats['date_range']}")
    
    # Test validation
    test_setup = "BGO_2024-11-22"
    print(f"\nTesting setup validation for {test_setup}:")
    print(f"Is confirmed: {validator.is_confirmed_setup(test_setup)}")
    print(f"Setup date: {validator.get_setup_date(test_setup)}")
    
    # Test historical data check
    test_date = "2024-11-20"
    print(f"Is {test_date} historical for {test_setup}: {validator.is_historical_data(test_setup, test_date)}") 