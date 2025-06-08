"""SQL transformation implementation using sqlglot."""

from typing import Any, Dict, List
import pandas as pd
import sqlglot
from ..interfaces import DatasetTransformer, Dataset
from ..models import DatasetItem, DatasetMetadata
from ..base_dataset import BaseDataset


class SQLTransformer(DatasetTransformer):
    """Transform datasets using SQL expressions."""

    def transform(self, dataset: Dataset, sql_expression: str, **kwargs) -> Dataset:
        """Transform dataset using SQL expression."""
        # Convert dataset to DataFrame for SQL operations
        df = self._dataset_to_dataframe(dataset)
        
        # Parse and execute SQL
        filtered_df = self._execute_sql(df, sql_expression)
        
        # Convert back to dataset
        return self._dataframe_to_dataset(filtered_df, dataset.metadata)

    def _dataset_to_dataframe(self, dataset: Dataset) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame."""
        data = []
        for item in dataset:
            # Flatten the item structure
            row = {**item.input}
            if item.expected_output:
                row.update({f'expected_{k}': v for k, v in item.expected_output.items()})
            data.append(row)
        return pd.DataFrame(data)

    def _execute_sql(self, df: pd.DataFrame, sql_expression: str) -> pd.DataFrame:
        """Execute SQL expression on DataFrame."""
        # For now, implement basic WHERE filtering
        # This is a simplified implementation
        if 'WHERE' in sql_expression.upper():
            return self._apply_where_filter(df, sql_expression)
        return df

    def _apply_where_filter(self, df: pd.DataFrame, sql_expression: str) -> pd.DataFrame:
        """Apply WHERE clause filtering."""
        # Simple implementation for common cases
        # Extract WHERE clause
        where_clause = sql_expression.upper().split('WHERE')[1].strip()
        
        # Handle simple equality filters
        if '=' in where_clause and 'LIKE' not in where_clause:
            column, value = where_clause.split('=', 1)
            column = column.strip(); column = self._find_column(df, column)
            value = value.strip().strip("'\"")
            return df[df[column] == value]
        
        return df

    def _dataframe_to_dataset(self, df: pd.DataFrame, original_metadata: DatasetMetadata) -> Dataset:
        """Convert DataFrame back to dataset."""
        items = []
        for _, row in df.iterrows():
            # Separate input and expected_output fields
            input_data = {}
            expected_data = {}
            
            for col, val in row.items():
                if col.startswith('expected_'):
                    expected_data[col[9:]] = val
                else:
                    input_data[col] = val
            
            items.append(DatasetItem(
                input=input_data,
                expected_output=expected_data if expected_data else None
            ))
        
        return BaseDataset(items, original_metadata)

    def _find_column(self, df: pd.DataFrame, column_name: str) -> str:
        """Find column name case-insensitively."""
        for col in df.columns:
            if col.lower() == column_name.lower():
                return col
        raise KeyError(f'Column {column_name} not found')
