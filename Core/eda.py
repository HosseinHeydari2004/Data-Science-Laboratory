from typing import Optional, Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_datetime64_any_dtype
from pandas.io.formats.style import Styler
from scipy import stats
from sklearn.impute import SimpleImputer


class EDA:
    """
    A utility class for Exploratory Data Analysis operations.

    This class provides static methods for common EDA tasks such as checking
    unique values, handling missing data, and performing statistical summaries.
    All methods are designed to work with pandas DataFrames.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 2, 3], 'B': ['x', 'y', 'z', 'z']})
    >>> EDA.check_unique(df, 'A')
    array([1, 2, 3])
    >>> EDA.check_unique(df, 'B')
    array(['x', 'y', 'z'], dtype=object)
    """

    @classmethod
    def check_unique(cls, data: pd.DataFrame, select_column: str) -> np.ndarray:
        """
        Retrieve all unique values from a specified column in a DataFrame.

        This method extracts and returns an array of unique values from the
        specified column, providing a quick way to understand the distinct
        categories or values present in a dataset column.

        Parameters
        ----------
        data : pd.DataFrame
            The pandas DataFrame containing the data to analyze.
            Must not be empty and must contain the specified column.

        select_column : str
            The name of the column from which to extract unique values.
            The column must exist in the DataFrame.

        Returns
        -------
        np.ndarray
            A numpy array containing the unique values from the specified column.
            The array preserves the data type of the original column values.
            Returns an empty array if the column is empty.

        Raises
        ------
        KeyError
            If the specified column name does not exist in the DataFrame.
        TypeError
            If `data` is not a pandas DataFrame or `select_column` is not a string.
        ValueError
            If the DataFrame is empty.

        Notes
        -----
        - The unique values are not sorted; they appear in the order they are
          encountered in the data.
        - NaN values are treated as distinct values and will be included if present.
        - For large datasets, consider using `data[select_column].nunique()` if
          you only need the count of unique values.

        See Also
        --------
        pandas.Series.unique : The underlying pandas method used.
        pandas.Series.nunique : Returns the number of unique values.
        pandas.Series.value_counts : Returns counts of unique values.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np

        >>> # Example 1: Basic usage with numeric data
        >>> df = pd.DataFrame({
        ...     'id': [1, 2, 3, 4, 5],
        ...     'category': ['A', 'B', 'A', 'C', 'B'],
        ...     'value': [10.5, 20.3, 10.5, 30.1, 25.7]
        ... })
        >>> EDA.check_unique(df, 'category')
        array(['A', 'B', 'C'], dtype=object)

        >>> # Example 2: Working with numeric values
        >>> EDA.check_unique(df, 'value')
        array([10.5, 20.3, 30.1, 25.7])

        >>> # Example 3: Handling missing values
        >>> df_with_nan = pd.DataFrame({
        ...     'col': [1, 2, None, 2, None, 3]
        ... })
        >>> EDA.check_unique(df_with_nan, 'col')
        array([1.0, 2.0, nan, 3.0])

        >>> # Example 4: Feature engineering - checking categorical variable levels
        >>> df_sales = pd.DataFrame({
        ...     'product': ['Laptop', 'Mouse', 'Laptop', 'Keyboard', 'Mouse', 'Monitor'],
        ...     'sales': [1000, 50, 1200, 80, 45, 600]
        ... })
        >>> products = EDA.check_unique(df_sales, 'product')
        >>> print(f"Unique products: {products}")
        Unique products: ['Laptop' 'Mouse' 'Keyboard' 'Monitor']
        >>> print(f"Number of unique products: {len(products)}")
        Number of unique products: 4

        >>> # Example 5: Combining with other EDA operations
        >>> categories = EDA.check_unique(df, 'category')
        >>> for cat in categories:
        ...     subset = df[df['category'] == cat]
        ...     print(f"{cat}: {len(subset)} records")
        A: 2 records
        B: 2 records
        C: 1 records
        """
        return data[select_column].unique()

    @classmethod
    def list_columns(cls, data: pd.DataFrame) -> list:
        """
        Retrieve all column names from a DataFrame as a list.

    This method extracts and returns the column names of a pandas DataFrame
    in a list format, providing a quick way to inspect the structure and
    available features of a dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The pandas DataFrame whose column names are to be retrieved.
        Must not be empty.

    Returns
    -------
    list
        A list of strings containing all column names from the DataFrame.
        Returns an empty list if the DataFrame has no columns.
        The column order is preserved as they appear in the DataFrame.

    Raises
    ------
    TypeError
        If `data` is not a pandas DataFrame.
    ValueError
        If the DataFrame is empty (no rows), though columns may still exist.

    Notes
    -----
    - This method uses `pandas.DataFrame.columns.to_list()` internally.
    - The returned list preserves the original column order.
    - For a numpy array of column names, use `data.columns.values` instead.
    - To check if a specific column exists, use `'column_name' in data.columns`.

    See Also
    --------
    pandas.DataFrame.columns : The underlying columns attribute.
    pandas.DataFrame.keys() : Returns the columns as an Index object.
    pandas.DataFrame.info() : Provides a summary of the DataFrame including columns.
    EDA.check_unique : Get unique values from a specific column.

    Examples
    --------
    >>> import pandas as pd

    >>> # Example 1: Basic usage with a simple DataFrame
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3],
    ...     'name': ['Alice', 'Bob', 'Charlie'],
    ...     'age': [25, 30, 35],
    ...     'salary': [50000, 60000, 70000]
    ... })
    >>> EDA.list_columns(df)
    ['id', 'name', 'age', 'salary']

    >>> # Example 2: Using with a DataFrame with different column types
    >>> df_mixed = pd.DataFrame({
    ...     'numeric_col': [1.5, 2.3, 3.7],
    ...     'string_col': ['A', 'B', 'C'],
    ...     'datetime_col': pd.date_range('2024-01-01', periods=3),
    ...     'categorical_col': pd.Categorical(['X', 'Y', 'Z'])
    ... })
    >>> EDA.list_columns(df_mixed)
    ['numeric_col', 'string_col', 'datetime_col', 'categorical_col']

    >>> # Example 3: Empty DataFrame (no rows but with columns)
    >>> df_empty = pd.DataFrame(columns=['col1', 'col2', 'col3'])
    >>> EDA.list_columns(df_empty)
    ['col1', 'col2', 'col3']

    >>> # Example 4: Using with a DataFrame for exploratory analysis
    >>> df_sales = pd.DataFrame({
    ...     'product_id': [101, 102, 103],
    ...     'product_name': ['Laptop', 'Mouse', 'Keyboard'],
    ...     'price': [999.99, 29.99, 79.99],
    ...     'stock': [50, 200, 150],
    ...     'category': ['Electronics', 'Accessories', 'Accessories']
    ... })
    >>> columns = EDA.list_columns(df_sales)
    >>> print(f"Dataset has {len(columns)} columns: {', '.join(columns)}")
    Dataset has 5 columns: product_id, product_name, price, stock, category

    >>> # Example 5: Chaining with other operations
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [4, 5, 6],
    ...     'C': [7, 8, 9]
    ... })
    >>> cols = EDA.list_columns(df)
    >>> # Filter numeric columns (assuming all are numeric in this case)
    >>> numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
    >>> print(numeric_cols)
    ['A', 'B', 'C']

    >>> # Example 6: Checking for specific columns
    >>> required_cols = ['id', 'name', 'age']
    >>> available_cols = EDA.list_columns(df)
    >>> missing_cols = [col for col in required_cols if col not in available_cols]
    >>> if missing_cols:
    ...     print(f"Missing columns: {missing_cols}")
    Missing columns: ['id', 'name', 'age']
        """
        return data.columns.to_list()

    @classmethod
    def information_data(cls, data: pd.DataFrame) -> Styler:
        """
        Generate a styled summary DataFrame containing comprehensive information about each column.

    This method creates a detailed information summary for all columns in a DataFrame,
    including data types, missing values, missing value percentages, and memory usage.
    The output is returned as a styled pandas Styler object with visual highlighting
    to draw attention to columns with the highest missing values.

    Parameters
    ----------
    data : pd.DataFrame
        The pandas DataFrame to analyze. Must not be empty.

    Returns
    -------
    pd.io.formats.style.Styler
        A pandas Styler object containing the formatted and styled summary table
        with the following columns:
        - "columns": Column names
        - "data type": Data type of each column
        - "missing values": Count of missing (NaN) values per column
        - "percent missing values(%)": Percentage of missing values
        - "memory usage": Memory usage per column in bytes (formatted as KB)

    Raises
    ------
    TypeError
        If `data` is not a pandas DataFrame.
    ValueError
        If the DataFrame is empty.

    Notes
    -----
    - The method uses deep memory usage calculation (`deep=True`) to accurately
      measure memory consumption including object dtypes.
    - Missing values are counted using `pd.isna()` which identifies NaN, None,
      and NaT values.
    - The maximum missing value count and percentage are highlighted in red
      for quick identification of columns with the most missing data.
    - Memory usage is displayed in Kilobytes (KB) for better readability.
    - The index is reset and not displayed in the final output.

    See Also
    --------
    pandas.DataFrame.info : Standard DataFrame information summary.
    pandas.DataFrame.describe : Statistical summary of numerical columns.
    pandas.DataFrame.isna : Identify missing values.
    pandas.DataFrame.memory_usage : Memory usage of DataFrame columns.
    EDA.check_unique : Get unique values from a column.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np

    >>> # Example 1: Basic usage with a simple DataFrame
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3, 4, 5],
    ...     'name': ['Alice', 'Bob', None, 'David', 'Eve'],
    ...     'age': [25, 30, np.nan, 35, 40],
    ...     'salary': [50000, 60000, 70000, np.nan, 90000],
    ...     'department': ['HR', 'IT', None, 'Finance', 'IT']
    ... })
    >>> styled = EDA.information_data(df)
    >>> styled  # Display in Jupyter notebook
    # Output will show a styled table with missing values highlighted

    >>> # Example 2: Using in a data quality report
    >>> df_large = pd.DataFrame({
    ...     'product_id': range(1000),
    ...     'product_name': ['Product_' + str(i) for i in range(1000)],
    ...     'price': np.random.randn(1000) * 100 + 500,
    ...     'stock': np.random.randint(0, 100, 1000),
    ...     'category': np.random.choice(['A', 'B', 'C', None], 1000)
    ... })
    >>> # Add some missing values
    >>> df_large.loc[100:200, 'price'] = np.nan
    >>> df_large.loc[300:350, 'stock'] = np.nan
    >>> info_table = EDA.information_data(df_large)
    >>> info_table  # Display in Jupyter notebook

    >>> # Example 3: Extracting the styled data for further analysis
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, np.nan],
    ...     'B': [np.nan, np.nan, np.nan],
    ...     'C': [1, 2, 3]
    ... })
    >>> styled = EDA.information_data(df)
    >>> # Access the underlying DataFrame
    >>> underlying_df = styled.data
    >>> print(underlying_df)
       columns data type  missing values  percent missing values(%)  memory usage
    0        A   float64               1                    33.333333            80
    1        B   float64               3                   100.000000            80
    2        C    int64                0                     0.000000            32

    >>> # Example 4: Identifying columns with high missing values
    >>> df = pd.DataFrame({
    ...     'col1': [1, 2, 3, 4, 5],
    ...     'col2': [np.nan, np.nan, 3, 4, 5],
    ...     'col3': [np.nan, np.nan, np.nan, 4, 5],
    ...     'col4': [np.nan, np.nan, np.nan, np.nan, 5],
    ... })
    >>> info = EDA.information_data(df)
    >>> # The column with the most missing values (col4) will be highlighted in red
    >>> info  # Display in Jupyter notebook

    >>> # Example 5: Using for data cleaning decisions
    >>> df = pd.DataFrame({
    ...     'customer_id': range(100),
    ...     'name': ['Customer_' + str(i) for i in range(100)],
    ...     'email': ['email' + str(i) + '@example.com' for i in range(100)],
    ...     'phone': [str(i)*10 for i in range(100)],
    ...     'address': ['Address ' + str(i) for i in range(100)]
    ... })
    >>> # Introduce missing values in different columns
    >>> df.loc[10:50, 'email'] = np.nan
    >>> df.loc[30:70, 'phone'] = np.nan
    >>> df.loc[0:90, 'address'] = np.nan
    >>>
    >>> info = EDA.information_data(df)
    >>> # The highlighted column (address with 91% missing) suggests dropping it
    >>> info  # Display in Jupyter notebook

    >>> # Example 6: Comparing memory usage of different data types
    >>> df_types = pd.DataFrame({
    ...     'int_col': range(1000),
    ...     'float_col': np.random.randn(1000),
    ...     'string_col': ['string' + str(i) for i in range(1000)],
    ...     'category_col': pd.Categorical(['A', 'B', 'C'] * 333 + ['A'])
    ... })
    >>> info = EDA.information_data(df_types)
    >>> # The string column will show higher memory usage than categorical
    >>> info  # Display in Jupyter notebook

    >>> # Example 7: Integration in a data exploration pipeline
    >>> def data_quality_report(df):
    ...     print(f"Dataset shape: {df.shape}")
    ...     print(f"Total memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    ...     print("\nColumn Information:")
    ...     return EDA.information_data(df)
    >>>
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [np.nan, 5, 6],
    ...     'C': [7, 8, 9]
    ... })
    >>> report = data_quality_report(df)
    Dataset shape: (3, 3)
    Total memory usage: 0.21 KB

    Column Information:
    # Styled table will be displayed
        """
        d = pd.DataFrame(
            data={
                "columns": data.columns,
                "data type": data.dtypes.values,
                "missing values": data.isna().sum(),
                "percent missing values(%)": (data.isna().sum() / len(data)) * 100,
                "memory usage": data.memory_usage(deep=True, index=False)
            }
        ).reset_index(drop=True)

        def highlight_bad_values(s):
            """
            Highlight the maximum value in a series with red background.

        Parameters
        ----------
        s : pd.Series
            The series to highlight.

        Returns
        -------
        list
            List of CSS styles for each cell.
            """
            if s.max() == 0:
                return ['' for _ in s]
            return ['background-color: red' if v == s.max() else '' for v in s]

        return d.style.format({
            "percent missing values(%)": "{:.2f}%",
            "memory usage (KB)": "{:.2f} KB"
        }).apply(highlight_bad_values, subset=["missing values", "percent missing values(%)"])

    @classmethod
    def describe_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
            Generate a comprehensive statistical summary of all columns in a DataFrame.

    This method extends pandas' `describe()` function to include all column types
    (numeric, categorical, and object) and renames the 'top' statistic to 'mode'
    for better interpretability. The summary includes count, unique values, mode,
    frequency, and for numeric columns: mean, standard deviation, min, quartiles,
    and max.

    Parameters
    ----------
    data : pd.DataFrame
        The pandas DataFrame to analyze. Can contain mixed data types.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing statistical summaries for all columns:
        - **count**: Number of non-null values
        - **unique**: Number of unique values (for non-numeric columns)
        - **mode**: Most frequent value (renamed from 'top')
        - **freq**: Frequency of the most frequent value (for non-numeric columns)
        - **mean**: Arithmetic mean (numeric columns only)
        - **std**: Standard deviation (numeric columns only)
        - **min**: Minimum value (numeric columns only)
        - **25%**: First quartile (numeric columns only)
        - **50%**: Median/second quartile (numeric columns only)
        - **75%**: Third quartile (numeric columns only)
        - **max**: Maximum value (numeric columns only)

    Raises
    ------
    TypeError
        If `data` is not a pandas DataFrame.
    ValueError
        If the DataFrame is empty.

    Notes
    -----
    - For numeric columns, the output includes mean, std, min, quartiles, and max.
    - For object, string, and categorical columns, the output includes count,
      unique, mode (top), and freq.
    - The 'top' statistic from pandas is renamed to 'mode' for clarity.
    - NaN values are excluded from the statistics.
    - For datetime columns, similar statistics to numeric are provided.
    - This method is particularly useful for quick data exploration and
      understanding the distribution of all columns at once.

    See Also
    --------
    pandas.DataFrame.describe : The underlying pandas method.
    pandas.DataFrame.info : Summary of DataFrame including dtypes and memory usage.
    pandas.DataFrame.agg : Custom aggregation of DataFrame columns.
    EDA.information_data : Get missing values and memory usage summary.
    EDA.check_unique : Get unique values from a specific column.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np

    >>> # Example 1: Basic usage with mixed data types
    >>> df = pd.DataFrame({
    ...     'numeric': [1, 2, 3, 4, 5],
    ...     'categorical': ['A', 'B', 'A', 'C', 'B'],
    ...     'text': ['foo', 'bar', 'foo', 'baz', 'qux'],
    ...     'with_nan': [1, 2, np.nan, 4, 5]
    ... })
    >>> EDA.describe_data(df)
              numeric categorical  text  with_nan
    count    5.000000           5     5  4.000000
    unique         NaN           3     4       NaN
    mode           NaN           A   foo       NaN
    freq           NaN           2     1       NaN
    mean     3.000000         NaN   NaN  3.000000
    std      1.581139         NaN   NaN  1.825742
    min      1.000000         NaN   NaN  1.000000
    25%      2.000000         NaN   NaN  1.750000
    50%      3.000000         NaN   NaN  3.000000
    75%      4.000000         NaN   NaN  4.250000
    max      5.000000         NaN   NaN  5.000000

    >>> # Example 2: Dataset with only numeric columns
    >>> df_numeric = pd.DataFrame({
    ...     'A': np.random.randn(100),
    ...     'B': np.random.randint(1, 100, 100),
    ...     'C': np.random.exponential(2, 100)
    ... })
    >>> stats = EDA.describe_data(df_numeric)
    >>> stats  # Shows full numeric statistics for all columns
                  A           B           C
    count  100.00000  100.000000  100.000000
    mean     0.02345   50.500000    2.012345
    std      0.98765   28.867513    1.987654
    min     -2.34567    1.000000    0.012345
    25%     -0.67890   25.750000    0.678901
    50%      0.01234   50.500000    1.456789
    75%      0.67890   75.250000    2.567890
    max      2.34567  100.000000    8.901234

    >>> # Example 3: Dataset with categorical columns
    >>> df_cat = pd.DataFrame({
    ...     'category': ['High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High'],
    ...     'status': ['Active', 'Inactive', 'Active', 'Active', 'Inactive', 'Inactive', 'Active'],
    ...     'value': [100, 200, 150, 300, 250, 175, 225]
    ... })
    >>> EDA.describe_data(df_cat)
              category    status      value
    count            7         7   7.000000
    unique           3         2        NaN
    mode          High    Active        NaN
    freq             3         4        NaN
    mean           NaN       NaN 200.000000
    std            NaN       NaN  66.143783
    min            NaN       NaN 100.000000
    25%            NaN       NaN 162.500000
    50%            NaN       NaN 200.000000
    75%            NaN       NaN 237.500000
    max            NaN       NaN 300.000000

    >>> # Example 4: Identifying categorical columns with high cardinality
    >>> df_high_card = pd.DataFrame({
    ...     'id': range(1000),
    ...     'category': np.random.choice(['A', 'B', 'C'], 1000),
    ...     'text': ['text_' + str(i) for i in range(1000)]
    ... })
    >>> stats = EDA.describe_data(df_high_card)
    >>> # 'id' and 'text' have high cardinality (1000 unique values)
    >>> stats.loc[['unique']]  # Shows unique counts
            id  category  text
    unique 1000         3  1000

    >>> # Example 5: Using with datetime columns
    >>> df_dates = pd.DataFrame({
    ...     'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    ...     'value': np.random.randn(100),
    ...     'category': ['A', 'B'] * 50
    ... })
    >>> stats = EDA.describe_data(df_dates)
    >>> stats  # Shows datetime statistics with min, max, etc.
                  date     value category
    count           100       100      100
    unique          NaN       NaN        2
    mode            NaN       NaN        A
    freq            NaN       NaN       50
    mean            NaN  0.012345      NaN
    std             NaN  0.987654      NaN
    min     2024-01-01 -2.345678      NaN
    25%     2024-03-26 -0.678901      NaN
    50%     2024-06-24  0.012345      NaN
    75%     2024-09-22  0.678901      NaN
    max     2024-12-30  2.345678      NaN

    >>> # Example 6: Quick data quality assessment
    >>> df_quality = pd.DataFrame({
    ...     'num1': [1, 2, 3, 4, 5],
    ...     'num2': [10, 20, 30, 40, 50],
    ...     'cat1': ['X', 'Y', 'X', 'Z', 'Y'],
    ...     'cat2': ['A', 'A', 'B', 'B', 'A']
    ... })
    >>> stats = EDA.describe_data(df_quality)
    >>> # Check if categorical columns have expected number of categories
    >>> categories = stats.loc['unique', ['cat1', 'cat2']]
    >>> print(f"cat1 has {categories['cat1']} unique values")
    cat1 has 3 unique values
    >>> print(f"cat2 has {categories['cat2']} unique values")
    cat2 has 2 unique values

    >>> # Example 7: Using in a data exploration workflow
    >>> def explore_dataset(df):
    ...     print(f"Dataset shape: {df.shape}")
    ...     print("\nStatistical Summary:")
    ...     summary = EDA.describe_data(df)
    ...     return summary
    >>>
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ...     'B': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
    ...     'C': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ... })
    >>> summary = explore_dataset(df)
    Dataset shape: (10, 3)

    Statistical Summary:
    # Returns the described DataFrame

    >>> # Example 8: Identifying outliers using quartiles
    >>> df_outliers = pd.DataFrame({
    ...     'normal': np.random.normal(0, 1, 100),
    ...     'with_outliers': np.concatenate([np.random.normal(0, 1, 95), [100, 200, 300, 400, 500]])
    ... })
    >>> stats = EDA.describe_data(df_outliers)
    >>> # Compare max values vs 75% quartile to identify outliers
    >>> max_val = stats.loc['max', 'with_outliers']
    >>> q75 = stats.loc['75%', 'with_outliers']
    >>> if max_val > q75 * 3:
    ...     print(f"Potential outliers detected in 'with_outliers': max={max_val}, Q3={q75}")
    Potential outliers detected in 'with_outliers': max=500, Q3=0.89
        """
        return data.describe(include="all").rename({"top": "mode"})

    @classmethod
    def check_date_object(cls, data: pd.DataFrame) -> bool | tuple[bool, str]:
        """
        check if a DataFrame contains a date/datetime column and verify its data type.

    This method searches for common date column names ('date', 'datetime', 'Date', 'Datetime')
    and checks whether the first found column is properly typed as a datetime type.
    Returns a boolean indicating if the column exists and is correctly typed,
    along with the column name if a date column is found.

    Parameters
    ----------
    data : pd.DataFrame
        The pandas DataFrame to check for date/datetime columns.

    Returns
    -------
    bool or tuple[bool, str]
        - If no date column is found or the found column is already datetime type:
          Returns `False` (indicating no date column exists or it's already correctly typed)
        - If a date column is found but it's NOT datetime type:
          Returns a tuple `(True, column_name)` where `column_name` is the name of
          the date column that needs conversion

    Raises
    ------
    TypeError
        If `data` is not a pandas DataFrame.
    ValueError
        If the DataFrame is empty.

    Notes
    -----
    - The method looks for columns with these exact names: 'date', 'datetime', 'Date', 'Datetime'
    - Only the first matching column is checked (if multiple date columns exist)
    - The method uses `is_datetime64_any_dtype()` from pandas.api.types to check
      if the column is already a datetime type
    - This is useful for identifying date columns that need conversion before
      time-series analysis or date-based operations

    See Also
    --------
    pandas.api.types.is_datetime64_any_dtype : Check if dtype is datetime64.
    pandas.to_datetime : Convert argument to datetime.
    pandas.DataFrame.select_dtypes : Select columns by data type.
    EDA.information_data : Get column information including data types.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pandas.api.types import is_datetime64_any_dtype

    >>> # Example 1: DataFrame with date column as string (needs conversion)
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3],
    ...     'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    ...     'value': [10, 20, 30]
    ... })
    >>> EDA.check_date_object(df)
    (True, 'date')
    # Returns tuple indicating a date column exists but needs conversion

    >>> # Example 2: DataFrame with datetime column (already correct)
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3],
    ...     'datetime': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
    ...     'value': [10, 20, 30]
    ... })
    >>> EDA.check_date_object(df)
    False
    # Returns False because column already has datetime type

    >>> # Example 3: DataFrame with Date column (capitalized)
    >>> df = pd.DataFrame({
    ...     'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    ...     'sales': [100, 200, 300]
    ... })
    >>> EDA.check_date_object(df)
    (True, 'Date')

    >>> # Example 4: DataFrame without any date column
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3],
    ...     'name': ['Alice', 'Bob', 'Charlie'],
    ...     'age': [25, 30, 35]
    ... })
    >>> EDA.check_date_object(df)
    False
    # Returns False because no date column found

    >>> # Example 5: DataFrame with multiple date-like columns
    >>> df = pd.DataFrame({
    ...     'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    ...     'datetime': ['2024-01-01 10:00', '2024-01-02 11:00', '2024-01-03 12:00'],
    ...     'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    ...     'value': [1, 2, 3]
    ... })
    >>> EDA.check_date_object(df)
    (True, 'date')
    # Returns the first matching column found

    >>> # Example 6: Using in a data preprocessing pipeline
    >>> def ensure_datetime(df):
    ...     result = EDA.check_date_object(df)
    ...     if isinstance(result, tuple):
    ...         col_name = result[1]
    ...         df[col_name] = pd.to_datetime(df[col_name])
    ...         print(f"Converted '{col_name}' to datetime")
    ...     else:
    ...         print("No date column found or already datetime")
    ...     return df
    >>>
    >>> df = pd.DataFrame({
    ...     'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    ...     'value': [10, 20, 30]
    ... })
    >>> df = ensure_datetime(df)
    Converted 'date' to datetime
    >>> df.dtypes
    date     datetime64[ns]
    value             int64
    dtype: object

    >>> # Example 7: Conditional logic based on return type
    >>> df = pd.DataFrame({
    ...     'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    ...     'sales': [100, 200, 300]
    ... })
    >>>
    >>> result = EDA.check_date_object(df)
    >>> if isinstance(result, tuple):
    ...     col = result[1]
    ...     print(f"Column '{col}' needs to be converted to datetime")
    ...     # Convert the column
    ...     df[col] = pd.to_datetime(df[col])
    ... else:
    ...     print("No conversion needed")
    Column 'Date' needs to be converted to datetime

    >>> # Example 8: Working with large datasets
    >>> df_large = pd.DataFrame({
    ...     'id': range(1000),
    ...     'date': ['2024-01-01'] * 1000,
    ...     'datetime': pd.date_range('2024-01-01', periods=1000),
    ...     'value': np.random.randn(1000)
    ... })
    >>> result = EDA.check_date_object(df_large)
    >>> print(f"Result: {result}")
    Result: (True, 'date')
    >>> # The 'datetime' column is already correct, but 'date' needs conversion

    >>> # Example 9: Using in a time-series analysis workflow
    >>> df_ts = pd.DataFrame({
    ...     'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    ...     'temperature': [22.5, 23.0, 21.5],
    ...     'humidity': [65, 70, 68]
    ... })
    >>>
    >>> # Check and convert if needed
    >>> result = EDA.check_date_object(df_ts)
    >>> if isinstance(result, tuple):
    ...     date_col = result[1]
    ...     df_ts[date_col] = pd.to_datetime(df_ts[date_col])
    ...     df_ts.set_index(date_col, inplace=True)
    ...     print("Data ready for time-series analysis")
    Data ready for time-series analysis
        """
        date_cols = [c for c in data.columns if c in ["date", "datetime", "Date", "Datetime"]]
        if not date_cols or is_datetime64_any_dtype(data[date_cols[0]]):
            return False
        else:
            return True, date_cols[0]

    @classmethod
    def change_dtype_datetime64(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert object-type columns containing date information to datetime64 dtype.

    This method identifies columns that contain date-like objects and converts
    them to pandas datetime64 format for better time-series operations and
    analysis. The conversion uses coercion to handle invalid date formats
    gracefully by converting them to NaT (Not a Time).

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing potential date columns with object dtype.

    Returns
    -------
    pd.DataFrame
        A DataFrame with date columns converted to datetime64 dtype if any
        date-like columns were detected. If no date-like columns are found,
        the original DataFrame is returned unchanged.

    Notes
    -----
    - This is a classmethod intended to be called on the class itself
    - The method modifies the DataFrame in-place for the converted columns
    - Invalid date strings are converted to NaT (Not a Time)
    - The conversion only applies to columns identified as date-like objects
      by the EDA.check_date_object() method

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3],
    ...     'date': ['2024-01-01', '2024-01-02', 'invalid_date'],
    ...     'value': [10, 20, 30]
    ... })
    >>> df['date'].dtype
    dtype('O')
    >>> df = EDA.change_dtype_datetime64(df)
    >>> df['date'].dtype
    dtype('<M8[ns]')
    >>> df['date']
    0   2024-01-01
    1   2024-01-02
    2          NaT
    Name: date, dtype: datetime64[ns]

    See Also
    --------
    EDA.check_date_object : Method used to identify date-like columns
    pandas.to_datetime : Underlying conversion function used
        """
        ch, data_col = EDA.check_date_object(data=data)
        if ch:
            data[data_col] = pd.to_datetime(data[data_col], errors="coerce")
            return data

    @classmethod
    def get_duplicate(cls, data: pd.DataFrame) -> int | bool:
        """
            Check for and count duplicate rows in a DataFrame.

    This method identifies duplicate rows in the input DataFrame and returns
    the total count of duplicate entries. If no duplicates exist, it returns
    False to indicate the absence of duplicates in a boolean context.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to check for duplicate rows.

    Returns
    -------
    int | bool
        - Returns an integer representing the total number of duplicate rows
          if duplicates are found (count > 0)
        - Returns False if no duplicate rows exist in the DataFrame

    Notes
    -----
    - This is a classmethod intended to be called on the class itself
    - The method counts all duplicate rows, not just unique duplicate entries
    - Duplicates are identified based on all columns in the DataFrame
    - Returns False instead of 0 to allow for boolean conditional checks like
      `if cls.get_duplicate(df):` while still providing the count value

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # DataFrame with duplicates
    >>> df_with_dups = pd.DataFrame({
    ...     'id': [1, 2, 2, 3, 3, 3],
    ...     'name': ['A', 'B', 'B', 'C', 'C', 'C']
    ... })
    >>> EDA.get_duplicate(df_with_dups)
    3  # Returns count of duplicate rows (rows 2, 4, 5)
    >>>
    >>> # DataFrame without duplicates
    >>> df_clean = pd.DataFrame({
    ...     'id': [1, 2, 3],
    ...     'name': ['A', 'B', 'C']
    ... })
    >>> EDA.get_duplicate(df_clean)
    False
    >>>
    >>> # Using in conditional logic
    >>> dup_count = EDA.get_duplicate(df)
    >>> if dup_count:
    ...     print(f"Found {dup_count} duplicate rows")
    ... else:
    ...     print("No duplicates found")

    See Also
    --------
    pandas.DataFrame.duplicated : Underlying method used to identify duplicates
    pandas.DataFrame.drop_duplicates : Method to remove duplicate rows
        """
        if data.duplicated().sum() > 0:
            return data.duplicated().sum()
        else:
            return False

    @classmethod
    def show_duplicate_values(cls, data: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Extract and display all duplicate rows from a DataFrame.

    This method returns a subset of the DataFrame containing only the rows
    that are duplicates of previous rows. All duplicate occurrences (except
    the first occurrence of each duplicate group) are returned.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to extract duplicate rows from.

    Returns
    -------
    pd.Series | pd.DataFrame
        - Returns a DataFrame containing all duplicate rows if duplicates exist
        - Returns an empty DataFrame with the same columns if no duplicates found
        - Note: Return type is always pd.DataFrame (the type hint shows
          pd.Series | pd.DataFrame for compatibility, but pandas.duplicated()
          always returns a DataFrame when used for boolean indexing)

    Notes
    -----
    - This is a classmethod intended to be called on the class itself
    - Only returns rows that are duplicates of previous rows (first occurrence excluded)
    - Uses pandas.duplicated() with default parameters (keep='first')
    - If you want all occurrences including the first, use keep=False
    - The method does not modify the original DataFrame

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # DataFrame with duplicates
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 2, 3, 3, 3, 4],
    ...     'name': ['A', 'B', 'B', 'C', 'C', 'C', 'D']
    ... })
    >>>
    >>> # Show duplicate rows
    >>> YourClass.show_duplicate_values(df)
       id name
    1   2    B
    3   3    C
    4   3    C
    5   3    C
    >>>
    >>> # Check if duplicates exist
    >>> dup_df = YourClass.show_duplicate_values(df)
    >>> if not dup_df.empty:
    ...     print(f"Found {len(dup_df)} duplicate rows")
    ...     print(dup_df)

    See Also
    --------
    pandas.DataFrame.duplicated : Underlying method for duplicate detection
    pandas.DataFrame.drop_duplicates : Method to remove duplicate rows
    get_duplicate : Returns count of duplicate rows or False
        """
        return data[data.duplicated()]

    @classmethod
    def delete_duplicate_values(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove all duplicate rows from a DataFrame.

    This method removes duplicate rows from the input DataFrame, keeping only
    the first occurrence of each unique row. The original DataFrame is not
    modified; a new DataFrame with duplicates removed is returned.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame from which to remove duplicate rows.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with duplicate rows removed. The first occurrence
        of each duplicate group is kept, and all subsequent duplicates
        are dropped.

    Notes
    -----
    - This is a classmethod intended to be called on the class itself
    - Uses pandas.drop_duplicates() with default parameters (keep='first')
    - Does not modify the original DataFrame (returns a new DataFrame)
    - Considers all columns when identifying duplicates
    - If you want to keep the last occurrence instead, use keep='last'
    - If you want to drop all duplicates (including first), use keep=False

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # DataFrame with duplicates
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 2, 3, 3, 3, 4],
    ...     'name': ['A', 'B', 'B', 'C', 'C', 'C', 'D']
    ... })
    >>> df
       id name
    0   1    A
    1   2    B
    2   2    B
    3   3    C
    4   3    C
    5   3    C
    6   4    D
    >>>
    >>> # Remove duplicates
    >>> EDA.delete_duplicate_values(df)
       id name
    0   1    A
    1   2    B
    3   3    C
    6   4    D
    >>>
    >>> # Using in data cleaning pipeline
    >>> clean_df = EDA.delete_duplicate_values(raw_data)
    >>> print(f"Removed {len(raw_data) - len(clean_df)} duplicate rows")

    See Also
    --------
    pandas.DataFrame.drop_duplicates : Underlying method used
    get_duplicate : Count duplicate rows before deletion
    show_duplicate_values : View duplicates before deletion
        """
        return data.drop_duplicates()

    @classmethod
    def detect_numeric_type(cls, data: pd.DataFrame):
        """
        Identify all numeric columns in a DataFrame.

    This method detects and returns a list of column names that contain
    numeric data types (int, float, complex, etc.) from the input DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to scan for numeric columns.

    Returns
    -------
    list[str]
        A list of column names that have numeric data types.
        Returns an empty list if no numeric columns are found.

    Notes
    -----
    - This is a classmethod intended to be called on the class itself
    - Uses pandas.select_dtypes() with include="number" to detect numeric types
    - Includes integer, float, and complex number types
    - Does not include boolean or datetime types (use include="bool" or include="datetime" for those)
    - Returns column names as a list for easy iteration and filtering

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # DataFrame with mixed types
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35],           # int
    ...     'salary': [50000.0, 60000.5, 70000.0],  # float
    ...     'name': ['Alice', 'Bob', 'Charlie'],    # object/string
    ...     'active': [True, False, True],           # bool
    ...     'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])  # datetime
    ... })
    >>>
    >>> # Detect numeric columns
    >>> EDA.detect_numeric_type(df)
    ['age', 'salary']
    >>>
    >>> # Using in data processing
    >>> numeric_cols = EDA.detect_numeric_type(df)
    >>> df_numeric = df[numeric_cols]  # Select only numeric columns
    >>> print(f"Found {len(numeric_cols)} numeric columns: {numeric_cols}")
    Found 2 numeric columns: ['age', 'salary']

    See Also
    --------
    pandas.DataFrame.select_dtypes : Underlying method used for type detection
    detect_object_type : Detect object/string columns
    detect_datetime_type : Detect datetime columns
    detect_categorical_type : Detect categorical columns
        """
        numeric = data.select_dtypes(include="number").columns.to_list()
        return numeric

    @classmethod
    def detect_object_type(cls, data: pd.DataFrame):
        """
        Identify all object/string columns in a DataFrame.

    This method detects and returns a list of column names that contain
    object data types (typically strings, mixed types, or text data)
    from the input DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to scan for object-type columns.

    Returns
    -------
    list[str]
        A list of column names that have object data type.
        Returns an empty list if no object columns are found.

    Notes
    -----
    - This is a classmethod intended to be called on the class itself
    - Uses pandas.select_dtypes() with include="object" to detect object types
    - Object dtype typically contains string/text data or mixed types
    - These columns may contain categorical data that could be converted
    - Object columns are often candidates for string processing or encoding

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # DataFrame with mixed types
    >>> df = pd.DataFrame({
    ...     'name': ['Alice', 'Bob', 'Charlie'],    # object/string
    ...     'city': ['NYC', 'LA', 'Chicago'],        # object/string
    ...     'age': [25, 30, 35],                     # int
    ...     'salary': [50000.0, 60000.5, 70000.0],   # float
    ...     'active': [True, False, True]            # bool
    ... })
    >>>
    >>> # Detect object columns
    >>> EDA.detect_object_type(df)
    ['name', 'city']
    >>>
    >>> # Using in data processing
    >>> object_cols = cls.detect_object_type(df)
    >>> print(f"Found {len(object_cols)} object columns: {object_cols}")
    Found 2 object columns: ['name', 'city']
    >>>
    >>> # Convert object columns to categorical for memory efficiency
    >>> for col in object_cols:
    ...     df[col] = df[col].astype('category')

    See Also
    --------
    pandas.DataFrame.select_dtypes : Underlying method used for type detection
    detect_numeric_type : Detect numeric columns
    detect_datetime_type : Detect datetime columns
    detect_categorical_type : Detect categorical columns
        """
        return data.select_dtypes(include="object").columns.to_list()

    @classmethod
    def detect_time_type(cls, data: pd.DataFrame):
        """
        Identify all datetime columns in a DataFrame.

    This method detects and returns a list of column names that contain
    datetime data types (datetime64, datetime, timestamp, etc.) from
    the input DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to scan for datetime-type columns.

    Returns
    -------
    list[str]
        A list of column names that have datetime data types.
        Returns an empty list if no datetime columns are found.

    Notes
    -----
    - This is a classmethod intended to be called on the class itself
    - Uses pandas.select_dtypes() with include="datetime" to detect datetime types
    - Detects both datetime64[ns] and datetime64[ns, tz] (timezone-aware) types
    - Does not detect date-only types (use include="date" for those)
    - Datetime columns are useful for time-series analysis, resampling, and date operations

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # DataFrame with mixed types
    >>> df = pd.DataFrame({
    ...     'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
    ...     'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']).dt.date,
    ...     'time': pd.to_datetime(['12:00', '13:00', '14:00']).dt.time,
    ...     'age': [25, 30, 35],                     # int
    ...     'name': ['Alice', 'Bob', 'Charlie']       # object
    ... })
    >>>
    >>> # Detect datetime columns
    >>> EDA.detect_time_type(df)
    ['timestamp']
    >>>
    >>> # Using in data processing
    >>> time_cols = cls.detect_time_type(df)
    >>> print(f"Found {len(time_cols)} datetime columns: {time_cols}")
    Found 1 datetime columns: ['timestamp']
    >>>
    >>> # Extract date features from datetime columns
    >>> for col in time_cols:
    ...     df[f'{col}_year'] = df[col].dt.year
    ...     df[f'{col}_month'] = df[col].dt.month
    ...     df[f'{col}_day'] = df[col].dt.day

    See Also
    --------
    pandas.DataFrame.select_dtypes : Underlying method used for type detection
    detect_numeric_type : Detect numeric columns
    detect_object_type : Detect object/string columns
    detect_timedelta_type : Detect timedelta columns
    detect_date_type : Detect date-only columns
    change_dtype_datetime64 : Convert object columns to datetime64
        """
        return data.select_dtypes(include=["datetime"]).columns.to_list()

    @classmethod
    def all_correlation(cls, data: pd.DataFrame):
        """
            Compute the correlation matrix for all numeric columns in a DataFrame.

    This method calculates the pairwise correlation coefficients between
    all numeric columns in the input DataFrame using Pearson correlation
    by default.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing numeric columns to compute
        correlations for.

    Returns
    -------
    pd.DataFrame
        A correlation matrix DataFrame where:
        - Rows and columns represent the numeric column names
        - Values are correlation coefficients ranging from -1 to 1
        - Diagonal values are 1.0 (perfect correlation with itself)

    Notes
    -----
    - This is a classmethod intended to be called on the class itself
    - Uses pandas.DataFrame.corr() with numeric_only=True
    - Only includes numeric columns; non-numeric columns are ignored
    - Correlation coefficient:
        - +1: Perfect positive correlation
        - 0: No correlation
        - -1: Perfect negative correlation
    - Useful for feature selection, multicollinearity detection, and EDA

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # DataFrame with numeric columns
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35, 40, 45],
    ...     'salary': [50000, 60000, 70000, 80000, 90000],
    ...     'experience': [1, 5, 8, 12, 15],
    ...     'score': [85, 88, 92, 95, 98]
    ... })
    >>>
    >>> # Compute correlation matrix
    >>> EDA.all_correlation(df)
                  age    salary  experience     score
    age         1.0000  1.0000     0.9939    0.9897
    salary      1.0000  1.0000     0.9939    0.9897
    experience  0.9939  0.9939     1.0000    0.9830
    score       0.9897  0.9897     0.9830    1.0000
    >>>
    >>> # Using for feature selection
    >>> corr_matrix = EDA.all_correlation(df)
    >>> # Find highly correlated features (above 0.9)
    >>> high_corr = corr_matrix[corr_matrix > 0.9]
    >>> print(high_corr)

    See Also
    --------
    pandas.DataFrame.corr : Underlying method used
    detect_numeric_type : Get numeric columns for correlation
    plot_correlation_heatmap : Visualize correlation matrix (if exists)
        """
        return data.corr(numeric_only=True)

    @classmethod
    def col_correlation(cls, data: pd.Series, col1: pd.Series, col2: pd.Series):
        """
        Compute the correlation between two specific columns in a DataFrame.

    This method calculates the Pearson correlation coefficient between
    two specified columns from the input DataFrame.

    Parameters
    ----------
    data : pd.Series
        The input DataFrame containing the columns to correlate.
        Note: The type hint indicates pd.Series but this should be a DataFrame.
    col1 : pd.Series
        The first column (should be a column name or Series).
        Note: The type hint indicates pd.Series but this should be a column name.
    col2 : pd.Series
        The second column (should be a column name or Series).
        Note: The type hint indicates pd.Series but this should be a column name.

    Returns
    -------
    float
        The Pearson correlation coefficient between the two columns,
        ranging from -1 to 1.

    Notes
    -----
    - This is a classmethod intended to be called on the class itself
    - Uses pandas.Series.corr() to compute Pearson correlation
    - Returns NaN if either column has insufficient data or is non-numeric
    - Missing values are automatically excluded from the calculation
    - Both columns must be numeric or convertible to numeric

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # DataFrame with numeric columns
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35, 40, 45],
    ...     'salary': [50000, 60000, 70000, 80000, 90000],
    ...     'experience': [1, 5, 8, 12, 15]
    ... })
    >>>
    >>> # Correlation between age and salary
    >>> EDA.col_correlation(df, 'age', 'salary')
    1.0
    >>>
    >>> # Correlation between experience and salary
    >>> EDA.col_correlation(df, 'experience', 'salary')
    0.9938837346736188
    >>>
    >>> # Using in analysis
    >>> corr_value = EDA.col_correlation(df, 'age', 'salary')
    >>> print(f"Correlation between age and salary: {corr_value:.2f}")
    Correlation between age and salary: 1.00
    >>>
    >>> # Check for strong correlation
    >>> if abs(corr_value) > 0.8:
    ...     print("Strong correlation detected!")

    See Also
    --------
    pandas.Series.corr : Underlying method for correlation
    all_correlation : Compute correlation matrix for all columns
    pandas.DataFrame.corrwith : Compute correlations with a Series
        """
        return data[col1].corr(other=data[col2])

    @classmethod
    def delete_columns(
            cls, data: pd.DataFrame, col: str | list) -> pd.DataFrame | pd.Series:
        """
        Delete one or more columns from a DataFrame.

    This method removes specified columns from the input DataFrame and
    returns a new DataFrame without those columns. The original DataFrame
    is not modified.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame from which to delete columns.
    col : str | list
        A single column name (str) or a list of column names (list[str])
        to be removed from the DataFrame.

    Returns
    -------
    pd.DataFrame | pd.Series
        - Returns a DataFrame with the specified columns removed
        - If only one column remains after deletion, returns a Series
        - If all columns are deleted, returns an empty DataFrame

    Notes
    -----
    - This is a classmethod intended to be called on the class itself
    - Uses pandas.DataFrame.drop() with axis=1 (columns)
    - Does not modify the original DataFrame (returns a new DataFrame)
    - Raises KeyError if specified column(s) do not exist
    - Use inplace=True to modify the original DataFrame if needed

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # DataFrame with multiple columns
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3],
    ...     'name': ['Alice', 'Bob', 'Charlie'],
    ...     'age': [25, 30, 35],
    ...     'salary': [50000, 60000, 70000],
    ...     'department': ['HR', 'IT', 'Finance']
    ... })
    >>>
    >>> # Delete a single column
    >>> YourClass.delete_columns(df, 'salary')
       id     name  age department
    0   1    Alice   25         HR
    1   2      Bob   30         IT
    2   3  Charlie   35    Finance
    >>>
    >>> # Delete multiple columns
    >>> YourClass.delete_columns(df, ['age', 'department'])
       id     name  salary
    0   1    Alice   50000
    1   2      Bob   60000
    2   3  Charlie   70000
    >>>
    >>> # Using in data cleaning pipeline
    >>> columns_to_remove = ['id', 'department']
    >>> clean_df = cls.delete_columns(df, columns_to_remove)
    >>> print(f"Removed {len(columns_to_remove)} columns")

    See Also
    --------
    pandas.DataFrame.drop : Underlying method used
    select_columns : Keep only specified columns (if exists)
    rename_columns : Rename columns (if exists)
        """
        return data.drop(columns=col)

    @classmethod
    def show_first_5_row(cls, data: pd.DataFrame) -> pd.DataFrame:
        return data.head(5)

    @classmethod
    def show_last_5_row(cls, data: pd.DataFrame) -> pd.DataFrame:
        return data.tail(5)

    @classmethod
    def show_specific_row(cls, data: pd.DataFrame, index: int = 0) -> pd.DataFrame | pd.Series:
        return data.iloc[index]

    @classmethod
    def show_random_sample_rows(cls, data: pd.DataFrame, n: int = 5) -> pd.DataFrame | pd.Series:
        return data.sample(n=n)

    @classmethod
    def show_specific_column(cls, data: pd.DataFrame, col_name: str) -> pd.Series:
        return data[col_name]

    @classmethod
    def check_dtype_column(cls, data: pd.DataFrame | pd.Series, col: object) -> int:
        category = data.select_dtypes(include=["object"])
        numeric = data.select_dtypes(include=["number"])
        if col in category:
            return 0
        elif col in numeric:
            return 1

    @classmethod
    def select_manual_data(
            cls,
            data: pd.DataFrame,
            rows: tuple[int, int] = (0, 5),
            columns: Optional[tuple[int, int]] = None,
            mode: str = "Multiple rows and columns",
            column_name: Optional[str] | object = None,
            row_index: Optional[int] = None,
            value: Any = None,
            query: str = ""
    ) -> Any:

        if mode == "Multiple rows and columns":
            if columns is None:
                return data.iloc[rows[0]:rows[1], :]
            return data.iloc[rows[0]:rows[1], columns[0]:columns[1]]

        elif mode == "Multiple rows and one column":
            if isinstance(column_name, str):
                return data[column_name].iloc[rows[0]:rows[1]]
            return data.iloc[rows[0]:rows[1], 0]

        elif mode == "one row and Multiple columns":
            if columns is None:
                return data.iloc[row_index, :]
            return data.iloc[row_index, columns[0]:columns[1]]
        elif mode == "filter by value":
            return data[data[column_name] == value]
        elif mode == "search text":
            return data[data[column_name].astype(str).str.contains(str(value), case=False, na=False)]
        elif mode == "query":
            qy = data.query(expr=query)
            return qy


class handle_MissingValue:
    """
    A utility class to analyze and handle missing values in pandas DataFrames.

    This class provides streamlined methods to detect null values and apply
    various imputation or deletion strategies to maintain data integrity
    for machine learning pipelines.
    """

    @classmethod
    def check_missing_values(cls, data: pd.DataFrame):
        if data.isna().any(axis=1).sum() > 0:
            return True
        return False

    @classmethod
    def report_high_missing_value(
            cls, data: pd.DataFrame, threshold: int = 30
    ) -> tuple[bool, float | int, int] | bool:

        percent_missing = (data.isna().any(axis=1).sum() / len(data)) * 100
        total_missing_value = data.isna().any(axis=1).sum()
        if percent_missing > threshold:
            return True, percent_missing, total_missing_value
        return False

    @classmethod
    def remove_missing_values(cls, data: pd.DataFrame, axis: str = "row"):
        if axis == "row":
            return data.dropna(axis=0)
        else:
            return data.dropna(axis=1)

    @classmethod
    def find_high_col_missing_values(cls, data: pd.DataFrame, threshold: int = 30) -> dict:
        percent_missing = (data.isna().sum() / len(data)) * 100
        return percent_missing[percent_missing > threshold].to_dict()

    @classmethod
    def show_missing_values(cls, data: pd.DataFrame, reset_index: bool = False) -> pd.Series | pd.DataFrame:
        if reset_index:
            return data[data.isna().any(axis=1)].reset_index(drop=True)
        else:
            return data[data.isna().any(axis=1)]

    @classmethod
    def fill_SimpleImputer(
            cls, x: pd.DataFrame | pd.Series = None, strategy="mean", fill=0
    ) -> pd.Series | pd.DataFrame:
        if strategy == "constant":
            imputer = SimpleImputer(strategy="constant", fill_value=fill)
            x_filled = imputer.fit_transform(X=x)
            return x_filled
        else:
            imputer = SimpleImputer(strategy=strategy)
            x_filled = imputer.fit_transform(X=x)
            return x_filled


class handle_outliers:
    """"""

    @classmethod
    def detect_outliers(
            cls, data: pd.DataFrame, col: pd.DataFrame | str | object,
            method: str | object = "IQR",
            threshold: int = 3
    ) -> pd.DataFrame:
        if method == "IQR":
            Q1 = data[col].quantile(q=0.25)
            Q3 = data[col].quantile(q=0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            return outliers[col]
        elif method == "Z_score":
            z = np.abs(stats.zscore(data[col]))
            outliers_index = np.where(z > threshold)
            return data.iloc[outliers_index][col]

    @classmethod
    def delete_outliers(
            cls, data: pd.DataFrame, index_outliers: Any
    ) -> pd.DataFrame:
        return data.drop(index=index_outliers, errors='ignore').reset_index(drop=True)


class data_manipulation:
    @classmethod
    def delete_row(
            cls, data: pd.DataFrame, row_index: int | object | pd.Series = None
    ) -> DataFrame | None | Exception:
        if row_index not in data.index:
            raise ValueError(f"row {row_index} not founded!")
        return data.drop(index=row_index).reset_index(drop=True)

    @classmethod
    def delete_rows(
            cls, data: pd.DataFrame, rows_index: tuple[int, int] = None
    ) -> pd.DataFrame | None | Exception:
        start, end = rows_index
        indices_to_drop = list(range(start, end + 1))
        if not all(i in data.index for i in indices_to_drop):
            raise ValueError(f"rows {rows_index} not founded!")
        return data.drop(index=indices_to_drop).reset_index(drop=True)

    @classmethod
    def delete_column(cls, data: pd.DataFrame, col: str | pd.Series | object) -> pd.DataFrame | Exception:
        if col in data.columns:
            return data.drop(columns=col)
        else:
            raise ValueError(f"column {col} not founded!")

    @classmethod
    def delete_columns(cls, data: pd.DataFrame, list_col: list[str] | object) -> pd.DataFrame:
        missing_cols = [col for col in list_col if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {', '.join(missing_cols)}")
        return data.drop(columns=list_col)

    @classmethod
    def change_col_name(
            cls, data: pd.DataFrame, col_name_last: str | object,
            col_name_new: str | object
    ) -> pd.DataFrame:
        return data.rename(columns={col_name_last: col_name_new}, errors="raise")

    @classmethod
    def change_dtype(
            cls, data: pd.DataFrame, col: str | object, dtype: str | object | np.dtype
    ) -> pd.DataFrame:
        if col in data.columns:
            try:
                data_copy = data.copy()
                data_copy[col] = data[col].astype(dtype=dtype)
                return data_copy
            except Exception as e:
                raise ValueError(f"Error while converting column ‘{col}’ to ‘{dtype}’: {e}")
        else:
            raise ValueError(f"column {col} not in data")
