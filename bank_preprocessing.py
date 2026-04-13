"""
Bank Marketing Data Preprocessing Script
=========================================
Author: Senior Python Data Engineer & Data Scientist
Description: Complete preprocessing pipeline for bank.csv dataset
             including data type conversion, target analysis, and unknown value detection.

This script is designed to be production-ready and can be committed directly to GitHub.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the bank.csv file with semicolon delimiter.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded DataFrame
        
    Raises:
        FileNotFoundError: If the file does not exist
        Exception: For any other loading errors
    """
    try:
        df = pd.read_csv(file_path, sep=';')
        print(f"[INFO] Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File '{file_path}' not found. Please check the file path.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An error occurred while loading the file: {e}")
        sys.exit(1)


def check_data_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Task 1: Overview and Data Type Conversion
    
    - Print total rows and columns
    - Print data types of each variable
    - Transform binary columns (default, housing, loan, y): "yes" -> 1, "no" -> 0
    - Ordinal encode education column
    - Return list of categorical columns suggested for One-Hot Encoding
    
    Args:
        df: Input DataFrame
        
    Returns:
        tuple: (Transformed DataFrame, List of columns for One-Hot Encoding)
    """
    print("\n" + "="*60)
    print("TASK 1: OVERVIEW AND DATA TYPE CONVERSION")
    print("="*60)
    
    # Create a copy to avoid modifying original data
    df_transformed = df.copy()
    
    # Print total rows and columns
    n_rows, n_cols = df_transformed.shape
    print(f"\n[INFO] Total rows: {n_rows}")
    print(f"[INFO] Total columns: {n_cols}")
    
    # Print original data types
    print("\n[INFO] Original Data Types:")
    print("-"*40)
    print(df_transformed.dtypes)
    
    # Define binary columns to transform
    binary_columns = ['default', 'housing', 'loan', 'y']
    
    # Map "yes" -> 1 and "no" -> 0 for binary columns
    print("\n[INFO] Transforming binary columns (yes->1, no->0):")
    for col in binary_columns:
        if col in df_transformed.columns:
            df_transformed[col] = df_transformed[col].map({'yes': 1, 'no': 0})
            print(f"  - Transformed column: {col}")
    
    # Ordinal Encoding for education column
    education_mapping = {
        'unknown': 0,
        'primary': 1,
        'secondary': 2,
        'tertiary': 3
    }
    
    if 'education' in df_transformed.columns:
        print("\n[INFO] Applying Ordinal Encoding to 'education' column:")
        print(f"  Mapping: {education_mapping}")
        df_transformed['education'] = df_transformed['education'].map(education_mapping)
        print("  - Transformed column: education")
    
    # Identify categorical columns for One-Hot Encoding suggestion
    categorical_ohe_candidates = ['job', 'marital', 'contact', 'poutcome', 'month']
    existing_ohe_columns = [col for col in categorical_ohe_candidates if col in df_transformed.columns]
    
    print("\n[INFO] Columns suggested for One-Hot Encoding (for future processing):")
    print(f"  {existing_ohe_columns}")
    
    return df_transformed, existing_ohe_columns


def analyze_target(df: pd.DataFrame, output_path: str = 'target_distribution.png') -> None:
    """
    Task 2: Target Variable Analysis
    
    - Count occurrences of 1 (yes) and 0 (no) in column y
    - Calculate percentage distribution
    - Plot countplot showing data imbalance
    - Save the plot as PNG file
    
    Args:
        df: Input DataFrame (with transformed target column)
        output_path: Path to save the plot
    """
    print("\n" + "="*60)
    print("TASK 2: TARGET VARIABLE ANALYSIS")
    print("="*60)
    
    if 'y' not in df.columns:
        print("[ERROR] Column 'y' not found in DataFrame")
        return
    
    # Count occurrences
    target_counts = df['y'].value_counts()
    print("\n[INFO] Count of target variable 'y':")
    print(f"  - Class 0 (no):  {target_counts.get(0, 0)}")
    print(f"  - Class 1 (yes): {target_counts.get(1, 0)}")
    
    # Calculate percentage distribution
    target_percentage = df['y'].value_counts(normalize=True) * 100
    print("\n[INFO] Percentage distribution of target variable 'y':")
    print(f"  - Class 0 (no):  {target_percentage.get(0, 0):.2f}%")
    print(f"  - Class 1 (yes): {target_percentage.get(1, 0):.2f}%")
    
    # Check for data imbalance
    imbalance_ratio = target_counts.min() / target_counts.max()
    print(f"\n[INFO] Imbalance ratio (minority/majority): {imbalance_ratio:.4f}")
    if imbalance_ratio < 0.5:
        print("  [WARNING] Dataset is imbalanced!")
    
    # Create countplot
    plt.figure(figsize=(10, 6))
    sns.set_style('whitegrid')
    
    ax = sns.countplot(data=df, x='y', hue='y', palette='Set2', edgecolor='black', legend=False)
    
    # Add title and labels
    plt.title('Target Variable Distribution (y)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Class (0 = No, 1 = Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add value labels on bars
    for i, v in enumerate(target_counts.values):
        ax.text(i, v + len(df)*0.01, f'{v}\n({target_percentage.iloc[i]:.1f}%)', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[INFO] Plot saved as '{output_path}'")
    plt.close()


def find_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task 3: Unknown Value Analysis
    
    - Scan entire DataFrame for 'unknown' string values
    - Count and calculate percentage of 'unknown' per column
    - Print summary report showing columns with 'unknown' > 0
    - Special check for pdays column with value -1 (missing value indicator)
    
    Args:
        df: Input DataFrame (original, before transformations)
        
    Returns:
        pd.DataFrame: Summary report of unknown values
    """
    print("\n" + "="*60)
    print("TASK 3: UNKNOWN VALUE ANALYSIS")
    print("="*60)
    
    # Create a copy to work with
    df_analysis = df.copy()
    
    # Initialize results dictionary
    unknown_summary = {
        'Column': [],
        'Unknown_Count': [],
        'Unknown_Percentage': []
    }
    
    # Scan each column for 'unknown' string values
    for col in df_analysis.columns:
        # Only check object/string columns for 'unknown' string
        if df_analysis[col].dtype == 'object':
            unknown_count = (df_analysis[col] == 'unknown').sum()
            if unknown_count > 0:
                unknown_percentage = (unknown_count / len(df_analysis)) * 100
                unknown_summary['Column'].append(col)
                unknown_summary['Unknown_Count'].append(unknown_count)
                unknown_summary['Unknown_Percentage'].append(round(unknown_percentage, 2))
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(unknown_summary)
    
    # Print report
    if len(summary_df) > 0:
        print("\n[INFO] Columns containing 'unknown' values:")
        print("-"*60)
        print(summary_df.to_string(index=False))
        print("-"*60)
        print(f"\n[INFO] Total columns with 'unknown' values: {len(summary_df)}")
    else:
        print("\n[INFO] No 'unknown' string values found in any column.")
    
    # Special check for pdays column with -1 values
    if 'pdays' in df_analysis.columns:
        pdays_minus_one_count = (df_analysis['pdays'] == -1).sum()
        pdays_minus_one_pct = (pdays_minus_one_count / len(df_analysis)) * 100
        
        print("\n[INFO] Special Report for 'pdays' column:")
        print("-"*60)
        print(f"  - Count of -1 values (missing indicator): {pdays_minus_one_count}")
        print(f"  - Percentage of -1 values: {pdays_minus_one_pct:.2f}%")
        print("  [NOTE] Value -1 in 'pdays' indicates the client was not contacted before")
        print("-"*60)
    
    return summary_df


def main():
    """
    Main function to execute the preprocessing pipeline sequentially.
    """
    print("\n" + "#"*60)
    print("# BANK MARKETING DATA PREPROCESSING PIPELINE")
    print("#"*60)
    
    # Define file path
    file_path = Path(__file__).parent / 'bank.csv'
    
    # Step 1: Load data
    df_original = load_data(str(file_path))
    
    # Keep original for unknown analysis (before transformations)
    df_for_unknown_analysis = df_original.copy()
    
    # Step 2: Task 1 - Check data types and transform
    df_transformed, ohe_columns = check_data_types(df_original)
    
    print("\n[INFO] Transformed DataFrame Data Types:")
    print("-"*40)
    print(df_transformed.dtypes)
    
    # Step 3: Task 2 - Analyze target variable
    analyze_target(df_transformed)
    
    # Step 4: Task 3 - Find unknown values (using original data)
    unknown_report = find_unknowns(df_for_unknown_analysis)
    
    # Final summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print("\n[SUMMARY]")
    print(f"  - Original shape: {df_original.shape}")
    print(f"  - Transformed shape: {df_transformed.shape}")
    print(f"  - Columns for One-Hot Encoding: {ohe_columns}")
    print(f"  - Target plot saved: target_distribution.png")
    print(f"  - Columns with 'unknown' values: {len(unknown_report)}")
    print("\n[INFO] Ready for next preprocessing steps!")
    print("="*60 + "\n")
    
    return df_transformed


if __name__ == "__main__":
    main()
