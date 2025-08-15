"""
Enhanced display utilities for punch code names and formatting
Aligned with enterprise configuration for column name display
"""
from config import PUNCH_CODE_NAMES, PUNCH_CODE_WORKFORCE_LIMITS

def get_display_name(punch_code, use_table_format=False):
    """
    Get friendly display name for punch code
    
    Args:
        punch_code: The punch code to get display name for
        use_table_format: If True, returns table column display name (same as regular name)
    """
    return PUNCH_CODE_NAMES.get(str(punch_code), str(punch_code))

def get_punch_code_tooltip(punch_code):
    """Get comprehensive tooltip text combining code and name"""
    name = get_display_name(punch_code)
    workforce_info = PUNCH_CODE_WORKFORCE_LIMITS.get(str(punch_code), {})
    
    tooltip = f"Code: {punch_code} - {name}"
    
    # Add workforce information to tooltip
    if workforce_info:
        worker_type = workforce_info.get('type', 'unknown')
        min_workers = workforce_info.get('min_workers', 0)
        max_workers = workforce_info.get('max_workers', 0)
        
        if worker_type == 'fixed':
            tooltip += f" | Fixed: {min_workers} workers"
        elif worker_type == 'kpi_based':
            tooltip += f" | KPI-based: {min_workers}-{max_workers} workers"
    
    return tooltip

def format_table_columns(df, punch_code_columns=None):
    """
    Format dataframe columns to use display names for punch codes
    
    Args:
        df: DataFrame to format
        punch_code_columns: List of columns that are punch codes, if None auto-detect
    """
    df_formatted = df.copy()
    
    if punch_code_columns is None:
        # Auto-detect punch code columns
        punch_code_columns = [col for col in df.columns if str(col) in PUNCH_CODE_NAMES]
    
    # Create column mapping
    column_mapping = {}
    for col in punch_code_columns:
        if str(col) in PUNCH_CODE_NAMES:
            column_mapping[col] = PUNCH_CODE_NAMES[str(col)]
    
    # Rename columns
    df_formatted = df_formatted.rename(columns=column_mapping)
    
    return df_formatted

def get_streamlit_column_config(punch_codes, format_type="%.2f"):
    """
    Generate Streamlit column configuration with display names and tooltips
    
    Args:
        punch_codes: List of punch codes
        format_type: Number format (e.g., "%.2f", "%.0f", "%.1f")
    """
    import streamlit as st
    
    column_config = {}
    
    for punch_code in punch_codes:
        punch_code_str = str(punch_code)
        if punch_code_str in PUNCH_CODE_NAMES:
            display_name = PUNCH_CODE_NAMES[punch_code_str]
            tooltip = get_punch_code_tooltip(punch_code_str)
            
            column_config[display_name] = st.column_config.NumberColumn(
                display_name,
                format=format_type,
                help=tooltip
            )
    
    return column_config

def transform_punch_code_columns(dataframe, exclude_columns=None):
    """
    Transform dataframe columns from punch codes to display names
    
    Args:
        dataframe: DataFrame with punch code columns
        exclude_columns: List of columns to exclude from transformation (e.g., ['TOTAL'])
    """
    if exclude_columns is None:
        exclude_columns = ['TOTAL']
    
    df_transformed = dataframe.copy()
    
    # Create column mapping
    new_columns = []
    for col in df_transformed.columns:
        if col in exclude_columns:
            new_columns.append(col)
        elif str(col) in PUNCH_CODE_NAMES:
            new_columns.append(PUNCH_CODE_NAMES[str(col)])
        else:
            new_columns.append(col)
    
    df_transformed.columns = new_columns
    return df_transformed

def get_workforce_display_info(punch_code):
    """Get formatted workforce information for display"""
    workforce_info = PUNCH_CODE_WORKFORCE_LIMITS.get(str(punch_code), {})
    display_name = get_display_name(punch_code)
    
    info = {
        'display_name': display_name,
        'punch_code': str(punch_code),
        'type': workforce_info.get('type', 'unknown'),
        'min_workers': workforce_info.get('min_workers', 0),
        'max_workers': workforce_info.get('max_workers', 0),
        'regular_workers': workforce_info.get('regular_workers')
    }
    
    return info

def validate_punch_code_display_config():
    """Validate that all punch codes have display names"""
    from config import ENHANCED_WORK_TYPES
    
    missing_names = []
    for punch_code in ENHANCED_WORK_TYPES:
        if punch_code not in PUNCH_CODE_NAMES:
            missing_names.append(punch_code)
    
    if missing_names:
        print(f"Warning: Missing display names for punch codes: {missing_names}")
        return False
    
    return True