# Import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Helper Functions
def extract_state_from_name(name):
    """
    Extract state from institution name using multiple methods
    """
    # Common abbreviations that might be confused with state codes
    not_states = ['ST', 'US', 'AM', 'PM', 'BS', 'MS', 'PhD']
    
    # Try to find state at the end of the name
    parts = str(name).split(',')
    if len(parts) > 1:
        potential_state = parts[-1].strip()
        if len(potential_state) == 2 and potential_state not in not_states:
            return potential_state
    
    # Try to find state in parentheses
    if '(' in str(name) and ')' in str(name):
        start = name.rindex('(')
        end = name.rindex(')')
        potential_state = name[start+1:end].strip()
        if len(potential_state) == 2 and potential_state not in not_states:
            return potential_state
    
    return None

def load_data():
    """
    Load the raw data file
    """
    try:
        raw_data = pd.read_csv('raw_scorecard_data.csv')
        print(f"Loaded {len(raw_data)} records")
        print("\nSample of raw data:")
        print(raw_data[['INSTNM', 'CONTROL']].head())
        return raw_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Data Cleaning Function
def clean_and_prepare_data(df):
    """
    Clean and prepare data focusing on columns with high coverage
    """
    # Select columns with high coverage
    selected_columns = [
        'UNITID', 'INSTNM', 'CREDLEV', 'CIPCODE', 'CIPDESC',
        'CONTROL', 'DISTANCE', 'EARN_MDN_5YR', 'EARN_COUNT_NWNE_5YR',
        'EARN_COUNT_WNE_5YR', 'EARN_IN_STATE_5YR'
    ]
    
    # Create cleaned dataframe
    cleaned_df = df[selected_columns].copy()
    
    # Debug print statements
    print("\nDebug Information:")
    print("CONTROL unique values:", cleaned_df['CONTROL'].unique())
    print("Sample of INSTNM values:", cleaned_df['INSTNM'].head())
    
    # Replace privacy suppressed values and NULL
    cleaned_df = cleaned_df.replace({
        'PrivacySuppressed': np.nan,
        'NULL': np.nan,
        'null': np.nan,
        '': np.nan
    })
    
    # Map institution types
    institution_type_map = {
        'Public': 'Public',
        'Private, nonprofit': 'Private Nonprofit',
        'Private, for-profit': 'Private For-Profit',
        'Foreign': 'Foreign'
    }
    cleaned_df['institution_type'] = cleaned_df['CONTROL'].map(institution_type_map)

    # State mapping dictionary
    state_mapping = {
        'Alabama A & M University': 'AL',
        'Alabama A&M University': 'AL',
        'University of Alabama': 'AL',
        'Arizona State University': 'AZ',
        'University of Arkansas': 'AR',
        'University of California': 'CA',
        'Colorado State University': 'CO',
        'University of Connecticut': 'CT',
        'University of Delaware': 'DE',
        'University of Florida': 'FL',
        'University of Georgia': 'GA',
        'University of Hawaii': 'HI',
        'University of Idaho': 'ID',
        'University of Illinois': 'IL',
        'Indiana University': 'IN',
        'University of Iowa': 'IA',
        'Kansas State University': 'KS',
        'University of Kentucky': 'KY',
        'Louisiana State University': 'LA',
        'University of Maine': 'ME',
        'University of Maryland': 'MD',
        'University of Massachusetts': 'MA',
        'Michigan State University': 'MI',
        'University of Minnesota': 'MN',
        'Mississippi State University': 'MS',
        'University of Missouri': 'MO',
        'Montana State University': 'MT',
        'University of Nebraska': 'NE',
        'University of Nevada': 'NV',
        'University of New Hampshire': 'NH',
        'Rutgers University': 'NJ',
        'New Mexico State University': 'NM',
        'New York University': 'NY',
        'University of North Carolina': 'NC',
        'North Dakota State University': 'ND',
        'Ohio State University': 'OH',
        'University of Oklahoma': 'OK',
        'Oregon State University': 'OR',
        'Pennsylvania State University': 'PA',
        'University of Rhode Island': 'RI',
        'University of South Carolina': 'SC',
        'South Dakota State University': 'SD',
        'University of Tennessee': 'TN',
        'Texas A&M University': 'TX',
        'University of Utah': 'UT',
        'University of Vermont': 'VT',
        'University of Virginia': 'VA',
        'University of Washington': 'WA',
        'West Virginia University': 'WV',
        'University of Wisconsin': 'WI',
        'University of Wyoming': 'WY'
    }
    
    # Extract state with multiple patterns
    state_patterns = [
        r'(?:.*,\s*)([A-Z]{2})(?:\s|$)',
        r'(?:.*\s)([A-Z]{2})(?:\s|$)',
        r'(?:.*-)([A-Z]{2})(?:\s|$)',
        r'.*\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC|PR)\b'
    ]
    
    cleaned_df['state'] = None
    for pattern in state_patterns:
        mask = cleaned_df['state'].isna()
        cleaned_df.loc[mask, 'state'] = cleaned_df.loc[mask, 'INSTNM'].str.extract(pattern)
    
    # Apply manual mapping where state is still missing
    mask = cleaned_df['state'].isna()
    cleaned_df.loc[mask, 'state'] = cleaned_df.loc[mask, 'INSTNM'].map(state_mapping)
    
    # Define valid US states
    us_states = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
        'DC', 'PR'
    ]
    
    # Clean up state codes
    cleaned_df.loc[~cleaned_df['state'].isin(us_states), 'state'] = None
    
    # Print updated distributions
    print("\nUpdated Institution Type Distribution:")
    print(cleaned_df['institution_type'].value_counts(dropna=False))
    
    print("\nUpdated State Distribution (top 20):")
    print(cleaned_df['state'].value_counts().head(20))
    
    # Convert numeric columns
    numeric_columns = [
        'EARN_MDN_5YR', 'EARN_COUNT_NWNE_5YR',
        'EARN_COUNT_WNE_5YR', 'EARN_IN_STATE_5YR'
    ]
    
    for col in numeric_columns:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    # Add program level descriptions
    cleaned_df['program_level'] = cleaned_df['CREDLEV'].map({
        6: "Master's Degree",
        7: "Doctoral Degree",
        8: "First Professional Degree"
    })
    
    print("\nData cleaning completed")
    print(f"Shape after cleaning: {cleaned_df.shape}")
    return cleaned_df


# Feature Creation Function
def create_analysis_features(df):
    """
    Create features using high-coverage metrics
    """
    features = df.copy()
    
    # Calculate employment metrics using 5-year data
    features['total_students_5yr'] = features['EARN_COUNT_WNE_5YR'] + features['EARN_COUNT_NWNE_5YR']
    features['employment_rate_5yr'] = features['EARN_COUNT_WNE_5YR'] / features['total_students_5yr']
    
    # Calculate program size categories
    features['program_size'] = pd.qcut(
        features['total_students_5yr'],
        q=4,
        labels=['Small', 'Medium', 'Large', 'Very Large']
    )
    
    # Calculate earnings metrics
    features['earnings_5yr'] = features['EARN_MDN_5YR']
    features['earnings_percentile'] = features['earnings_5yr'].rank(pct=True)
    
    # Calculate in-state employment rate
    features['in_state_rate'] = features['EARN_IN_STATE_5YR'] / features['total_students_5yr']
    
    print("Feature creation completed")
    print(f"Shape after feature creation: {features.shape}")
    return features

# Map Visualization Function
def create_us_state_map(data, metric='earnings_5yr', title=None):
    """
    Create a choropleth map of the US showing state-level metrics
    """
    # Load US states shapefile
    us_states = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip')
    
    # Convert state data to state-level aggregates
    state_metrics = data.groupby('state')[metric].median().reset_index()
    
    # Convert state codes to state names for mapping
    state_code_map = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 
        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
        'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 
        'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
        'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
        'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
        'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
        'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
        'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
    }
    state_metrics['state_name'] = state_metrics['state'].map(state_code_map)
    
    # Merge with geographic data
    merged_data = us_states.merge(state_metrics, how='left', left_on='NAME', right_on='state_name')
    
    # Create figure with extra space at bottom for legend
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_axes([0, 0.1, 1, 0.9])  # Leave space at bottom for legend
    
    # Create custom colormap
    colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
    cmap = LinearSegmentedColormap.from_list('custom_blues', colors)
    
    # Format the legend label based on the metric
    if metric == 'earnings_5yr':
        legend_label = 'Median Earnings ($)'
        formatter = lambda x, p: f'${x:,.0f}'
    elif metric == 'employment_rate_5yr':
        legend_label = 'Employment Rate'
        formatter = lambda x, p: f'{x:.1%}'
    else:
        legend_label = metric.replace('_', ' ').title()
        formatter = lambda x, p: f'{x:,.0f}'
    
    # Plot the map
    merged_data.plot(
        column=metric,
        ax=ax,
        legend=True,
        legend_kwds={
            'label': legend_label,
            'orientation': 'horizontal',
            'fraction': 0.046,
            'pad': 0.04,
            'format': formatter
        },
        missing_kwds={'color': 'lightgrey'},
        cmap=cmap
    )
    
    # Customize the map
    ax.axis('off')
    if title is None:
        title = f'State-Level {metric.replace("_", " ").title()}'
    plt.title(title, fontsize=20, pad=20)
    
    plt.show()

# Basic Visualization Functions
def create_program_level_plot(data):
    """Create program level distribution plot"""
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x='program_level', order=data['program_level'].value_counts().index)
    plt.title('Program Level Distribution', fontsize=14)
    plt.xticks(rotation=45)
    plt.xlabel('Program Level', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.show()

def create_institution_type_plot(data):
    """Create institution type distribution plot"""
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x='institution_type', order=data['institution_type'].value_counts().index)
    plt.title('Institution Type Distribution', fontsize=14)
    plt.xticks(rotation=45)
    plt.xlabel('Institution Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.show()

def create_earnings_distribution_plot(data):
    """Create earnings distribution plot"""
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x='earnings_5yr', bins=50)
    plt.title('5-Year Earnings Distribution', fontsize=14)
    plt.xlabel('Median Earnings ($)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.show()

def create_earnings_by_program_plot(data):
    """Create earnings by program level plot"""
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='program_level', y='earnings_5yr')
    plt.title('Earnings Distribution by Program Level', fontsize=14)
    plt.xticks(rotation=45)
    plt.xlabel('Program Level', fontsize=12)
    plt.ylabel('Median Earnings ($)', fontsize=12)
    plt.tight_layout()
    plt.show()

def create_earnings_by_institution_plot(data):
    """Create earnings by institution type plot"""
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='institution_type', y='earnings_5yr')
    plt.title('Earnings Distribution by Institution Type', fontsize=14)
    plt.xticks(rotation=45)
    plt.xlabel('Institution Type', fontsize=12)
    plt.ylabel('Median Earnings ($)', fontsize=12)
    plt.tight_layout()
    plt.show()

def create_employment_rate_plot(data):
    """Create employment rate distribution plot"""
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x='employment_rate_5yr', bins=50)
    plt.title('5-Year Employment Rate Distribution', fontsize=14)
    plt.xlabel('Employment Rate', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.show()

# Main Processing Functions
def process_graduate_programs(raw_data):
    """
    Process graduate programs and print all results
    """
    # Filter for graduate programs
    grad_data = raw_data[raw_data['CREDLEV'].isin([6, 7, 8])]
    print(f"Found {len(grad_data)} graduate programs")
    print("-" * 80)
    
    # Clean and prepare data
    cleaned_data = clean_and_prepare_data(grad_data)
    print("-" * 80)
    
    # Create features
    final_data = create_analysis_features(cleaned_data)
    print("-" * 80)
    
    # Print detailed analysis
    print("\nDETAILED ANALYSIS RESULTS")
    print("-" * 80)
    
    # Program Level Analysis
    print("\nPROGRAM LEVEL STATISTICS:")
    print("\nProgram Count by Level:")
    print(final_data['program_level'].value_counts())
    
    print("\nMedian Earnings by Program Level:")
    print(final_data.groupby('program_level')['earnings_5yr'].median().round(2))
    
    print("\nMean Employment Rate by Program Level:")
    print(final_data.groupby('program_level')['employment_rate_5yr'].mean().round(3))
    
    # Institution Type Analysis
    print("\nINSTITUTION TYPE STATISTICS:")
    print("\nProgram Count by Institution Type:")
    print(final_data['institution_type'].value_counts())
    
    print("\nMedian Earnings by Institution Type:")
    print(final_data.groupby('institution_type')['earnings_5yr'].median().round(2))
    
    print("\nMean Employment Rate by Institution Type:")
    print(final_data.groupby('institution_type')['employment_rate_5yr'].mean().round(3))
    
    # State Analysis
    print("\nSTATE STATISTICS:")
    print("\nTop 20 States by Program Count:")
    print(final_data['state'].value_counts().head(20))
    
    print("\nTop 10 States by Median Earnings:")
    state_earnings = final_data.groupby('state')['earnings_5yr'].median().round(2)
    print(state_earnings.sort_values(ascending=False).head(10))
    
    # Overall Statistics
    print("\nOVERALL STATISTICS:")
    print("\nEarnings Statistics ($):")
    earnings_stats = final_data['earnings_5yr'].describe().round(2)
    print(earnings_stats)
    
    print("\nEmployment Rate Statistics:")
    employment_stats = final_data['employment_rate_5yr'].describe().round(3)
    print(employment_stats)
    
    return final_data

def ensure_data_loaded():
    """
    Ensures raw_data is loaded, loading it if necessary
    """
    global raw_data
    if 'raw_data' not in globals() or raw_data is None:
        raw_data = load_data()
    return raw_data

# Initial Data Loading and Processing
raw_data = ensure_data_loaded()
if raw_data is not None:
    final_data = process_graduate_programs(raw_data)
    print("\nProcessing completed successfully!")
else:
    print("Error: Could not load data")

# Basic Distribution Plots
# Program Level Distribution
create_program_level_plot(final_data)

# Institution Type Distribution
create_institution_type_plot(final_data)

# Earnings Distribution
create_earnings_distribution_plot(final_data)

# Earnings by Program Level
create_earnings_by_program_plot(final_data)

# Earnings by Institution Type
create_earnings_by_institution_plot(final_data)

# Employment Rate Distribution
create_employment_rate_plot(final_data)

# State Maps
# Earnings Map
create_us_state_map(
    final_data, 
    metric='earnings_5yr',
    title='Median 5-Year Earnings by State'
)

# Employment Rate Map
create_us_state_map(
    final_data, 
    metric='employment_rate_5yr',
    title='Mean Employment Rate by State'
)

# Program Count Map
state_counts = final_data.groupby('state').size().reset_index(name='program_count')
create_us_state_map(
    state_counts, 
    metric='program_count',
    title='Graduate Program Count by State'
)
