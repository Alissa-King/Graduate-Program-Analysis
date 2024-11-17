# Graduate-Program-Analysis

## Description
This project analyzes success factors in online graduate education using the Department of Education's College Scorecard dataset. It examines over 28,000 graduate programs across the United States, providing insights into employment outcomes, earnings potential, and geographic patterns to help higher education institutions optimize their program offerings.

## Features
- Comprehensive analysis of graduate program outcomes
- Geographic visualization of program distribution
- State-level success metrics mapping
- Comparative analysis across program types
- Employment and earnings pattern analysis
- Institution type performance comparison

## Technologies Used
- Python 3.x
- Pandas for data manipulation
- NumPy for numerical operations
- Matplotlib and Seaborn for data visualization
- GeoPandas for geographic mapping
- SQLite for data storage

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/Alissa-King/graduate-program-analysis.git
   cd graduate-program-analysis
   ```
2. Install required packages:
   ```
   pip install pandas numpy matplotlib seaborn geopandas
   ```
3. Download the College Scorecard dataset:
   - Visit https://collegescorecard.ed.gov/data/
   - Download the latest field of study data file

## Usage
1. Run the main analysis script:
   ```
   python graduate_analysis.py
   ```
2. View generated visualizations in the 'plots' directory
3. Access analysis results in 'results.csv'

## Analysis Components
- **Program Distribution Analysis**: Breakdown of graduate program types
- **Geographic Analysis**: State-level program distribution and performance
- **Earnings Analysis**: Comparison across program levels and institution types
- **Employment Analysis**: Employment rates and geographic patterns

## Data Processing
The project includes comprehensive data cleaning and preparation:
- Institution name standardization
- Geographic location extraction
- Numeric data validation
- Missing value handling

## Contributing
Contributions are welcome. Please feel free to submit a Pull Request.

## License
This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments
- Data provided by U.S. Department of Education
- State boundary data from U.S. Census Bureau
