import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Apply a style sheet for a cleaner look
plt.style.use('seaborn-v0_8-whitegrid')

df = pd.read_csv('fixed_metrics.csv')


metric_cols = ['NIQE', 'BRISQUE', 'PIQE', 'MetaIQA', 'RankIQA', 'HyperIQA', 'Contrique', 'CNNIQA', 'TReS', 'CLIPIQA']



region_averages = df.groupby('Region').agg({
    metric: ['mean', 'std'] for metric in metric_cols
}).reset_index()


# Flatten the multi-level columns
region_averages.columns = ['Region'] + [f"{metric}_{stat}" for metric in metric_cols for stat in ['mean', 'std']]

# Create a formatted table with mean ± std for each metric
formatted_table = pd.DataFrame()
formatted_table['Region'] = region_averages['Region']

# Format each metric as "mean ± std"
for metric in metric_cols:
    formatted_table[metric] = region_averages.apply(
        lambda row: f"{row[f'{metric}_mean']:.2f} ± {row[f'{metric}_std']:.2f}",
        axis=1
    )

# Display the formatted table
# print("\nRegion Metrics (mean ± std):")
print(formatted_table.to_string(index=False))

# # Save the formatted table to CSV
formatted_table.to_csv('tables/region_metrics_formatted.csv', index=False)
# print("Saved formatted metrics to region_metrics_formatted.csv")


# Perform statistical significance testing to compare each region with North America
from scipy import stats

# Set North America as the baseline region
baseline_region = 'North America'

# Check if the baseline region exists in the data
if baseline_region not in region_averages['Region'].values:
    print(f"Error: Baseline region '{baseline_region}' not found in the data.")
else:
    # Get the baseline region data
    baseline_data = df[df['Region'] == baseline_region]
    
    # Create a DataFrame to store significance test results
    significance_results = pd.DataFrame()
    significance_results['Region'] = region_averages['Region']
    
    # For each metric, perform t-test comparing each region with North America
    for metric in metric_cols:
        # Get baseline metric values
        baseline_values = baseline_data[metric].values
        
        # For each region, perform t-test
        p_values = []
        for region in region_averages['Region']:
            if region == baseline_region:
                p_values.append(1.0)  # Same region, p-value = 1
            else:
                region_values = df[df['Region'] == region][metric].values
                # Perform independent t-test
                _, p_value = stats.ttest_ind(region_values, baseline_values, equal_var=False)
                p_values.append(p_value)
        
        # Add p-values to results DataFrame
        significance_results[f"{metric}_p_value"] = p_values
        
        # Add significance indicators (* for p < 0.05, ** for p < 0.01, *** for p < 0.001)
        significance_results[f"{metric}_sig"] = significance_results[f"{metric}_p_value"].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        )
    
    # Create a formatted table with significance indicators
    sig_formatted_table = pd.DataFrame()
    sig_formatted_table['Region'] = formatted_table['Region']
    
    # Format each metric with significance indicators
    for metric in metric_cols:
        sig_formatted_table[metric] = [
            f"{formatted_table.loc[i, metric]} {significance_results.loc[i, f'{metric}_sig']}" 
            if significance_results.loc[i, 'Region'] != baseline_region 
            else formatted_table.loc[i, metric]
            for i in range(len(formatted_table))
        ]
    
    # Display the significance-annotated table
    # print("\nRegion Metrics with Statistical Significance vs. North America:")
    # print("(* p<0.05, ** p<0.01, *** p<0.001, ns: not significant)")
    # print(sig_formatted_table.to_string(index=False))
    
    # Create a boolean table showing if metrics are significantly different
    significance_bool_table = pd.DataFrame()
    significance_bool_table['Region'] = region_averages['Region']
    
    for metric in metric_cols:
        # True if p < 0.05 (significant), False otherwise
        significance_bool_table[metric] = significance_results[f"{metric}_p_value"].apply(
            lambda p: True if p < 0.05 else False
        )
        # Set baseline region to False (not different from itself)
        significance_bool_table.loc[significance_bool_table['Region'] == baseline_region, metric] = False
    
    # Filter out the baseline region (North America) from the display
    display_significance_table = significance_bool_table[significance_bool_table['Region'] != baseline_region]
    
    print("\nSignificant Difference from North America (True/False):")
    print(display_significance_table.to_string(index=False))
    
    # Summarize the significance results
    summary_results = pd.DataFrame()
    summary_results['Region'] = region_averages['Region']
    
    for metric in metric_cols:
        summary_results[metric] = sig_formatted_table[metric]
    
    # # Save the significance-annotated table to CSV
    # sig_formatted_table.to_csv('tables/region_metrics_significance.csv', index=False)
    # print("Saved significance-annotated metrics to region_metrics_significance.csv")
