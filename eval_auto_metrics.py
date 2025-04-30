import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Apply a style sheet for a cleaner look
plt.style.use('seaborn-v0_8-whitegrid')

df = pd.read_csv('fixed_metrics.csv')


metric_cols = ['NIQE', 'BRISQUE', 'PIQE', 'MetaIQA', 'RankIQA', 'HyperIQA', 'Contrique', 'CNNIQA', 'TReS', 'CLIPIQA']


# Calculate average metrics for each region
region_averages = df.groupby('Region').agg({
    metric: 'mean' for metric in metric_cols
}).reset_index()


updated_region_averages = df.groupby('Region').agg({
    metric: ['mean', 'std'] for metric in metric_cols
}).reset_index()

# Flatten the multi-level columns
updated_region_averages.columns = ['Region'] + [f"{metric}_{stat}" for metric in metric_cols for stat in ['mean', 'std']]



# Display the result
# print(region_averages)

regions = region_averages['Region'].tolist()

def plot_percent_changes():
    # Use North America as a baseline for comparison
    baseline_region = 'North America'

    # Check if the baseline region exists in the data
    if baseline_region not in region_averages['Region'].values:
        print(f"Error: Baseline region '{baseline_region}' not found in the data.")
    else:
        # Get the baseline values for North America
        baseline_values = region_averages[region_averages['Region'] == baseline_region].iloc[0]
        
        # Create a new DataFrame to store the percent changes
        percent_changes = pd.DataFrame({'Region': region_averages['Region']})
        
        # Calculate percent change for each metric relative to North America
        # Flip the sign to show loss (negative values indicate worse performance)
        for metric in metric_cols:
            baseline_value = baseline_values[metric]
            if baseline_value != 0:  # Avoid division by zero
                # Calculate percent change relative to North America (positive values indicate improvement)
                percent_changes[f"{metric}_pct_change"] = ((region_averages[metric] - baseline_value) / baseline_value) * 100
            else:
                percent_changes[f"{metric}_pct_change"] = float('nan')
        
        # Display the percent changes
        # print("\nPercent change relative to North America:")
        # print(percent_changes)
        
        # Create a plot to visualize the percent changes
        # import matplotlib.pyplot as plt # No longer needed here
        # import numpy as np # No longer needed here

        # Filter out the baseline region from the visualization
        plot_data = percent_changes[percent_changes['Region'] != baseline_region]
        
        # Get regions and metrics for plotting
        regions = plot_data['Region'].tolist()
        pct_change_cols = [col for col in plot_data.columns if col.endswith('_pct_change')]
        
        # Define colors for each region
        region_colors = {
            'Africa': '#FF9900',
            'Asia': '#3366CC',
            'Europe': '#DC3912',
            'South America': '#109618',
            'Australia': '#990099',
            'Middle East': '#0099C6',
            'Central America': '#DD4477',
            'Caribbean': '#66AA00',
            'Pacific Islands': '#B82E2E',
            'Antarctica': '#316395'
        }
        
        # Save each metric as a separate plot
        for metric_col in pct_change_cols:
            metric_name = metric_col.replace('_pct_change', '')
            
            # Create a new figure for each metric
            plt.figure(figsize=(10, 6))
            
            # Create bar chart for this metric with region-specific colors and edge color
            bars = plt.bar(regions, plot_data[metric_col], width=0.7, edgecolor='black', linewidth=0.8)
            
            # Apply colors to each bar based on region
            for j, bar in enumerate(bars):
                region = regions[j]
                if region in region_colors:
                    bar.set_color(region_colors[region])
            
            # Add labels and title
            plt.xlabel('Region', fontsize=12)
            plt.ylabel('Change (%)', fontsize=12)
            plt.title(f'{metric_name}', fontsize=20, fontweight='bold')
            plt.xticks(range(len(regions)), regions, rotation=45, ha='right', fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            
            # Add a horizontal line at 0%
            plt.axhline(y=0, color='grey', linestyle='-', linewidth=1, alpha=0.7)
            
            # Add a legend for region colors
            handles = [plt.Rectangle((0, 0), 1, 1, color=color, ec='black') 
                      for region, color in region_colors.items() if region in regions]
            region_labels = [region for region in region_colors.keys() if region in regions]
            plt.legend(handles, region_labels, loc='best', title="Regions", fontsize=9, title_fontsize=10)
            
            # Add border around the plot
            plt.gca().spines['top'].set_visible(True)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['top'].set_linewidth(1.0)
            plt.gca().spines['right'].set_linewidth(1.0)
            plt.gca().spines['bottom'].set_linewidth(1.0)
            plt.gca().spines['left'].set_linewidth(1.0)
            
            plt.tight_layout()
            plt.savefig(f'graphs/{metric_name}_percent_change.png', dpi=300, bbox_inches='tight')
            print(f"Saved {metric_name} graph to graphs/{metric_name}_percent_change.png")
            plt.close()
        
        # Also save the combined plot for reference
        n_metrics = len(pct_change_cols)
        n_cols = 2  # You can adjust this for layout
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        # Create figure and subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()  # Flatten to make indexing easier
        
        # Plot each metric in its own subplot
        for i, metric_col in enumerate(pct_change_cols):
            metric_name = metric_col.replace('_pct_change', '')
            ax = axes[i]
            
            # Create bar chart for this metric with region-specific colors and edge color
            bars = ax.bar(regions, plot_data[metric_col], width=0.7, edgecolor='black', linewidth=0.8)
            
            # Apply colors to each bar based on region
            for j, bar in enumerate(bars):
                region = regions[j]
                if region in region_colors:
                    bar.set_color(region_colors[region])
            
            # Add labels and title
            ax.set_xlabel('Region', fontsize=10)
            ax.set_ylabel('Change (%)', fontsize=10)
            ax.set_title(f'{metric_name}', fontsize=20, fontweight='bold')
            ax.set_xticks(range(len(regions)))
            ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=9)
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            
            # Add a horizontal line at 0%
            ax.axhline(y=0, color='grey', linestyle='-', linewidth=1, alpha=0.7)
            
            # Add border around each subplot
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['top'].set_linewidth(1.0)
            ax.spines['right'].set_linewidth(1.0)
            ax.spines['bottom'].set_linewidth(1.0)
            ax.spines['left'].set_linewidth(1.0)
        
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        # Add a legend for region colors below the plots
        handles = [plt.Rectangle((0, 0), 1, 1, color=color, ec='black') for region, color in region_colors.items() if region in regions]
        region_labels = [region for region in region_colors.keys() if region in regions]
        fig.legend(handles, region_labels, loc='lower center', ncol=min(len(region_labels), 5), bbox_to_anchor=(0.5, -0.02), title="Regions", fontsize=9, title_fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig('graphs/percent_change_subplots_styled.png', dpi=300, bbox_inches='tight')
        print("Saved combined plot to graphs/percent_change_subplots_styled.png")
        plt.close()


def calculate_object_averages(csv_file='fixed_metrics.csv'):
    """
    Calculate and return the average metrics for each object across all regions.
    
    Args:
        csv_file (str): Path to the CSV file containing metrics data
        
    Returns:
        pandas.DataFrame: DataFrame containing average metrics for each object
    """
    import pandas as pd
    
    # Load the metrics data
    try:
        metrics_df = pd.read_csv(csv_file)
        # print(f"Loaded data with {len(metrics_df)} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Define metric columns
    metric_cols = ['NIQE', 'BRISQUE', 'PIQE', 'MetaIQA', 'RankIQA', 'HyperIQA', 
                  'Contrique', 'CNNIQA', 'TReS', 'CLIPIQA']
    
    # Calculate average metrics for each object
    object_averages = metrics_df.groupby('Object').agg({
        metric: 'mean' for metric in metric_cols
    }).reset_index()
    
    # print(f"Calculated averages for {len(object_averages)} objects")
    # print(object_averages)
    return object_averages

def save_object_averages(output_file='object_averages.csv'):
    """
    Calculate object averages and save to a CSV file.
    
    Args:
        output_file (str): Path to save the output CSV file
    """
    object_averages = calculate_object_averages()
    
    if object_averages is not None:
        object_averages.to_csv(output_file, index=False)
        print(f"Object averages saved to {output_file}")

# Uncomment to run:
# save_object_averages()


avg_obj_metrics = calculate_object_averages()

def plot_object_metrics_scatter(metrics_df=None, csv_file='fixed_metrics.csv', output_file='object_metrics_boxplot.png'):
    """
    Create boxplots to show the distribution of metric values per object.
    
    Args:
        metrics_df (pandas.DataFrame, optional): DataFrame containing metrics data
        csv_file (str): Path to the CSV file containing metrics data (used if metrics_df is None)
        output_file (str): Path to save the output plot
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load the metrics data if not provided
    if metrics_df is None:
        try:
            metrics_df = pd.read_csv(csv_file)
            # print(f"Loaded data with {len(metrics_df)} rows")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    # Define metric columns
    metric_cols = ['NIQE', 'BRISQUE', 'PIQE', 'MetaIQA', 'RankIQA', 'HyperIQA', 
                  'Contrique', 'CNNIQA', 'TReS', 'CLIPIQA']
    
    # Set up the figure with increased size
    plt.figure(figsize=(20, 16))
    
    # Create a boxplot for each metric
    for i, metric in enumerate(metric_cols):
        plt.subplot(2, 5, i+1)  # 2 rows, 5 columns
        
        # Create a dataframe with object and metric values
        plot_data = metrics_df[['Object', metric]].dropna()
        
        # Create the boxplot
        sns.boxplot(x='Object', y=metric, data=plot_data)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90)
        
        # Set title and adjust layout
        plt.title(f'{metric} by Object', fontsize=14)
        
    # Apply tight layout after all subplots are created
    plt.tight_layout(pad=3.0)
    
    # Save the figure
    plt.savefig('graphs/'+output_file, dpi=300, bbox_inches='tight')
    print(f"Boxplot saved to {output_file}")

# Example usage:
plot_percent_changes()
# plot_object_metrics_scatter()

