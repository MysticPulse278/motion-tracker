import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from itertools import cycle


def separate_and_plot_teeth(data, exclude_indices=None):
    """
    Separates a sawtooth-like wave where teeth are negative values 
    surrounded by zeros, and plots each tooth with a different color.
    Each tooth is aligned to start at the same time point (index 0).
    
    Parameters:
    data (array-like): Input data array
    exclude_indices (list or None): Indices of teeth to exclude from plotting
    
    Returns:
    fig, ax: The matplotlib figure and axis objects
    teeth_dict: Dictionary mapping the original tooth index to the tooth data
    """
    # Convert to numpy array if not already
    data = np.array(data)
    
    # Initialize exclude_indices if None
    if exclude_indices is None:
        exclude_indices = []
    
    # Find where the data transitions from zero to non-zero and vice versa
    transitions = np.where(np.diff(data != 0))[0]
    
    # Initialize dictionary to store each tooth with its original index
    teeth_dict = {}
    
    # Find each tooth (regions of negative values)
    tooth_index = 0
    for i in range(len(transitions) - 1):
        start_idx = transitions[i] + 1
        end_idx = transitions[i + 1]
        
        # Check if this is a negative segment (a tooth)
        if start_idx < len(data) and data[start_idx] < 0:
            # Extract the tooth data
            tooth_data = data[start_idx:end_idx + 1]
            teeth_dict[tooth_index] = tooth_data
            tooth_index += 1
    
    # Handle possible last tooth
    if len(transitions) > 0 and transitions[-1] + 1 < len(data) and data[transitions[-1] + 1] < 0:
        tooth_data = data[transitions[-1] + 1:]
        teeth_dict[tooth_index] = tooth_data
    
    # Create filtered dictionary excluding specified teeth
    filtered_teeth = {idx: data for idx, data in teeth_dict.items() if idx not in exclude_indices}
    
    # Get a colormap for different colors
    cmap = get_cmap('tab10')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each tooth with a different color, starting at the same time (index 0)
    for idx, tooth in filtered_teeth.items():
        # Create a new x-axis starting at 0 for each tooth
        x_range = np.arange(len(tooth))
        
        # Get a color based on the original index to maintain color consistency
        color = cmap(idx % 10)
        
        # Plot this tooth
        ax.plot(x_range, tooth, color=color, linewidth=2, label=f'Tooth {idx}')
    
    # Set labels and title
    ax.set_xlabel('Relative Frame Index (Starting at 0)')
    ax.set_ylabel('Pixel position')
    excluded_str = f", excluding 'teeth' {exclude_indices}" if exclude_indices else ""
    ax.set_title(f'Separated "teeth" {excluded_str}')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend with smaller font size if there are many teeth
    if len(filtered_teeth) > 10:
        ax.legend(fontsize='small', loc='best')
    else:
        ax.legend(loc='best')
    
    plt.tight_layout()
    return fig, ax, teeth_dict

# Function to display all teeth and then allow filtering
def analyze_and_filter_teeth(data):
    """
    Two-step function to first show all teeth, then allow filtering out unwanted ones.
    
    Parameters:
    data (array-like): Input data array
    
    Returns:
    None (displays plots)
    """
    # First plot with all teeth
    print("Plotting all teeth. Note the indices to exclude if needed.")
    fig, ax, teeth_dict = separate_and_plot_teeth(data)
    plt.show()
    
    # Ask for teeth to exclude
    exclude_str = input("Enter indices of teeth to exclude (comma-separated, e.g. '0,3,5'): ")
    if exclude_str.strip():
        exclude_indices = [int(idx.strip()) for idx in exclude_str.split(',')]
        
        # Plot again with exclusions
        print(f"Plotting teeth excluding indices: {exclude_indices}")
        fig, ax, _ = separate_and_plot_teeth(data, exclude_indices=exclude_indices)
        plt.show()
    else:
        print("No teeth excluded.")

# Example usage with sample data
if __name__ == "__main__":
        # Initialize lists to store frame numbers and y positions
    frames = []
    y_positions = []

    # Read the CSV file
    with open('trajectories 20250224_133014.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            # Append frame number and y position to respective lists
            frames.append(int(row[0]))
            y_positions.append(-float(row[2]))

    
    # Separate and plot the teeth
    analyze_and_filter_teeth(y_positions)