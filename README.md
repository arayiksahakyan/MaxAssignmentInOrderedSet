# MaxAssignmentInOrderedSet

# Matrix To Path

## Description
This project provides a Streamlit-based web application that converts a matrix into a graph and finds the shortest path from the top-left corner to the bottom. The shortest path is visualized by highlighting the corresponding elements in the matrix.

## Features
- **Dynamic Matrix Input:** Users can specify matrix dimensions and edit values.
- **Graph Representation:** Converts a matrix into a directed acyclic graph (DAG).
- **Shortest Path Calculation:** Uses topological sorting and dynamic programming.
- **Path Visualization:** Highlights the shortest path in the original matrix.
- **Web Interface:** Built with Streamlit for easy interaction.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   streamlit run general.py
   ```

## Usage
1. Open the Streamlit interface.
2. Set the number of rows and columns for the matrix.
3. Edit the matrix values if needed.
4. Click "Run" to compute the shortest path.
5. View the highlighted shortest path in the output matrix.

## Algorithms Used
### 1. **Matrix to Graph Conversion**
   - Each matrix element is represented as a graph node.
   - Directed edges connect valid movements downward.
   - Edge weights are calculated based on matrix values.

### 2. **Shortest Path Calculation**
   - **Topological Sort:** Orders the graph for efficient processing.
   - **Dynamic Programming:** Finds shortest paths in `O(V + E)`.
   - **Backtracking:** Reconstructs the shortest path.

## Example
**Input Matrix:**
```
1  2  3
4  5  6
7  8  9
```
**Shortest Path Output:**
```
1  0  0
0  5  0
0  0  9
```

## Requirements
- Python 3.8+
- NumPy
- Pandas
- Streamlit

## License
This project is open-source under the MIT License.

