from collections import deque
import numpy as np
import streamlit as st
import pandas as pd

# Поиск кратчайших путей
def shortest_path(graph, start):
    order = list(graph.keys())
    dist = {node: float('inf') for node in graph}
    prev = {node: None for node in graph}  # Храним предыдущие вершины для восстановления пути
    dist[start] = 0
    
    for node in order:
        if dist[node] != float('inf'):
            for neighbor, weight in graph[node]:
                if dist[neighbor] > dist[node] + weight:
                    dist[neighbor] = dist[node] + weight
                    prev[neighbor] = node
    return dist, prev

# Функция восстановления пути
def reconstruct_path(prev, start, end):
    path = []
    while end is not None:
        path.append(end)
        end = prev[end]
    
    path = path[::-1][1:-1]  # Remove first and last elements
    
    formatted_path = [[int(node.split('/')[0][1:]), int(node.split('/')[1])]
                      for node in path]  # Extract numbers
    return formatted_path

# Преобразование матрицы в граф
def matrix_to_graph(matrix):
    graph = {} 
    rows, cols = len(matrix), len(matrix[0])
    num = cols - rows + 1
    max_element = max(map(max, matrix))

    graph['a0/0'] = []  # Initialize list
    for i in range(0, num):
        graph['a0/0'].append((f'a1/{i}', max_element - matrix[0][i]))

    for i in range(0, rows - 1):
        for j in range(i, min(i + num, cols)):  # Limit columns
            for k in range(j, min(j + num - (j - i), cols)):  # Reduce range size in each iteration
                node = f'a{i+1}/{j}'
                if node not in graph:
                    graph[node] = []
                graph[node].append((f'a{i+2}/{k+1}', max_element - matrix[i+1][k+1]))
    
    for i in range(cols - num - 1, cols):
        graph[f'a{rows}/{i}'] = []
        graph[f'a{rows}/{i}'].append((f'a{rows+1}/{0}', 0))
    
    graph[f'a{rows+1}/{0}'] = []
    return graph

st.title("Matrix To Path")

rows = st.number_input("Rows", min_value=1, step=1, value=3)
cols = st.number_input("Columns", min_value=1, step=1, value=3)

default_matrix = "\n".join([" ".join(["0"] * cols) for _ in range(rows)])

matrix_input = st.text_area("Edit the matrix:", default_matrix, height=150)

try:
    matrix = np.array([list(map(float, row.split())) for row in matrix_input.split("\n") if row])

    if matrix.shape != (rows, cols):
        st.warning(f"Matrix shape {matrix.shape} doesn't match the specified dimensions ({rows}, {cols})!")
    else:
        st.write("### Input Matrix:")
        st.write(pd.DataFrame(matrix))
        graph = matrix_to_graph(matrix)
        
        start_node = 'a0/0'
        end_node = next(reversed(graph))
        distances, predecessors = shortest_path(graph, start_node)

        shortest_path_to_end = reconstruct_path(predecessors, start_node, end_node)

        result_matrix = np.zeros_like(matrix)
        for i, j in shortest_path_to_end:
            result_matrix[i-1, j] = matrix[i-1][j]

        st.write("### Path:")
        st.write(pd.DataFrame(result_matrix))
        
except Exception as e:
    st.error(f"Invalid matrix format: {e}")
