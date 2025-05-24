def is_valid(grid, row, col, num):
    """Check if placing `num` at `(row, col)` is valid."""
    # Check row and column
    for i in range(9):
        if grid[row][i] == num or grid[i][col] == num:
            return False
    
    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if grid[box_row + i][box_col + j] == num:
                return False
    return True

def find_empty_cell(grid):
    """Find the next empty cell (marked as 0)."""
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                return (i, j)
    return None

def backtracking_solve(grid):
    """Solve Sudoku using backtracking."""
    empty = find_empty_cell(grid)
    if not empty:
        return True  # Puzzle solved
    row, col = empty

    for num in range(1, 10):
        if is_valid(grid, row, col, num):
            grid[row][col] = num
            if backtracking_solve(grid):
                return True
            grid[row][col] = 0  # Backtrack

    return False