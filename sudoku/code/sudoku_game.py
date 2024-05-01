import random

def print_board(board):
    """
    Function to print the Sudoku board.
    """
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            print(board[i][j], end=" ")
        print()

def generate_sudoku(difficulty):
    """
    Function to generate a random Sudoku puzzle based on the selected difficulty level.
    """
    board = [[0 for _ in range(9)] for _ in range(9)]
    if difficulty == "easy":
        filled_cells = 30
    elif difficulty == "medium":
        filled_cells = 25
    elif difficulty == "hard":
        filled_cells = 20
    else:
        print("Invalid difficulty level. Defaulting to easy.")
        filled_cells = 30
    
    while filled_cells > 0:
        row = random.randint(0, 8)
        col = random.randint(0, 8)
        if board[row][col] == 0:
            num = random.randint(1, 9)
            if is_valid(board, row, col, num):
                board[row][col] = num
                filled_cells -= 1
    return board

def is_valid(board, row, col, num):
    """
    Function to check if a number can be placed in a specific cell without violating any Sudoku rules.
    """
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[i + start_row][j + start_col] == num:
                return False
    return True

def get_hint(board):
    """
    Function to provide a hint to the user by filling at least one number in place of 0.
    """
    empty_cells = [(i, j) for i in range(9) for j in range(9) if board[i][j] == 0]
    if not empty_cells:
        print("No empty cell to provide hint.")
        return
    row, col = random.choice(empty_cells)
    num = random.randint(1, 9)
    while not is_valid(board, row, col, num):
        num = random.randint(1, 9)
    board[row][col] = num
    print(f"Hint: Fill {num} at row {row+1}, column {col+1}")

def is_incorrect(board, row, col, num):
    """
    Function to check if the entered number is incorrect based on Sudoku rules.
    """
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return True
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[i + start_row][j + start_col] == num:
                return True
    return False

def solve_sudoku(board):
    """
    Function to solve the entire Sudoku puzzle using backtracking.
    """
    empty_cell = find_empty_cell(board)
    if not empty_cell:
        return True
    row, col = empty_cell
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0
    return False

def find_empty_cell(board):
    """
    Function to find the next empty cell on the Sudoku board.
    """
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

def play_sudoku():
    """
    Function to implement interactive gameplay for Sudoku.
    """
    difficulty = input("Choose difficulty (easy/medium/hard): ").lower()
    sudoku_board = generate_sudoku(difficulty)
    solved_board = [row[:] for row in sudoku_board]  # Make a copy of the initial board
    solve_sudoku(solved_board)  # Solve the puzzle
    print("\nGenerated Sudoku Puzzle:")
    print_board(sudoku_board)

    while True:
        action = input("\nEnter 'stop' to stop entering numbers, or press Enter to continue: ").lower()
        if action == "stop":
            print("\nGame stopped.")
            return

        row = int(input("Enter row number (1-9): ")) - 1
        col = int(input("Enter column number (1-9): ")) - 1
        if not (0 <= row <= 8 and 0 <= col <= 8):
            print("Invalid row or column number. Please try again.")
            continue
        if sudoku_board[row][col] != 0:
            print("This cell is already filled. Please choose another one.")
            continue
        num = int(input("Enter the number (1-9) for this cell: "))
        if not (1 <= num <= 9):
            print("Invalid number. Please enter a number between 1 and 9.")
            continue
        if not is_valid(sudoku_board, row, col, num):
            print("Invalid number. This number violates Sudoku rules.")
            continue
        if num != solved_board[row][col]:
            print("Incorrect number. This number does not match the solution.")
            continue
        sudoku_board[row][col] = num
        print("\nUpdated Sudoku Puzzle:")
        print_board(sudoku_board)

        hint_requested = input("\nWould you like a hint? (yes/no): ").lower()
        if hint_requested == "yes":
            get_hint(sudoku_board)
            print("\nSudoku Puzzle with Hint:")
            print_board(sudoku_board)

        if all(0 not in row for row in sudoku_board):
            print("\nCongratulations! You've solved the Sudoku puzzle!")
            break

    print("\nSolved Sudoku Puzzle:")
    print_board(solved_board)

play_sudoku()
