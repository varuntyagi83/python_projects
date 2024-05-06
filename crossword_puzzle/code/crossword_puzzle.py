import random
import pandas as pd

def create_crossword_puzzle(data):
    # Extract clues and answers from the dataset
    clues = data['clue'].tolist()
    answers = data['answer'].tolist()

    # Ensure dataset contains enough words
    if len(clues) < 5:
        return None, None

    while True:
        try:
            # Select 5 random words from the dataset
            selected_indices = random.sample(range(len(clues)), 5)
            selected_clues = [clues[i] for i in selected_indices]
            selected_answers = [answers[i] for i in selected_indices]

            # Generate a 10x10 grid with blanks
            grid = [[' ' for _ in range(10)] for _ in range(10)]

            # Define the pattern for black and white squares (0 for white, 1 for black)
            pattern = [
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]

            # Place words horizontally and vertically
            word_positions = []  # Store the positions of placed words
            occupied_cells = set()  # Track occupied cells
            for i, word in enumerate(selected_answers):
                direction = random.choice(['horizontal', 'vertical'])
                if direction == 'horizontal':
                    row = random.randint(0, 9)
                    col = random.randint(0, 9 - len(word))
                    while any((row, col + j) in occupied_cells and grid[row][col + j] != word[j] for j in range(len(word))):
                        row = random.randint(0, 9)
                        col = random.randint(0, 9 - len(word))
                    for j in range(len(word)):
                        grid[row][col + j] = '_'
                        occupied_cells.add((row, col + j))
                    word_positions.append((row, col, direction, len(word), selected_clues[i], word))
                else:
                    row = random.randint(0, 9 - len(word) + 1)  # Adjusted range
                    col = random.randint(0, 9)
                    while any((row + j, col) in occupied_cells and grid[row + j][col] != word[j] for j in range(len(word))):
                        row = random.randint(0, 9 - len(word) + 1)
                        col = random.randint(0, 9)
                    for j in range(len(word)):
                        grid[row + j][col] = '_'
                        occupied_cells.add((row + j, col))
                    word_positions.append((row, col, direction, len(word), selected_clues[i], word))

            # Apply black and white squares pattern
            for i in range(10):
                for j in range(10):
                    if pattern[i][j] == 1:
                        grid[i][j] = '#'

            return grid, word_positions, occupied_cells

        except ValueError:
            # Retry if ValueError occurs (empty range for randrange)
            continue


def print_empty_grid(grid):
    print("Crossword Puzzle:")
    print("   ", end="")
    for col_num in range(1, 11):
        print(f"{col_num} ", end="")
    print()
    for row_num, row in enumerate(grid, start=1):
        print(f"{row_num:2} ", end="")
        for col in row:
            if col == '#':
                print('# ', end="")
            else:
                print('  ', end="")
        print()
    print()


def print_filled_grid(grid):
    print("\nCurrent Progress:")
    print("   ", end="")
    for col_num in range(1, 11):
        print(f"{col_num} ", end="")
    print()
    for row_num, row in enumerate(grid, start=1):
        print(f"{row_num:2} ", end="")
        for col in row:
            if col == '#':
                print('# ', end="")
            else:
                print(col + ' ', end="")
        print()
    print()


def generate_crossword_from_dataset(data):
    grid, word_positions, occupied_cells = create_crossword_puzzle(data)

    if grid is None or word_positions is None:
        return "Sorry, dataset does not contain enough words."

    print_empty_grid(grid)

    # Prompt user for guesses
    for row, col, direction, word_len, clue, answer in word_positions:
        if direction == 'horizontal':
            question = f"Across ({row + 1}, {col + 1}): {clue}"
        else:
            question = f"Down ({row + 1}, {col + 1}): {clue}"
        
        while True:
            guess = input(question + ": ").strip().lower()
            if guess == answer.lower():
                print("Correct!")
                break
            else:
                print("Incorrect. Please try again.")

        # Update grid with correct guesses
        for i, char in enumerate(answer):
            if direction == 'horizontal':
                grid[row][col + i] = char
            else:
                grid[row + i][col] = char

        print_filled_grid(grid)  # Print the filled grid after each correct answer

    # Print the clues and answers
    print("\nCorrect Answers:")
    for clue, answer in zip([item[4] for item in word_positions], [item[5] for item in word_positions]):
        print(clue + ":", answer)

generate_crossword_from_dataset(data)
