board = [
    [3, 0, 6, 5, 0, 8, 4, 0, 0],
    [5, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 7, 0, 0, 0, 0, 3, 1],
    [0, 0, 3, 0, 1, 0, 0, 8, 0],
    [9, 0, 0, 8, 6, 3, 0, 0, 5],
    [0, 5, 0, 0, 9, 0, 6, 0, 0],
    [1, 3, 0, 0, 0, 0, 2, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 7, 4],
    [0, 0, 5, 2, 0, 6, 3, 0, 0],
]

# prints the given sudoku board with clear indications of a 3x3 box
def print_board(bd):
    for i in range(len(bd)): # prints horizontal border
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - -")

        for j in range(len(bd[0])): # prints vertical border
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(bd[i][j])
            else:
                print(str(bd[i][j]) + " ", end="")

# locates where a given 0 is in the sudoku board
def find_zero(bd):
    for i in range(len(bd)):
        for j in range(len(bd[0])):
            if bd[i][j] == 0:
                return (i, j)
    return None

# makes sure the number is valid
def num_valid(bd, num, pos):
    for i in range(len(bd[0])):
        if bd[pos[0]][i] == num and pos[1] != i:
            return False

    for i in range(len(bd)):
        if bd[i][pos[1]] == num and pos[0] != i:
            return False

    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y * 3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bd[i][j] == num and (i,j) != pos:
                return False
    
    return True
    
def solution(bd):
    find = find_zero(bd)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1, 10):
        if num_valid(bd, i, (row, col)):
            bd[row][col] = i
            if solution(bd):
                return True
            bd[row][col] = 0
    return False

print("unsolved: ")
print_board(board)
print("\nsolved using backtrack:")
solution(board)
print_board(board)
    

