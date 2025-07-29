import math

def minimax(xState, yState, isAIturn):
    winner = checkWin(xState, yState)
    if winner == "O":
        return 10 # gives score to the ai if wins
    if winner == "X":
        return -10 # negative score to the ai if X wins
    if checkDraw(xState, yState):
        return 0 # return 0 if drow happens

    if isAIturn:
        bestScore = -math.inf # initialising with -infinity
        for i in range(9):
            if xState[i] == 0 and yState[i] == 0:
                yState[i] = 1
                score = minimax(xState, yState, False)
                yState[i] = 0
                bestScore = max(bestScore, score)
        return bestScore
    else:
        bestScore = math.inf # initialising with infinity
        for i in range(9):
            if xState[i] == 0 and yState[i] == 0:
                xState[i] = 1
                score = minimax(xState, yState, True)
                xState[i] = 0
                bestScore = min(bestScore, score)
        return bestScore


def bestMove(xState, yState):
    bestScore = -math.inf
    move = None
    for i in range(9):
        if xState[i] == 0 and yState[i] == 0: # checking the empty cell
            yState[i] = 1 # initilising the initial value of ai move
            score = minimax(xState, yState, False) # finding out max score when no ai turn nesecary because evry human turn and ai turn must be calculate
            yState[i] = 0 # taking to the original state
            if score > bestScore:
                bestScore = score
                move = i # tells that of which the the move is of
    yState[move] = 1
    return yState




def printBoard(xState, yState):
    def cell(i):
        if xState[i]:
            return 'X'
        elif yState[i]:
            return 'O'
        else:
            return str(i+1)

    print(f"{cell(0)} | {cell(1)} | {cell(2)}")
    print("--+---+--")
    print(f"{cell(3)} | {cell(4)} | {cell(5)}")
    print("--+---+--")
    print(f"{cell(6)} | {cell(7)} | {cell(8)}")


def checkDraw(xState, yState):
    if all(xState[i] or yState[i] for i in range(9)):
        if checkWin(xState, yState) is None:
            return True
    return False


def checkWin(xState, yState):
    wins = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]

    for combo in wins:
        if xState[combo[0]] and xState[combo[1]] and xState[combo[2]]:
            return "X"

    for combo in wins:
        if yState[combo[0]] and yState[combo[1]] and yState[combo[2]]:
            return "O"
    return None


if __name__ == '__main__':
    xState = [0,0,0,0,0,0,0,0,0]
    yState = [0,0,0,0,0,0,0,0,0]
    turn = 1

    print('Welcome to Tic Tac Toe Game')
    while True:
        winner = checkWin(xState, yState)
        drow = checkDraw(xState, yState)

        if drow is not None:
            if drow is True:
                printBoard(xState, yState)
                print("Draw")
                break

        if winner:
            printBoard(xState, yState)
            print(f"{winner} wins!")
            break

        if turn == 1:
            print("X's Chance")
            value = int(input("Enter a value (0-8): "))
            xState[value-1] = 1
        else:
            print("O's Chance")
            yState = bestMove(xState, yState)

        turn = not turn

        printBoard(xState, yState)
