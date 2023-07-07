#!/bin/env python3
import chess
import pickle
import sys
import pp

if __name__ == "__main__":
    n = len(sys.argv)
    if n == 1:
        print('Usage: ', sys.argv[0], ' <model> [onehot|heuristics]')
    ai, modeltype = sys.argv[1:]
    ai = pickle.load(open(ai, 'rb'))
    encode = None
    if modeltype == 'onehot':
        encode = pp.board_to_onehot
    elif modeltype == 'heuristics':
        encode = pp.board_to_heuristics
    else:
        print(f'model type {modeltype} not supported')
        exit()
    print('starting game:')
    board = chess.Board()
    turn = chess.WHITE
    print(board)
    print(ai.predict_proba([encode(board)]))
    while not board.is_game_over():
        input()
        if turn != chess.WHITE:
            board = board.mirror()
        best_move = (None, 0)
        for move in board.legal_moves:
            b = board.copy(stack=False)
            b.push(move)
            win = ai.predict_proba([encode(b.mirror())])[0][0]
            if win > best_move[1]:
                best_move = (move, win)
        board.push(move)
        if turn != chess.WHITE:
            board = board.mirror()
        print(board.peek())
        print(board)
        print(win)
