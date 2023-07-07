#!/bin/env python3
import chess
import chess.pgn
import csv
import numpy as np
import pickle
from tqdm import tqdm  # progress bar
import sys


def repl():
    import code
    code.InteractiveConsole(locals=globals()).interact()


COLORS = [chess.WHITE, chess.BLACK]
PIECES = [chess.PAWN, chess.ROOK, chess.KNIGHT,
          chess.BISHOP, chess.QUEEN, chess.KING]


def board_to_onehot(board):
    ret = np.array([], dtype=bool)
    for color in COLORS:
        for piece in PIECES:
            ret = np.append(ret, board.pieces(piece, color).tolist())
    return ret


def board_to_heuristics(board):
    def count(ss):
        return sum(bool(x) for x in ss.tolist())

    def material(color):
        values = {chess.PAWN: 1,
                  chess.BISHOP: 3,
                  chess.KNIGHT: 3,
                  chess.BISHOP: 3,
                  chess.ROOK: 4,
                  chess.QUEEN: 9,
                  chess.KING: 0}
        return sum(values[piece] * count(board.pieces(piece, color))
                   for piece in PIECES)

    def threats(color):
        return sum(count(board.attackers(color, s))
                   for s in chess.SQUARES if board.color_at(s) != color)

    def space(color):
        return sum(count(board.attacks(s))
                   for s in chess.SQUARES if board.color_at(s) == color)

    def player(color):  # TODO: more features
        return [material(color), threats(color), space(color)]
    return np.array(player(chess.WHITE) + player(chess.BLACK))


def lc(filename):
    return len(open(filename).readlines())


def csv_to_boards(filename):
    win_boards = []
    loss_boards = []
    draw_boards = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        c_winner = header.index('winner')
        c_moves = header.index('moves')
        for row in tqdm(reader, ncols=lc(filename), desc='csv->boards'):
            winner = row[c_winner]
            board = chess.Board()
            drawn = winner == 'draw'
            win = winner == 'white'
            mirror = False
            for m in row[c_moves].split(' '):
                board.push_san(m)
                mirror = not mirror
                win = not win
                b = board.mirror() if mirror else board
                if drawn:
                    draw_boards.append(b)
                elif win:
                    win_boards.append(b)
                else:
                    loss_boards.append(b)
    return {'win': win_boards, 'loss': loss_boards, 'draw': draw_boards}


def phi(x):  # something in numpy probably does this faster
    return np.append(np.append(x, x*x), 1)


def pgn_to_boards(filename):
    win_boards = []
    loss_boards = []
    draw_boards = []
    with open(filename) as pgn:
        game = chess.pgn.read_game(pgn)
        # for _ in tqdm(range(10000)): # hack- getting 10000 games in huge file
        while game is not None:
            board = game.board()
            drawn = game.headers['Result'] == '1/2-1/2'
            win = game.headers['Result'] == '1-0'
            mirror = False
            for move in game.mainline_moves():
                board.push(move)
                mirror = not mirror
                win = not win
                b = board.mirror() if mirror else board
                if drawn:
                    draw_boards.append(b)
                elif win:
                    win_boards.append(b)
                else:
                    loss_boards.append(b)
            game = chess.pgn.read_game(pgn)
    return {'win': win_boards, 'loss': loss_boards, 'draw': draw_boards}


if __name__ == '__main__':
    n = len(sys.argv)
    if n == 1:
        print('Usage: ', sys.argv[0],
              ' [csv|pgn|onehot|heuristics|phi] <infile> <outfile>')
    cmd, infile, outfile = sys.argv[1:]
    if cmd == 'csv':
        boards = csv_to_boards(infile)
        pickle.dump(boards, open(outfile, 'wb'))
    elif cmd == 'pgn':
        boards = pgn_to_boards(infile)
        pickle.dump(boards, open(outfile, 'wb'))
    elif cmd == 'onehot' or cmd == 'heuristics':
        f = board_to_onehot if cmd == 'onehot' else board_to_heuristics
        boards = pickle.load(open(infile, 'rb'))
        for k in boards.keys():
            boards[k] = [np.array(f(x)) for x in
                         tqdm(boards[k], desc=f'{k} boards->{f.__name__}')]
        pickle.dump(boards, open(outfile, 'wb'))
    elif cmd == 'phi':
        boards = pickle.load(open(infile, 'rb'))
        for k in boards.keys():
            boards[k] = [phi(x) for x in tqdm(boards[k], desc=f'{k} phi')]
        pickle.dump(boards, open(outfile, 'wb'))
