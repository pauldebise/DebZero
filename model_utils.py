import chess
import numpy as np

PIECE_TO_CHANNEL = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}


def encode_board(board: chess.Board) -> np.ndarray:

    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)

        if piece:
            channel = PIECE_TO_CHANNEL[piece.symbol()]
            row = sq // 8
            col = sq % 8
            tensor[channel, row, col] = 1.0

    return tensor

def mirror_move(move: chess.Move) -> chess.Move:
    from_square = chess.square_mirror(move.from_square)
    to_square = chess.square_mirror(move.to_square)
    return chess.Move(from_square, to_square, promotion=move.promotion)

def encode_move(move: chess.Move) -> np.ndarray:
    policy = np.zeros(4096, dtype=np.float32)
    from_square = move.from_square
    to_square = move.to_square
    index = 64 * from_square + to_square
    policy[index] = 1
    return policy