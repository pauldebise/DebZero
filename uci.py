from engine import Engine
import chess
import os


def get_absolute_path(relative_path):
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def position(arg: str) -> chess.Board:
    arg_list = arg.split(" ")
    if arg_list[0] == 'fen':
        board = chess.Board(' '.join(arg_list[1:7]))
    elif "startpos" in arg_list:
        board = chess.Board()
    else:
        board = chess.Board()
    if 'moves' in arg_list:
        for move in arg_list[arg_list.index('moves')+1:]:
            board.push(chess.Move.from_uci(move))

    return board


def uci():
    print("id name DebZero_3.1", flush=True)
    print("id author Paul Debise", flush=True)
    print("uciok", flush=True)


def isready():
    print("readyok", flush=True)



def go(arg: str, state_board: chess.Board):
    arg_list = arg.split(" ")
    search_board = state_board.copy()

    if 'nodes' in arg_list:
        try:
            index = arg_list.index('nodes') + 1
            nodes = int(arg_list[index])
        except (ValueError, IndexError):
            nodes = 1000
    else:
        nodes = 15_000

    if "wtime" in arg_list:
        wtime = int(arg_list[arg_list.index("wtime")+1])
    else:
        wtime = None
    if "btime" in arg_list:
        btime = int(arg_list[arg_list.index("btime")+1])
    else:
        btime = None
    if "winc" in arg_list:
        winc = int(arg_list[arg_list.index("winc")+1])
    else:
        winc = None
    if "binc" in arg_list:
        binc = int(arg_list[arg_list.index("binc")+1])
    else:
        binc = None
    if "movestogo" in arg_list:
        movestogo = int(arg_list[arg_list.index("movestogo")+1])
    else:
        movestogo = 0
    if "movetime" in arg_list:
        movetime = int(arg_list[arg_list.index("movetime")+1])
    else:
        movetime = None

    search_time = time_scheduler(state_board.turn, wtime, btime, winc, binc, movestogo, movetime)


    engine = Engine(
        model_path,
        nb_simulations=nodes,
        search_time=search_time,
        cpuct=1.2)

    best_move, _, pv_line = engine.search(search_board)
    ponder_move = pv_line[1] if len(pv_line)>=2 else None
    if ponder_move is not None:
        print(f"bestmove {best_move} ponder {ponder_move}", flush=True)
    else:
        print(f"bestmove {best_move}", flush=True)


def time_scheduler(side_to_move: bool, wtime=None, btime=None, winc=None, binc=None, movestogo=0, movetime=None, default=3):
    if movetime:
        t = movetime
    else:

        if side_to_move:
            our_time = wtime if wtime is not None else 0
            our_inc = winc if winc is not None else 0
        else:
            our_time = btime if btime is not None else 0
            our_inc = binc if binc is not None else 0
        if our_time == 0:
            return default

        if movestogo > 0:
            t = our_time // movestogo + our_inc //2
        else:
            t = our_time // 15 + our_inc//2

    t_min = 15
    return max(t - 15, t_min)/1000




if __name__ == '__main__':
    state_board = chess.Board()
    model_path = get_absolute_path(r"model2\model2_onnx_fp32.onnx")

    while True:
        command = input()
        command_list = command.split(" ")
        if command == "uci":
            uci()

        elif command == "isready":
            isready()

        elif command == "ucinewgame":
            state_board = chess.Board()

        elif command_list[0] == "position":
            state_board = position(" ".join(command_list[1:]))

        elif command == "d":
            print(state_board)

        elif command_list[0] == "go":
            go(" ".join(command_list[1:]), state_board)

        elif command == "quit" or command == "exit":
            break
