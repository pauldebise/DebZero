from model_utils import encode_board, encode_move, mirror_move
import chess
import numpy as np
from time import perf_counter
import onnxruntime as ort

def initialize_session(model_path):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    return ort_session, input_name

def onnx_model_inference(session, input_name, board) -> tuple[list[tuple[float, chess.Move]], float, float]:

    if board.turn == chess.WHITE:
        board_to_encode = board
    else:
        board_to_encode = board.mirror()

    encoded_board = encode_board(board_to_encode).astype(np.float32)
    encoded_board = np.expand_dims(encoded_board, 0)

    t_start = perf_counter()

    outputs = session.run(None, {input_name: encoded_board})

    t_end = perf_counter()

    output_0 = outputs[0]
    output_1 = outputs[1]

    policy_arr = output_0
    value_arr = output_1

    policy = policy_arr.reshape(4096)
    value = value_arr.item()

    move_list = list(board_to_encode.legal_moves)
    proba_moves = []

    for move in move_list:
        move_vector = encode_move(move)
        move_index = np.argmax(move_vector)
        move_probability = policy[move_index]
        proba_moves.append((move_probability, move))

    if board.turn == chess.BLACK:
        proba_moves = [(p, mirror_move(m)) for (p, m) in proba_moves]

    return proba_moves, value, t_end - t_start



if __name__ == "__main__":

    model_path = r"model2/model2_onnx_fp32.onnx"

    ort_session, input_name = initialize_session(model_path)


    board = chess.Board()
    onnx_model_inference(ort_session, input_name, board)


    test_duration = 2
    total_time = 0
    n_loops = 0


    while True:
        moves, val, dt = onnx_model_inference(ort_session, input_name, board)
        total_time += dt
        n_loops += 1

        if total_time > test_duration:
            break


    avg_time_ms = (total_time / n_loops) * 1000


    print(f"Test de performance de l'inference du modèle {model_path}.")
    print(f"Temps MOYEN par inférence : {avg_time_ms:.3f} ms")
    print(f"Inférences par seconde (IPS) : {(1000 / avg_time_ms):.1f}")