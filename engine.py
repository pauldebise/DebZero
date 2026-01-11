import chess
from typing import Optional, Dict, List, Tuple, Any
from math import sqrt
from model_inference import initialize_session, onnx_model_inference
from time import perf_counter

class Node:

    __slots__ = ["move", "children", "prior_prob", "visits", "total_value", "parent"]

    def __init__(self, move: Optional[chess.Move] = None, parent: Optional["Node"] = None, prior_prob: float = 0.0) -> None:
        self.move = move
        self.parent = parent
        self.prior_prob = prior_prob
        self.children: Dict[chess.Move, "Node"] = {}
        self.visits: int = 0
        self.total_value: float = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_value(self) -> float:
        return self.total_value / (self.visits + 1e-6)

    def select_child(self, cpuct: float = 1.4) -> "Node":
        best_score = -float("inf")
        best_child: Optional["Node"] = None

        for child in self.children.values():
            q = -child.get_value()
            u = cpuct * child.prior_prob * sqrt(self.visits) / (child.visits + 1e-6) ** 0.75
            score = q + u

            if score > best_score:
                best_score = score
                best_child = child

        return best_child


def king_distance_heuristic(board: chess.Board) -> float:
    if board.occupied.bit_count() >= 5:
        return 0.0

    our_king = board.king(board.turn)
    their_king = board.king(not board.turn)
    our_rank, our_file = chess.square_rank(our_king), chess.square_file(our_king)
    their_rank, their_file = chess.square_rank(their_king), chess.square_file(their_king)

    dist_center_rank = max(3 - their_rank, their_rank - 4)
    dist_center_file = max(3 - their_file, their_file - 4)
    center_distance = dist_center_rank + dist_center_file
    dist_kings = abs(their_rank - our_rank) + abs(their_file - our_file)

    score = 0.1 * center_distance - 0.05 * dist_kings
    return score


def material_heuristic(board: chess.Board) -> float:
    piece_value = {chess.PAWN : 0.02, chess.BISHOP : 0.06, chess.KNIGHT : 0.06, chess.ROOK : 0.10, chess.QUEEN : 0.18}

    score = 0.0
    for _, piece in board.piece_map().items():
        score += piece_value.get(piece.piece_type, 0) * (1 if piece.color else -1)

    return score if board.turn == chess.WHITE else -score


class Engine:

    def __init__(self, model_path, nb_simulations: int = 1000, search_time: float = 1, cpuct: float = 1.4) -> None:
        self.nb_simulations = nb_simulations
        self.search_time = search_time
        self.cpuct: float = cpuct
        self.ort_session, self.input_name = initialize_session(model_path)

    def expand_node(self, node: Node, board: chess.Board, policy_prob: Dict[chess.Move, float]) -> None:

        legal_moves = list(board.legal_moves)

        total_prob = sum(policy_prob.get(move, 0.0) for move in legal_moves)
        for move in legal_moves:
            prob = policy_prob.get(move, 0.0) / total_prob
            node.children[move] = Node(move, node, prob)

    def backpropagate(self, search_path: List[Node], value: float) -> None:
        for node in search_path[::-1]:
            node.visits += 1
            node.total_value += value
            value = -value

    def get_best_move(self, root: Node) -> Tuple[chess.Move, Dict[chess.Move, float]]:

        visit_count = {move: child.visits for move, child in root.children.items()}
        total_visits = sum(visit_count.values())
        visit_prob = {move: visit / total_visits for move, visit in visit_count.items()}

        best_move = max(visit_count.items(), key=lambda x: x[1])[0]
        return best_move, visit_prob

    def get_pv_line(self, node: Node, depth_limit: int = 10) -> List[chess.Move]:
        pv = []
        current = node

        for _ in range(depth_limit):

            if current.is_leaf():
                break

            best_move, best_child = max(current.children.items(), key=lambda x: x[1].visits)
            pv.append(best_move)

            if best_child.visits == 0:
                break

            current = best_child

        return pv

    def search(self, board: chess.Board) -> tuple[Any, Any, list[chess.Move]]:

        t_start = perf_counter()
        root = Node()
        policy_prob, _, _ = onnx_model_inference(self.ort_session, self.input_name, board)
        dict_policy_prob = {elem[1]: elem[0] for elem in policy_prob}
        self.expand_node(root, board, dict_policy_prob)

        for node_count in range(self.nb_simulations):

            if node_count % 300 == 0 and node_count != 0 or self.search_time <= perf_counter()-t_start:
                children_values = [child.get_value() for child in root.children.values()]
                score = max(children_values)
                search_duration = perf_counter() - t_start
                pv_line = self.get_pv_line(root, depth_limit=10)
                pv_str = " ".join([m.uci() for m in pv_line])
                print(f"info depth {len(pv_line)} score cp {int(1500 * score) * (1 if board.turn else -1)} nodes {node_count} nps {int(node_count//search_duration)} time {int(1000 * search_duration)} pv {pv_str}", flush=True)

            if perf_counter() - t_start >= self.search_time:
                break

            node = root
            search_path = [node]

            while not node.is_leaf() and not board.is_game_over():
                node = node.select_child()
                board.push(node.move)
                search_path.append(node)

            if board.is_game_over():
                result = board.result()
                if result == "1-0":
                    winner = 1
                elif result == "0-1":
                    winner = -1
                else:
                    winner = 0

                current_player = 1 if board.turn == chess.WHITE else -1

                if winner == 0:
                    value = 0.0
                else:
                    current_depth = len(search_path)
                    value = float(winner * current_player)*(0.98**current_depth)

            elif board.halfmove_clock >= 4 and board.can_claim_draw():
                value = 0.0

            elif len(list(board.legal_moves)) == 0 and board.is_stalemate():
                value = 0.0

            else:
                # Sinon, utiliser le réseau de neurones
                policy_probs, value, time = onnx_model_inference(self.ort_session, self.input_name, board)
                material_bonus = material_heuristic(board)
                value += material_bonus
                if material_bonus > 0.1:
                    value += king_distance_heuristic(board)
                value *= (1 - board.halfmove_clock/100)
                d_policy_probs = {elem[1]: elem[0] for elem in policy_probs}
                self.expand_node(node, board, d_policy_probs)

            for _ in range(len(search_path)-1):
                board.pop()
            self.backpropagate(search_path, value)

        best_move = self.get_best_move(root)
        return *best_move, self.get_pv_line(root)




if __name__ == '__main__':
    fen = "r2q2k1/3b1ppp/1pp2b2/2pQ1n2/2P5/3P4/PPN2PPP/R1B1rBK1 w - - 0 21"
    board = chess.Board(fen)
    model_path = r"model2/model2_onnx_fp32.onnx"
    engine = Engine(model_path, 10000, 4.5, 1.4)
    print(engine.search(board)[0].uci())