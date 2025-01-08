#!/usr/bin/env python3

import argparse

from random import randint, seed, shuffle


def gen_stations(stations: "list[str]", max_line_len: int) -> "list[str]":
    stns = list(stations)
    max_line_len = min(max_line_len, len(stns))
    r = randint(max(max_line_len - 3, 2), max(max_line_len, 2))

    shuffle(stns)
    return stns[:r]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate test cases for the metro problem."
    )
    parser.add_argument("S", type=int, help="metro stations in the network (max 17576)")
    parser.add_argument(
        "max_popularity", type=int, help="greatest possible popularity to be generated"
    )
    parser.add_argument(
        "max_link_weight",
        type=int,
        help="greatest possible link weight to be generated",
    )
    parser.add_argument(
        "max_num_trains",
        type=int,
        help="maximum no. of trains per line to be generated",
    )
    parser.add_argument(
        "max_line_len",
        type=int,
        help="maximum no. of stations per line to be generated",
    )
    parser.add_argument("N", type=int, help="ticks to run for the simulation")
    parser.add_argument(
        "--seed", type=int, default=42069, help="seed to feed random generator"
    )
    parser.add_argument(
        "--num_ticks_to_print",
        type=int,
        default=5,
        help="number of ticks (from the end) to output simulation state",
    )
    parser.add_argument(
        "--num_train_lines",
        type=int,
        default=3,
        help="no. of train lines to be generated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed(args.seed)

    S = min(17576, args.S)
    print(S)
    print(args.num_train_lines)

    Smap = {}

    # stations
    stations = [""] * S
    for i in range(S):
        if i:
            print(" ", end="")
        vs = []
        x = i
        # assume extreme upper bound of 17576 stations, but our actual testcases will be nowhere near it xD
        # just note that the testcases we will be using to test will execute on bench_seq within 1.5 minutes on a i7-7700 machine.
        for _ in range(3):
            vs.append(x % 26)
            x //= 26
        vs.reverse()
        for v in vs:
            stations[i] += chr(97 + v)
        print(stations[i], end="")
        Smap[stations[i]] = i
    print()

    # popularities
    for i in range(S):
        if i:
            print(" ", end="")
        print(randint(1, args.max_popularity), end="")
    print()

    # adjacency matrix
    mat = [[0] * S for _ in range(S)]
    for i in range(S):
        for j in range(S):
            mat[i][j] = randint(1, args.max_link_weight) if i < j else mat[j][i]

    # lines and bitmap for whether links should exist
    bitmap = [[0] * S for _ in range(S)]
    lines = ""
    for i in range(args.num_train_lines):
        currline = gen_stations(stations, args.max_line_len)
        prev = -1
        for s in currline:
            if prev != -1:
                bitmap[Smap[s]][prev] = 1
                bitmap[prev][Smap[s]] = 1
            prev = Smap[s]

        lines += " ".join(currline) + "\n"

    # printing adjacency matrix
    for i in range(S):
        for j in range(S):
            if j:
                print(" ", end="")
            if bitmap[i][j]:
                print(mat[i][j], end="")
            else:
                print(0, end="")
        print()

    print(lines, end="")

    def ml(): return randint(max(0, args.max_num_trains - args.num_train_lines), args.max_num_trains)

    print(args.N)
    print(' '.join([str(ml()) for _ in range(args.num_train_lines)]))
    print(args.num_ticks_to_print)


if __name__ == "__main__":
    main()