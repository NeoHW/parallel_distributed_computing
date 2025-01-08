#!/bin/bash

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RESET="\033[0m"


# ensure correct usage
if [ $# -lt 3 ]; then
    echo "Usage: $0 <architecture> <path to input file> <num threads> [summarize output if this option is non empty]"
    echo "E.g 1. $0 i7-7700 tests/input.in 8"
    echo "E.g 2. $0 i7-7700 tests/input.in 8 x"
    exit 1
fi


studentexec="sim.perf"
mactype=$1
testfile=$2
numthreads=$3
summary=$4

# Build all targets
if [[ "x$summary" == "x" ]]; then
    make all
else
    make all > /dev/null
fi

# make cache folder
cachedir="./checker_cache"
mkdir -p $cachedir
mkdir -p ./out

# If the build fails, exit
if [[ $? -ne 0 ]]; then
    echo -e "${RED}Build failed${RESET}"
    exit 1
fi

# Run the simulation
BENCHES=$(ls bench-*.perf)
BENCHFASTER=()
IMPLFASTER=()

if [[ "x$summary" == "x" ]]; then
    echo "Running your code..."
fi

# always run student impl
usrtime=$(srun --partition=$mactype --time=00:10:00 perf stat -e task-clock -r 3 ./$studentexec $testfile $numthreads 2>&1 | grep "time elapsed" | awk '{print $1}')

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Your program terminated with exit code $?${RESET}"
    exit $?
fi

if [[ "x$summary" == "x" ]]; then
    echo Your runtime $usrtime
fi

for bench in ${BENCHES[@]}; do
    testfilename=$(echo $testfile | awk -F '/' '{print $NF}')

    cachefile="$testfilename-$bench-$numthreads-$mactype"

    # Check if cachefile exists and non-empty
    if [[ -s "$cachedir/$cachefile" ]]; then
        if [[ "x$summary" == "x" ]]; then
            echo "Cache file exists for $bench $mactype $testfile $numthreads, skipping run"
        fi
    else
        if [[ "x$summary" == "x" ]]; then
            echo "Running ./$bench $testfile $numthreads"
        fi
        srun --partition=$mactype --time=00:10:00 perf stat -e task-clock -r 3 ./$bench $testfile $numthreads 2>&1 | grep "time elapsed" | awk '{print $1}' > $cachedir/$cachefile
        if [[ $? -ne 0 ]]; then
            echo Benchmark terminated with exit code $?
            exit $?
        fi
    fi

    benchtime=$(cat $cachedir/$cachefile)
    # check if benchtime is a number
    if ! [[ $benchtime =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
        echo -e "${RED}ERROR! Bench time not found for $bench... terminating${RESET}"
        exit 1
    fi

    if [[ "x$summary" == "x" ]]; then
        echo Bench time for $bench: $benchtime
    fi
    
    if (( $(echo "$benchtime < $usrtime" | bc -l) )); then
        BENCHFASTER+=("$mactype-$bench")
    else
        IMPLFASTER+=("$mactype-$bench")
    fi
done

if [[ "x$summary" != "x" ]]; then
    total_length=$((${#BENCHFASTER[@]} + ${#IMPLFASTER[@]}))
    echo ${#IMPLFASTER[@]}/$total_length
else
    # summary output
    echo -e "${RED}These benchmarks on $mactype are faster than your implementation:${RESET}"
    for test in "${BENCHFASTER[@]}"; do
        echo -e "${RED}$test${RESET}"
    done

    echo -e "${GREEN}Your implementation on $mactype is faster than these benchmarks:${RESET}"
    for test in "${IMPLFASTER[@]}"; do
        echo -e "${GREEN}$test${RESET}"
    done
fi
