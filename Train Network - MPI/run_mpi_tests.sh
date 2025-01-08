#!/bin/bash

# Directory to store outputs
OUTPUT_DIR="outputs"
TESTCASE_DIR="testcases"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Testcase file names
TESTCASES=(
    "stations_30.in"
    "stations_800.in"
    "stations_1600.in"
    "stations_3200.in"
    "stations_6400.in"
    "stations_12800.in"
    "trains_500.in"
    "trains_1000.in"
    "trains_2000.in"
    "trains_4000.in"
    "trains_8000.in"
    "trains_16000.in"
    "ticks_500.in"
    "ticks_4000.in"
    "ticks_8000.in"
    "ticks_16000.in"
    "ticks_32000.in"
    "ticks_64000.in"
)

# Partitions and configurations
PARTITIONS=(
    "i7-7700"
    "xs4114"
)

# Nodes and tasks
NODES=1
NTASKS=4

# Executable file
EXECUTABLE="./trains"

# Loop through each partition
for PARTITION in "${PARTITIONS[@]}"; do
    echo "Running on partition: $PARTITION"

    # Loop through each testcase
    for TESTCASE in "${TESTCASES[@]}"; do
        TESTCASE_PATH="$TESTCASE_DIR/$TESTCASE"
        OUTPUT_PATH="$OUTPUT_DIR/${TESTCASE%.in}_${PARTITION}_ntasks${NTASKS}.out"
        
        echo "Processing testcase: $TESTCASE"

        # Allocate resources and run the program
        salloc -p "$PARTITION" --nodes "$NODES" --ntasks "$NTASKS" mpirun "$EXECUTABLE" < "$TESTCASE_PATH" > "$OUTPUT_PATH"

        # Check if execution succeeded
        if [ $? -eq 0 ]; then
            echo "Finished: $TESTCASE on $PARTITION"
        else
            echo "Error running: $TESTCASE on $PARTITION" >&2
        fi
    done
done

echo "All tests completed."
