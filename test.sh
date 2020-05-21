# !bin/bash

set -e

run_tests() {
    TEST_CMD="pytest --showlocals --pyargs"

    set -x  # print executed commands to the terminal

    $TEST_CMD pymove/tests
}

run_tests
