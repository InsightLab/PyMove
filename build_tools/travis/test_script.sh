set -e

run_tests() {
    TEST_CMD="pytest --showlocals --pyargs"

    mkdir -p $TEST_DIR

    cp setup.cfg $TEST_DIR
    cd $TEST_DIR 

    set -x  # print executed commands to the terminal

    $TEST_CMD pymove/pymove/tests
}

run_tests