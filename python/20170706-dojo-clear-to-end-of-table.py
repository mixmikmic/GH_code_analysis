get_ipython().run_cell_magic('script', 'bash', '\n# Ignore this boring cell.\n# It allows one to do C in Jupyter notebook.\n\ncat >20170706_head.c <<EOF\n#include <stdlib.h>\n#include <stdio.h>\n\n#define LINES (3)\n#define COLUMNS (4)\n\nvoid print_buf(char buf[LINES][COLUMNS])\n{\n    for (int row = 0; row < LINES; row++) {\n        for (int column = 0; column < COLUMNS; column++)\n            putchar(buf[row][column]);\n        putchar(\'\\n\');\n    }\n}\nEOF\ncat >20170706_tail.c <<EOF\n\nint main(int argc, char *argv())\n{\n    char buf[LINES][COLUMNS];\n    \n    for (int row = 0; row < LINES; row++)\n        for (int column = 0; column < COLUMNS; column++)\n            buf[row][column] = \'@\';\n    blank(buf, 1, 2);\n    print_buf(buf);\n}\nEOF\n\nprogram_name="${PATH%%:*}/20170706_c_foo"\necho $program_name\ncat >"$program_name" <<EOF\n#!/usr/bin/env sh\ncat 20170706_head.c >20170706_blank.c\ncat >>20170706_blank.c\ncat 20170706_tail.c >>20170706_blank.c\ncc     20170706_blank.c   -o 20170706_blank\n./20170706_blank | tr \' \' \'.\'\nEOF\nchmod +x "$program_name"')

get_ipython().run_cell_magic('script', '20170706_c_foo', "\nvoid blank(char buf[LINES][COLUMNS], int row, int column)\n{\n    goto MIDDLE;\n\n    for ( ; row < LINES; row++)\n        for (column = 0; column < COLUMNS; column++)\n            MIDDLE: buf[row][column] = ' ';\n}")

get_ipython().run_cell_magic('script', '20170706_c_foo', "\nvoid blank(char buf[LINES][COLUMNS], int row, int column)\n{\n    for ( ; row < LINES; row++) {\n        for ( ; column < COLUMNS; column++)\n            buf[row][column] = ' ';\n        column = 0;\n    }\n}")

get_ipython().run_cell_magic('script', '20170706_c_foo', "\nvoid blank_to_end_of_row(char buf[LINES][COLUMNS], int row, int column)\n{\n    for ( ;column < COLUMNS; column++)\n        buf[row][column] = ' ';\n}\n\nvoid blank_row(char buf[LINES][COLUMNS], int row)\n{\n    blank_to_end_of_row(buf, row, 0);\n}\n\nvoid blank(char buf[LINES][COLUMNS], int row, int column)\n{\n    blank_to_end_of_row(buf, row++, column);\n\n    for ( ; row < LINES; row++)\n        blank_row(buf, row);\n}")

get_ipython().run_cell_magic('script', '20170706_c_foo', "\nvoid blank_to_end_of_row(char buf[LINES][COLUMNS], int row, int column)\n{\n    for ( ;column < COLUMNS; column++)\n        buf[row][column] = ' ';\n}\n\nvoid blank(char buf[LINES][COLUMNS], int row, int column)\n{\n    blank_to_end_of_row(buf, row++, column);\n\n    for ( ; row < LINES; row++)\n        blank_to_end_of_row(buf, row, 0);\n}")

LINES = 3
COLUMNS = 4

def foo(row=1, column=2):
    buf = [
        ['@' for _ in range(COLUMNS)]
        for _ in range(LINES)
    ]
    blank(buf, row, column)
    for row in buf:
        print(''.join(row).replace(' ', '.'))

def blank(buf, row, column):
    for row in range(row, LINES):
        for column in range(column, COLUMNS):
            buf[row][column] = ' '
        column = 0

foo()

def blank_to_end_of_row(buf, row, column):
    for column in range(column, COLUMNS):
        buf[row][column] = ' '

def blank_row(buf, row):
    blank_to_end_of_row(buf, row, 0)

def blank(buf, row, column):
    blank_to_end_of_row(buf, row, column)
    row += 1

    for row in range(row, LINES):
        blank_row(buf, row)

foo()

def blank_to_end_of_row(buf, row, column):
    for column in range(column, COLUMNS):
        buf[row][column] = ' '

def blank(buf, row, column):
    blank_to_end_of_row(buf, row, column)
    row += 1

    for row in range(row, LINES):
        blank_to_end_of_row(buf, row, 0)

foo()

get_ipython().run_cell_magic('script', '20170706_c_foo', "\n/* this is wrong: fails to clear beginning of following lines */\n\nint blank(char buf[LINES][COLUMNS], int row_arg, int column_arg)\n{\n    int row;\n    int column;\n\n    for (row = row_arg; row < LINES; row++)\n        for (column = column_arg; column < COLUMNS; column++)\n            buf[row][column] = ' ';\n}")

