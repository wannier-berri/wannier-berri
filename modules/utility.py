__debug = True

import inspect

def print_my_name_start():
    if __debug: 
        print("Running {} ..".format(inspect.stack()[1][3]))


def print_my_name_end():
    if __debug: 
        print("Running {} ..".format(inspect.stack()[1][3]))
