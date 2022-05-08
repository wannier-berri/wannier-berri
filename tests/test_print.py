# this test only checks that printing of options is done without Exceptions
import wannierberri as wb


def test_print():
    wb.welcome()
    wb.__old_API.__main.print_options()
