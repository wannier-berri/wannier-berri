# this test only checks that printing of options is done without Exceptions
import wannierberri as wb


def test_print():
    versions = wb.welcome()
    for k in wb.welcome_message._needed_packages.keys():
        assert k in versions
