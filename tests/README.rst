==========================
Wannier Berri Test Suite
==========================
The test suite uses the ``pytest`` module.
When commit is pushed to a GitHub repository, the test suite is automatically run by GitHub Actions.
You can click the codecov badge in README to find out which part of the code is covered and which is not.

Dependencies
==========================
To run the test suite, one needs the ``pytest`` module, that can be installed e.g. via ``pip install --user pytest``.

How to run the tests
==========================
Running the test
--------------------------
In the ``tests`` folder, run ``pytest``.
You can run a specific test by giving its name as argument: ``pytest test_kubo.py``.
Run ``pytest -h`` for further information about ``pytest``.

Cleaning up
--------------------------
After running tests, many files are created.
These output files should be listed in the ``tests/.gitignore`` file.

These ignored files can be deleted by ``git clean -Xd -f``.
Be careful because it will delete all ignored files, including those not related to tests.
Interactive cleaning can be used by ``git clean -Xd -i``.

How to write a new test
==========================

Brief introduction to pytest
-----------------------------
* Fixtures
    Fixtures are functions that arranges data for the tests.     A function is marked as a fixture by the ``@pytest.fixture`` decorator.
    To use a fixture in a test or in another fixture, the name of the fixture should be included in the argument of the function where it is used.

    The results of the fixture is reused within its scope.
    The default scope is function. For example, the fixture of ``system_Fe_W90`` fixture is ``session``. So, the ``System`` object is computed only once during the test.

* Test functions
    Tests are written as functions. The name of each test function should start with ``test_``.
    Variables can be tested using the ``assert`` statement.

Recommended styles
-----------------------------
* Objects that can be shared by multiple tests should be implemented in ``common_*.py``. Then, import the object in ``conftest.py``. The objects does not need to be imported in individual ``test_*.py`` files.

* Do not use fixtures for simple constants such as ``Efermi_*`` or ``symmetries_*``. Use fixtures if some tests change the values of the variable, so it should be reset every time.

* Use fixtures only if 1) it takes a long time or should not be run multiple times, or 2) test error (assert) can be raised during the execution of that part, or 3) to parametrize tests, or 4) some tests change the value of the variable so the variable should be reset each time.

How to write a test
-----------------------------
1. Create/choose a system
    A ``System`` object can be used from the list of fixtures.
    If you need to create a new ``System`` object, you need to write a new fixture.

    * **Creating a new ab initio data:** If you add an additional dataset for the test computed using an external program (such as Quantum ESPRESSO), create a new folder in ``tests/data`` and include all the input and pseudopotential files to regenerate the data. Also, compress large text files such as the ``*.mmn`` and ``*.amn`` files. If possible, do not add large data files such as the ``uHu``, ``sHu``, and ``sIu`` files to the repository; they can be created inside the system fixture using the ``mmn2uHu`` utility.

    * **Creating a new System from Model:** Models (created through the ``wb_models`` interface) should be listed in ``common_systems.py``, and named ``model_*``. Models should be an ordinary variable rather than a fixture because they are never modified. Then, create a ``system_*`` fixture.

2. Create/choose a parser
    When comparing the ``*.dat`` files written from an ``EnergyResult`` object, you can use the ``compare_energyresult`` fixture.
    When a different type of file should be parsed, write a new fixture for that type of file.

3. Create a test
    Tests that test a single module can be grouped into a single file. For example, ``test_kubo.py`` tests functionalities in the ``__kubo.py`` module.

    To make the tests run faster, expensive calculations such as ``wannierberri.integrate`` should be done inside a fixture with ``scope="module"``. Then, the calculation results are reused for the functions inside the file. Test functions can request the results by have the fixture name in the argument.

    Note that by default, ``pytest.approx`` uses a relative tolerance of ``1e-6``. So, ``assert`` may be problematic if the values are close to zero.
    You can use the ``abs`` argument of the ``pytest.approx`` function to set an absolute tolerance. If an absolute tolerance is set, the default relative tolerance is not used.
