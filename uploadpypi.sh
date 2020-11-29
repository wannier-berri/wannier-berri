rm -r dist/*  build/*
python3 setup.py bdist_wheel
python3 -m twine upload  -u stepan-tsirkin dist/*
