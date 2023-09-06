rm -r dist/*  build/*
python3 -m build
python3 -m twine upload  -u stepan-tsirkin dist/*

