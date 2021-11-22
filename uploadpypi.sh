rm -r dist/*  build/*
python3 setup.py bdist_wheel
python3 -m twine upload  -u stepan-tsirkin dist/*

# Add git tag
version="v$(python3 setup.py --version)"
git tag -a $version -m "release of $version"
git push origin $version
