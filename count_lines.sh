echo "Number of non-empty lines in the code:"
cat wannierberri/*.py wannierberri/*/*.py | sed '/^\s*$/d' | wc -l
echo "Number of lines excluding comments:"
cat wannierberri/*.py wannierberri/*/*.py| sed '/^\s*#/d;/^\s*$/d' | wc -l
