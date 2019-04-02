#!/bin/bash
echo ${1}.py
parallel -k python ::: ./py_scripts/${1}.py ::: 0 1 2 3 4 5 6 7 8 9 >> ./script_output/${1}.out 
