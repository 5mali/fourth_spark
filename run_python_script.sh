#!/bin/bash
parallel -k python ::: ./pyscripts/${1}.py ::: 0 1 2 3 4 5 6 7 8 9 >> ./outputs/${1}.out 
