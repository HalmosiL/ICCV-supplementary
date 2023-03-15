#!/bin/bash

device=1 model=normal epsilon=0.01 alpha=0.01 python test_base_line_pgd.py
device=1 model=normal epsilon=0.01 alpha=1 python test_base_cosine.py
device=1 model=normal epsilon=0.01 alpha=1 python test_base_cosine_combination.py

device=1 model=normal epsilon=0.02 alpha=0.01 python test_base_line_pgd.py
device=1 model=normal epsilon=0.02 alpha=1 python test_base_cosine.py
device=1 model=normal epsilon=0.02 alpha=1 python test_base_cosine_combination.py

device=1 model=normal epsilon=0.03 alpha=0.01 python test_base_line_pgd.py
device=1 model=normal epsilon=0.03 alpha=1 python test_base_cosine.py
device=1 model=normal epsilon=0.03 alpha=1 python test_base_cosine_combination.py
