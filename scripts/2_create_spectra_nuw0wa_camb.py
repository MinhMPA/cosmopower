#!/usr/bin/env python3
from generate_nuw0wa_training_data import main
import sys

if __name__ == '__main__':
    main(['run-shard', *sys.argv[1:]])
