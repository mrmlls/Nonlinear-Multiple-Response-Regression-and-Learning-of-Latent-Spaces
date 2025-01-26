#The code of the score function is based on the code of the following repository: https://github.com/miskcoo/kscore.

import sys
from utils import parse_args_and_config, M1

def main():
    args, config = parse_args_and_config()
    M1(args, config)
    return 0


if __name__ == '__main__':
    sys.exit(main())



