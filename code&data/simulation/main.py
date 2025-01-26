# This file is the main file for the simulation. It is responsible for parsing the arguments and the configuration file, and then calling the appropriate function to run the simulation.
import sys
from utils import parse_args_and_config

def main():
    args, config = parse_args_and_config()
    if args.type == 'ln':
        from utils import ln
        ln(args, config)
    elif args.type == 'nln_m':
        from utils import nln_m
        nln_m(args, config)
    else:
        if args.type != 'nln_r':
            print('Invalid type, defaulting to nln_r')
        from utils import nln_r
        if args.gen_comb == 'Y':
            from utils import gen_comb
            gen_comb(args.nln_r_comb_dir)
        nln_r(args, config)
    return 0


if __name__ == '__main__':
    sys.exit(main())