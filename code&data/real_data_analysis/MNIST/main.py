#The code of the score function is based on the code of the following repository: https://github.com/ermongroup/ncsn/tree/master.


import sys
from utils import parse_args_and_config, mnist, pre_process



def main():
    args, config, model_config = parse_args_and_config()
    if args.pre_process == 'Y':
        pre_process(args.data_dir)

    mnist(args, config, model_config)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())