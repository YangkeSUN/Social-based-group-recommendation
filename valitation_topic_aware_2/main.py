import argparse

import valide_util as ns_util

def load_config(config_path):
    with ns_util.smart_open(config_path) as config_file:
        return json.load(config_file)
# end def load_config

def main():
    parser = argparse.ArgumentParser(description='recommend films to groups of user')
    parser.add_argument('--config', nargs=1, metavar='FILE', type=str, required=False, default='../config.json'
                        , help='configuration file containing input file locations and default parameters (default=config.json)')
    parser.add_argument('--nb_item', nargs=1, metavar='K', type=float, required=False,
                        help='number of items_to_be_tested')
    parser.add_argument('--coef_sw', nargs=1, metavar='A', type=float, required=False,
                        help='Social welfare coeficient (between 0.0 and 1.0)')
    parser.add_argument('--coef_f', nargs=1, metavar='B', type=float, required=False,
                        help='faireness coeficient (between 0.0 and 1.0)')
    parser.add_argument('--nb_user_group', nargs=1, metavar='M', type=float, required=False,
                        help="number of users in a group")
    parser.add_argument('--x_core', nargs=1, metavar='X', type=float, required=False,
                        help="minim1um number of links from one user to other users in a group")
    args = parser.parse_args()

    config = load_config(args.config)
    if (args.coef_sw != None):
        config["coef_sw"] = args.coef_sw[0]
    if (args.coef_f != None):
        config["coef_f"] = args.coef_f[0]
    if (args.nb_user_group != None):
        config["nb_user_group"] = args.nb_user_group[0]
    if (args.x_core != None):
        config["x_core"] = args.x_core[0]
    if (args.nb_item != None):
        config["nb_item"] = args.nb_item[0]
    #validation(config, config["nb_item"], config["nb_user_group"], config["x_core"], config["coef_sw"], config["coef_f"])


# end def main

if __name__ == '__main__':
    main()