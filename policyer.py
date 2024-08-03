import argparse


def parser_add_ACT(parser: argparse.ArgumentParser):
    # for ACT
    parser.add_argument(
        "-kw", "--kl_weight", action="store", type=int, help="KL Weight", required=False
    )
    parser.add_argument(
        "-cs",
        "--chunk_size",
        action="store",
        type=int,
        help="chunk_size",
        required=False,
    )
    parser.add_argument(
        "-hd",
        "--hidden_dim",
        action="store",
        type=int,
        help="hidden_dim",
        required=False,
    )
    parser.add_argument(
        "-df",
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument(
        "-ta",
        "--temporal_agg",
        action="store",
        type=bool,
        help="temporal_agg",
        required=False,
    )


def parser_add(policy_class: str, parser: argparse.ArgumentParser):
    if policy_class == "ACT":
        parser_add_ACT(parser)
    elif policy_class == "CNNMLP":
        pass  # no additional args
    else:
        raise NotImplementedError


def make_policy(config):
    policy_maker = config["policy_maker"]
    policy = policy_maker(config)
    if policy is None:  # use default policy by policy_class
        policy_class = config["policy_class"]
        if policy_class == "ACT":
            from policy import ACTPolicy

            policy = ACTPolicy(config)
        elif policy_class == "CNNMLP":
            from policy import CNNMLPPolicy

            policy = CNNMLPPolicy(config)
        else:
            raise NotImplementedError(f"{policy_class}")
    return policy