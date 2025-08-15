from collections import defaultdict


def tree():
    return defaultdict(tree)


unw_eff1 = tree()
eval_time = tree()

eval_time["gg_4g"]["LC"] = 2.9 / 100e3
eval_time["gg_5g"]["LC"] = 20.3 / 100e3
eval_time["gg_6g"]["LC"] = 1901 / 100e3
eval_time["gg_7g"]["LC"] = 26184 / 100e3
eval_time["gg_ddbar2g"]["LC"] = 1.0 / 100e3
eval_time["gg_ddbar3g"]["LC"] = 5.9 / 100e3
eval_time["gg_ddbar4g"]["LC"] = 45 / 100e3
eval_time["gg_ddbar5g"]["LC"] = 4815 / 100e3

eval_time["gg_4g"]["FC"] = 4.2 / 100e3
eval_time["gg_5g"]["FC"] = 55 / 100e3
eval_time["gg_6g"]["FC"] = 6984 / 100e3
eval_time["gg_7g"]["FC"] = 183638 / 100e3
eval_time["gg_ddbar2g"]["FC"] = 1.1 / 100e3
eval_time["gg_ddbar3g"]["FC"] = 7.2 / 100e3
eval_time["gg_ddbar4g"]["FC"] = 88 / 100e3
eval_time["gg_ddbar5g"]["FC"] = 12130 / 100e3

eval_time = {
    "gg_4g": {
        "MLP": {"t_eval": 2.3879313333333336e-05},
        "Transformer": {"t_eval": 0.00010307204333333333},
        "L-GATr": {"t_eval": 0.0024563862933333334},
        "GNN": {"t_eval": 0.00023689792000000003},
        "LC": 2.9e-05,
        "FC": 4.2e-05,
    },
    "gg_5g": {
        "MLP": {"t_eval": 2.3823003333333333e-05},
        "Transformer": {"t_eval": 0.00011804186333333333},
        "L-GATr": {"t_eval": 0.0027904853833333335},
        "GNN": {"t_eval": 0.0004177259633333333},
        "LC": 0.000203,
        "FC": 0.00055,
    },
    "gg_6g": {
        "MLP": {"t_eval": 2.331071e-05},
        "Transformer": {"t_eval": 0.00012635237333333334},
        "L-GATr": {"t_eval": 0.00315832192},
        "GNN": {"t_eval": 0.0005112517366666667},
        "LC": 0.01901,
        "FC": 0.06984,
    },
    "gg_7g": {
        "MLP": {"t_eval": 2.4133153333333332e-05},
        "Transformer": {"t_eval": 0.00014483639666666666},
        "L-GATr": {"t_eval": 0.003491112406666667},
        "GNN": {"t_eval": 0.0006166459299999999},
        "LC": 0.26184,
        "FC": 1.83638,
    },
    "gg_ddbar2g": {
        "MLP": {"t_eval": 2.4215260000000002e-05},
        "Transformer": {"t_eval": 0.00010357848},
        "L-GATr": {"t_eval": 0.00251463425},
        "GNN": {"t_eval": 0.00025660013},
        "LC": 1e-05,
        "FC": 1.1e-05,
    },
    "gg_ddbar3g": {
        "MLP": {"t_eval": 2.38959e-05},
        "Transformer": {"t_eval": 0.00012085204666666667},
        "L-GATr": {"t_eval": 0.0028615595900000003},
        "GNN": {"t_eval": 0.00041315342000000004},
        "LC": 5.9e-05,
        "FC": 7.2e-05,
    },
    "gg_ddbar4g": {
        "MLP": {"t_eval": 2.3662923333333333e-05},
        "Transformer": {"t_eval": 0.00012886209666666666},
        "L-GATr": {"t_eval": 0.003151712493333333},
        "GNN": {"t_eval": 0.0004948330166666667},
        "LC": 0.00045,
        "FC": 0.00088,
    },
    "gg_ddbar5g": {
        "MLP": {"t_eval": 2.7882050000000002e-05},
        "Transformer": {"t_eval": 0.00014412778666666665},
        "L-GATr": {"t_eval": 0.003462262916666667},
        "GNN": {"t_eval": 0.0006339516133333333},
        "LC": 0.04815,
        "FC": 0.1213,
    },
}


unw_eff1 = {
    "gg_4g": {
        "MLP": 0.88465026,
        "Transformer": 0.8743301733,
        "L-GATr": 0.8745681559,
        "GNN": 0.8724483436,
        "LC": 0.0189,
    },
    "gg_5g": {
        "MLP": 0.8006207521,
        "Transformer": 0.7862384267,
        "L-GATr": 0.7886181018,
        "GNN": 0.7705409337,
        "LC": 0.00419,
    },
    "gg_6g": {
        "MLP": 0.6603309633,
        "Transformer": 0.6799749569,
        "L-GATr": 0.6741837726,
        "GNN": 0.6788424623,
        "LC": 0.00208,
    },
    "gg_7g": {
        "MLP": 0.5849038822,
        "Transformer": 0.5582312897,
        "L-GATr": 0.5672788081,
        "GNN": 0.5730996927,
        "LC": 0.00119,
    },
    "gg_ddbar2g": {
        "MLP": 0.9031034842,
        "Transformer": 0.9022855917,
        "L-GATr": 0.9008463209,
        "GNN": 0.8934880275,
        "LC": 0.00943,
    },
    "gg_ddbar3g": {
        "MLP": 0.8080714813,
        "Transformer": 0.8071880033,
        "L-GATr": 0.814173303,
        "GNN": 0.8123292579,
        "LC": 0.00413,
    },
    "gg_ddbar4g": {
        "MLP": 0.7478706121,
        "Transformer": 0.7385788632,
        "L-GATr": 0.7351430712,
        "GNN": 0.7300044262,
        "LC": 0.00189,
    },
    "gg_ddbar5g": {
        "MLP": 0.6860827572,
        "Transformer": 0.6268221463,
        "L-GATr": 0.6543178169,
        "GNN": 0.6324968122,
        "LC": 0.00106,
    },
}
