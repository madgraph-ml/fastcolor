from collections import defaultdict


def tree():
    return defaultdict(tree)


unw_eff1 = tree()
eval_time = tree()

# LC timings
eval_time["gg_4g"]["LC"] = 2.9 / 100e3
eval_time["gg_5g"]["LC"] = 20.3 / 100e3
eval_time["gg_6g"]["LC"] = 1901 / 100e3
eval_time["gg_7g"]["LC"] = 26184 / 100e3


eval_time["gg_ddbar2g"]["LC"] = 1.0 / 100e3
eval_time["gg_ddbar3g"]["LC"] = 5.9 / 100e3
eval_time["gg_ddbar4g"]["LC"] = 45 / 100e3
eval_time["gg_ddbar5g"]["LC"] = 4815 / 100e3


eval_time["dbard_4g"]["LC"] = 1.2 / 100e3
eval_time["dbard_5g"]["LC"] = 5.8 / 100e3
eval_time["dbard_6g"]["LC"] = 45 / 100e3
eval_time["dbard_7g"]["LC"] = 4944 / 100e3


eval_time["gg_ddbaruubar0g_co1"]["LC"] = 0.1 / 100e3
eval_time["gg_ddbaruubar1g_co1"]["LC"] = 1.9 / 100e3
eval_time["gg_ddbaruubar2g_co1"]["LC"] = 13 / 100e3
eval_time["gg_ddbaruubar3g_co1"]["LC"] = 116 / 100e3

eval_time["gg_ddbaruubar0g_co2"]["LC"] = 0.1 / 100e3
eval_time["gg_ddbaruubar1g_co2"]["LC"] = 1.9 / 100e3
eval_time["gg_ddbaruubar2g_co2"]["LC"] = 13 / 100e3
eval_time["gg_ddbaruubar3g_co2"]["LC"] = 225 / 100e3


eval_time["ddbar_uubar2g_co1"]["LC"] = 0.1 / 100e3
eval_time["ddbar_uubar3g_co1"]["LC"] = 2.3 / 100e3
eval_time["ddbar_uubar4g_co1"]["LC"] = 13 / 100e3
eval_time["ddbar_uubar5g_co1"]["LC"] = 179 / 100e3

eval_time["ddbar_uubar2g_co2"]["LC"] = 0.1 / 100e3
eval_time["ddbar_uubar3g_co2"]["LC"] = 2.2 / 100e3
eval_time["ddbar_uubar4g_co2"]["LC"] = 13 / 100e3
eval_time["ddbar_uubar5g_co2"]["LC"] = 178 / 100e3


# FC timings
eval_time["gg_4g"]["FC"] = 4.2 / 100e3
eval_time["gg_5g"]["FC"] = 55 / 100e3
eval_time["gg_6g"]["FC"] = 6984 / 100e3
eval_time["gg_7g"]["FC"] = 183638 / 100e3


eval_time["gg_ddbar2g"]["FC"] = 1.1 / 100e3
eval_time["gg_ddbar3g"]["FC"] = 7.2 / 100e3
eval_time["gg_ddbar4g"]["FC"] = 88 / 100e3
eval_time["gg_ddbar5g"]["FC"] = 12130 / 100e3


eval_time["dbard_4g"]["FC"] = 1.3 / 100e3
eval_time["dbard_5g"]["FC"] = 7.0 / 100e3
eval_time["dbard_6g"]["FC"] = 88 / 100e3
eval_time["dbard_7g"]["FC"] = 12832 / 100e3


eval_time["gg_ddbaruubar0g_co1"]["FC"] = 0.1 / 100e3
eval_time["gg_ddbaruubar1g_co1"]["FC"] = 2.1 / 100e3
eval_time["gg_ddbaruubar2g_co1"]["FC"] = 17 / 100e3
eval_time["gg_ddbaruubar3g_co1"]["FC"] = 262 / 100e3

eval_time["gg_ddbaruubar0g_co2"]["FC"] = 0.1 / 100e3
eval_time["gg_ddbaruubar1g_co2"]["FC"] = 2.0 / 100e3
eval_time["gg_ddbaruubar2g_co2"]["FC"] = 17 / 100e3
eval_time["gg_ddbaruubar3g_co2"]["FC"] = 545 / 100e3

eval_time["ddbar_uubar2g_co1"]["FC"] = 0.1 / 100e3
eval_time["ddbar_uubar3g_co1"]["FC"] = 2.5 / 100e3
eval_time["ddbar_uubar4g_co1"]["FC"] = 17 / 100e3
eval_time["ddbar_uubar5g_co1"]["FC"] = 342 / 100e3

eval_time["ddbar_uubar2g_co2"]["FC"] = 0.1 / 100e3
eval_time["ddbar_uubar3g_co2"]["FC"] = 2.5 / 100e3
eval_time["ddbar_uubar4g_co2"]["FC"] = 17 / 100e3
eval_time["ddbar_uubar5g_co2"]["FC"] = 340 / 100e3

# ── Network timings (add after your LC/FC block) ───────────────────────────────

# gg_ng
eval_time["gg_4g"]["GNN"]["t_eval"] = 0.00027374593
eval_time["gg_4g"]["GNN"]["ut_eval"] = 1.304353782708682e-05
eval_time["gg_4g"]["L-GATr"]["t_eval"] = 0.0020754428799999998
eval_time["gg_4g"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_4g"]["MLP"]["t_eval"] = 3.809707e-05
eval_time["gg_4g"]["MLP"]["ut_eval"] = 1.6258811364727936e-05
eval_time["gg_4g"]["Transformer"]["t_eval"] = 0.00014068771
eval_time["gg_4g"]["Transformer"]["ut_eval"] = 9.230064914700537e-06

eval_time["gg_5g"]["GNN"]["t_eval"] = 0.0004657274
eval_time["gg_5g"]["GNN"]["ut_eval"] = 2.1959124416398493e-05
eval_time["gg_5g"]["L-GATr"]["t_eval"] = 0.00243224499
eval_time["gg_5g"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_5g"]["MLP"]["t_eval"] = 2.541051e-05
eval_time["gg_5g"]["MLP"]["ut_eval"] = 1.2306168697402293e-06
eval_time["gg_5g"]["Transformer"]["t_eval"] = 0.00015371613999999999
eval_time["gg_5g"]["Transformer"]["ut_eval"] = 9.900609692710462e-06

eval_time["gg_6g"]["GNN"]["t_eval"] = 0.00055583724
eval_time["gg_6g"]["GNN"]["ut_eval"] = 2.9809942286706113e-05
eval_time["gg_6g"]["L-GATr"]["t_eval"] = 0.0027323227100000003
eval_time["gg_6g"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_6g"]["MLP"]["t_eval"] = 2.5255709999999997e-05
eval_time["gg_6g"]["MLP"]["ut_eval"] = 1.3829501109883684e-06
eval_time["gg_6g"]["Transformer"]["t_eval"] = 0.00017043659999999998
eval_time["gg_6g"]["Transformer"]["ut_eval"] = 1.1492902374822313e-05

eval_time["gg_7g"]["GNN"]["t_eval"] = 0.00065230378
eval_time["gg_7g"]["GNN"]["ut_eval"] = 1.4188286230854858e-05
eval_time["gg_7g"]["L-GATr"]["t_eval"] = 0.0029305662
eval_time["gg_7g"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_7g"]["MLP"]["t_eval"] = 2.376627e-05
eval_time["gg_7g"]["MLP"]["ut_eval"] = 2.5033452617763535e-06
eval_time["gg_7g"]["Transformer"]["t_eval"] = 0.00017842856999999998
eval_time["gg_7g"]["Transformer"]["ut_eval"] = 2.2850282694247026e-05

# gg_ddbarng
eval_time["gg_ddbar2g"]["GNN"]["t_eval"] = 0.0002786068
eval_time["gg_ddbar2g"]["GNN"]["ut_eval"] = 1.8940496686783975e-05
eval_time["gg_ddbar2g"]["L-GATr"]["t_eval"] = 0.00209278594
eval_time["gg_ddbar2g"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_ddbar2g"]["MLP"]["t_eval"] = 2.508813e-05
eval_time["gg_ddbar2g"]["MLP"]["ut_eval"] = 1.955096423326442e-06
eval_time["gg_ddbar2g"]["Transformer"]["t_eval"] = 0.00013607651999999999
eval_time["gg_ddbar2g"]["Transformer"]["ut_eval"] = 8.116615308722387e-06

eval_time["gg_ddbar3g"]["GNN"]["t_eval"] = 0.00043681163
eval_time["gg_ddbar3g"]["GNN"]["ut_eval"] = 9.539257688135788e-06
eval_time["gg_ddbar3g"]["L-GATr"]["t_eval"] = 0.00235388749
eval_time["gg_ddbar3g"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_ddbar3g"]["MLP"]["t_eval"] = 2.602201e-05
eval_time["gg_ddbar3g"]["MLP"]["ut_eval"] = 1.8420121244880649e-06
eval_time["gg_ddbar3g"]["Transformer"]["t_eval"] = 0.00017124281
eval_time["gg_ddbar3g"]["Transformer"]["ut_eval"] = 3.159064600855444e-05

eval_time["gg_ddbar4g"]["GNN"]["t_eval"] = 0.00052411588
eval_time["gg_ddbar4g"]["GNN"]["ut_eval"] = 2.7979233060486214e-05
eval_time["gg_ddbar4g"]["L-GATr"]["t_eval"] = 0.00267271929
eval_time["gg_ddbar4g"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_ddbar4g"]["MLP"]["t_eval"] = 2.6676510000000002e-05
eval_time["gg_ddbar4g"]["MLP"]["ut_eval"] = 2.7208114639366148e-06
eval_time["gg_ddbar4g"]["Transformer"]["t_eval"] = 0.000167509
eval_time["gg_ddbar4g"]["Transformer"]["ut_eval"] = 1.0727754967711286e-05

eval_time["gg_ddbar5g"]["GNN"]["t_eval"] = 0.0006654559
eval_time["gg_ddbar5g"]["GNN"]["ut_eval"] = 2.366833694678674e-05
eval_time["gg_ddbar5g"]["L-GATr"]["t_eval"] = 0.00288946958
eval_time["gg_ddbar5g"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_ddbar5g"]["MLP"]["t_eval"] = 2.606868e-05
eval_time["gg_ddbar5g"]["MLP"]["ut_eval"] = 1.282126475859091e-06
eval_time["gg_ddbar5g"]["Transformer"]["t_eval"] = 0.00018999122
eval_time["gg_ddbar5g"]["Transformer"]["ut_eval"] = 1.4008876766795574e-05

# dbard_ng
eval_time["dbard_4g"]["GNN"]["t_eval"] = 0.00027851844
eval_time["dbard_4g"]["GNN"]["ut_eval"] = 1.2370089133892706e-05
eval_time["dbard_4g"]["L-GATr"]["t_eval"] = 0.00208454191
eval_time["dbard_4g"]["L-GATr"]["ut_eval"] = 0.0
eval_time["dbard_4g"]["MLP"]["t_eval"] = 2.836278e-05
eval_time["dbard_4g"]["MLP"]["ut_eval"] = 4.121620651604239e-06
eval_time["dbard_4g"]["Transformer"]["t_eval"] = 0.00013730325
eval_time["dbard_4g"]["Transformer"]["ut_eval"] = 9.186721609634254e-06

eval_time["dbard_5g"]["GNN"]["t_eval"] = 0.00044653209
eval_time["dbard_5g"]["GNN"]["ut_eval"] = 1.8163602974572665e-05
eval_time["dbard_5g"]["L-GATr"]["t_eval"] = 0.00235333942
eval_time["dbard_5g"]["L-GATr"]["ut_eval"] = 0.0
eval_time["dbard_5g"]["MLP"]["t_eval"] = 2.555362e-05
eval_time["dbard_5g"]["MLP"]["ut_eval"] = 1.8864278740168589e-06
eval_time["dbard_5g"]["Transformer"]["t_eval"] = 0.00015348836000000002
eval_time["dbard_5g"]["Transformer"]["ut_eval"] = 1.7299590734260165e-05

eval_time["dbard_6g"]["GNN"]["t_eval"] = 0.00054028385
eval_time["dbard_6g"]["GNN"]["ut_eval"] = 2.140696862108568e-05
eval_time["dbard_6g"]["L-GATr"]["t_eval"] = 0.00260886944
eval_time["dbard_6g"]["L-GATr"]["ut_eval"] = 0.0
eval_time["dbard_6g"]["MLP"]["t_eval"] = 2.7047270000000002e-05
eval_time["dbard_6g"]["MLP"]["ut_eval"] = 4.1352117563826045e-06
eval_time["dbard_6g"]["Transformer"]["t_eval"] = 0.00015907372
eval_time["dbard_6g"]["Transformer"]["ut_eval"] = 1.7583661130753758e-05

eval_time["dbard_7g"]["GNN"]["t_eval"] = 0.0006845457899999999
eval_time["dbard_7g"]["GNN"]["ut_eval"] = 5.4899846127342295e-05
eval_time["dbard_7g"]["L-GATr"]["t_eval"] = 0.00292440336
eval_time["dbard_7g"]["L-GATr"]["ut_eval"] = 0.0
eval_time["dbard_7g"]["MLP"]["t_eval"] = 2.597325e-05
eval_time["dbard_7g"]["MLP"]["ut_eval"] = 1.309762642668876e-06
eval_time["dbard_7g"]["Transformer"]["t_eval"] = 0.00018972249999999998
eval_time["dbard_7g"]["Transformer"]["ut_eval"] = 1.514206799441357e-05

# gg_ddbaruubar_co1
eval_time["gg_ddbaruubar0g_co1"]["GNN"]["t_eval"] = 0.00026341982999999997
eval_time["gg_ddbaruubar0g_co1"]["GNN"]["ut_eval"] = 1.8895816689018426e-05
eval_time["gg_ddbaruubar0g_co1"]["L-GATr"]["t_eval"] = 0.00223855663
eval_time["gg_ddbaruubar0g_co1"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_ddbaruubar0g_co1"]["MLP"]["t_eval"] = 2.999614e-05
eval_time["gg_ddbaruubar0g_co1"]["MLP"]["ut_eval"] = 3.2539294005375015e-06
eval_time["gg_ddbaruubar0g_co1"]["Transformer"]["t_eval"] = 0.00013420769999999998
eval_time["gg_ddbaruubar0g_co1"]["Transformer"]["ut_eval"] = 6.239490213136382e-06

eval_time["gg_ddbaruubar1g_co1"]["GNN"]["t_eval"] = 0.00043672351
eval_time["gg_ddbaruubar1g_co1"]["GNN"]["ut_eval"] = 1.2623852449666823e-05
eval_time["gg_ddbaruubar1g_co1"]["L-GATr"]["t_eval"] = 0.0023414837499999997
eval_time["gg_ddbaruubar1g_co1"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_ddbaruubar1g_co1"]["MLP"]["t_eval"] = 2.613017e-05
eval_time["gg_ddbaruubar1g_co1"]["MLP"]["ut_eval"] = 2.6127031332815104e-06
eval_time["gg_ddbaruubar1g_co1"]["Transformer"]["t_eval"] = 0.00015279912
eval_time["gg_ddbaruubar1g_co1"]["Transformer"]["ut_eval"] = 1.883555522147848e-05

eval_time["gg_ddbaruubar2g_co1"]["GNN"]["t_eval"] = 0.00054006611
eval_time["gg_ddbaruubar2g_co1"]["GNN"]["ut_eval"] = 2.7366382455098544e-05
eval_time["gg_ddbaruubar2g_co1"]["L-GATr"]["t_eval"] = 0.0025954368799999995
eval_time["gg_ddbaruubar2g_co1"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_ddbaruubar2g_co1"]["MLP"]["t_eval"] = 2.001713e-05
eval_time["gg_ddbaruubar2g_co1"]["MLP"]["ut_eval"] = 2.8941150945144147e-07
eval_time["gg_ddbaruubar2g_co1"]["Transformer"]["t_eval"] = 0.00016437929
eval_time["gg_ddbaruubar2g_co1"]["Transformer"]["ut_eval"] = 1.9510447440015402e-05

eval_time["gg_ddbaruubar3g_co1"]["GNN"]["t_eval"] = 0.0006656855
eval_time["gg_ddbaruubar3g_co1"]["GNN"]["ut_eval"] = 4.570267083712037e-05
eval_time["gg_ddbaruubar3g_co1"]["L-GATr"]["t_eval"] = 0.00302478201
eval_time["gg_ddbaruubar3g_co1"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_ddbaruubar3g_co1"]["MLP"]["t_eval"] = 2.5190199999999998e-05
eval_time["gg_ddbaruubar3g_co1"]["MLP"]["ut_eval"] = 1.42993215331886e-06
eval_time["gg_ddbaruubar3g_co1"]["Transformer"]["t_eval"] = 0.00018191706
eval_time["gg_ddbaruubar3g_co1"]["Transformer"]["ut_eval"] = 2.495191323586416e-05

# gg_ddbaruubar_co2
eval_time["gg_ddbaruubar0g_co2"]["GNN"]["t_eval"] = 0.00028784512
eval_time["gg_ddbaruubar0g_co2"]["GNN"]["ut_eval"] = 1.3728759835848656e-05
eval_time["gg_ddbaruubar0g_co2"]["L-GATr"]["t_eval"] = 0.0020581584
eval_time["gg_ddbaruubar0g_co2"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_ddbaruubar0g_co2"]["MLP"]["t_eval"] = 2.7661699999999996e-05
eval_time["gg_ddbaruubar0g_co2"]["MLP"]["ut_eval"] = 1.919337262056262e-06
eval_time["gg_ddbaruubar0g_co2"]["Transformer"]["t_eval"] = 0.00013513489
eval_time["gg_ddbaruubar0g_co2"]["Transformer"]["ut_eval"] = 5.696444613493113e-06

eval_time["gg_ddbaruubar1g_co2"]["GNN"]["t_eval"] = 0.00044462209
eval_time["gg_ddbaruubar1g_co2"]["GNN"]["ut_eval"] = 1.108479402594728e-05
eval_time["gg_ddbaruubar1g_co2"]["L-GATr"]["t_eval"] = 0.00243067619
eval_time["gg_ddbaruubar1g_co2"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_ddbaruubar1g_co2"]["MLP"]["t_eval"] = 2.290437e-05
eval_time["gg_ddbaruubar1g_co2"]["MLP"]["ut_eval"] = 2.9554221891774903e-06
eval_time["gg_ddbaruubar1g_co2"]["Transformer"]["t_eval"] = 0.00015602927
eval_time["gg_ddbaruubar1g_co2"]["Transformer"]["ut_eval"] = 6.6904828904455505e-06

eval_time["gg_ddbaruubar2g_co2"]["GNN"]["t_eval"] = 0.0005574221
eval_time["gg_ddbaruubar2g_co2"]["GNN"]["ut_eval"] = 2.2325253558665204e-05
eval_time["gg_ddbaruubar2g_co2"]["L-GATr"]["t_eval"] = 0.00269145961
eval_time["gg_ddbaruubar2g_co2"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_ddbaruubar2g_co2"]["MLP"]["t_eval"] = 2.7574990000000002e-05
eval_time["gg_ddbaruubar2g_co2"]["MLP"]["ut_eval"] = 8.000867566385807e-07
eval_time["gg_ddbaruubar2g_co2"]["Transformer"]["t_eval"] = 0.00016438110000000002
eval_time["gg_ddbaruubar2g_co2"]["Transformer"]["ut_eval"] = 1.7357984519354404e-05

eval_time["gg_ddbaruubar3g_co2"]["GNN"]["t_eval"] = 0.0006451364300000001
eval_time["gg_ddbaruubar3g_co2"]["GNN"]["ut_eval"] = 2.5075388270154776e-05
eval_time["gg_ddbaruubar3g_co2"]["L-GATr"]["t_eval"] = 0.00303450419
eval_time["gg_ddbaruubar3g_co2"]["L-GATr"]["ut_eval"] = 0.0
eval_time["gg_ddbaruubar3g_co2"]["MLP"]["t_eval"] = 2.398737e-05
eval_time["gg_ddbaruubar3g_co2"]["MLP"]["ut_eval"] = 2.6408159146793248e-06
eval_time["gg_ddbaruubar3g_co2"]["Transformer"]["t_eval"] = 0.00018050283
eval_time["gg_ddbaruubar3g_co2"]["Transformer"]["ut_eval"] = 2.2163921877580588e-05

# ddbar_uubar_co1
eval_time["ddbar_uubar2g_co1"]["GNN"]["t_eval"] = 0.00027667812
eval_time["ddbar_uubar2g_co1"]["GNN"]["ut_eval"] = 1.041017129680976e-05
eval_time["ddbar_uubar2g_co1"]["L-GATr"]["t_eval"] = 0.00208589477
eval_time["ddbar_uubar2g_co1"]["L-GATr"]["ut_eval"] = 0.0
eval_time["ddbar_uubar2g_co1"]["MLP"]["t_eval"] = 3.130728e-05
eval_time["ddbar_uubar2g_co1"]["MLP"]["ut_eval"] = 7.708654676820086e-06
eval_time["ddbar_uubar2g_co1"]["Transformer"]["t_eval"] = 0.00014084868999999998
eval_time["ddbar_uubar2g_co1"]["Transformer"]["ut_eval"] = 9.181726181331128e-06

eval_time["ddbar_uubar3g_co1"]["GNN"]["t_eval"] = 0.00044092512999999997
eval_time["ddbar_uubar3g_co1"]["GNN"]["ut_eval"] = 2.0748493955385867e-05
eval_time["ddbar_uubar3g_co1"]["L-GATr"]["t_eval"] = 0.00235013245
eval_time["ddbar_uubar3g_co1"]["L-GATr"]["ut_eval"] = 0.0
eval_time["ddbar_uubar3g_co1"]["MLP"]["t_eval"] = 2.506331e-05
eval_time["ddbar_uubar3g_co1"]["MLP"]["ut_eval"] = 1.20719029807218e-06
eval_time["ddbar_uubar3g_co1"]["Transformer"]["t_eval"] = 0.00016103549
eval_time["ddbar_uubar3g_co1"]["Transformer"]["ut_eval"] = 8.900321853348978e-06

eval_time["ddbar_uubar4g_co1"]["GNN"]["t_eval"] = 0.0005384510499999999
eval_time["ddbar_uubar4g_co1"]["GNN"]["ut_eval"] = 2.3712625639834698e-05
eval_time["ddbar_uubar4g_co1"]["L-GATr"]["t_eval"] = 0.0026774616300000004
eval_time["ddbar_uubar4g_co1"]["L-GATr"]["ut_eval"] = 0.0
eval_time["ddbar_uubar4g_co1"]["MLP"]["t_eval"] = 2.4929389999999998e-05
eval_time["ddbar_uubar4g_co1"]["MLP"]["ut_eval"] = 1.1194647466221383e-06
eval_time["ddbar_uubar4g_co1"]["Transformer"]["t_eval"] = 0.00017119021
eval_time["ddbar_uubar4g_co1"]["Transformer"]["ut_eval"] = 8.625956917636396e-06

eval_time["ddbar_uubar5g_co1"]["GNN"]["t_eval"] = 0.00065706869
eval_time["ddbar_uubar5g_co1"]["GNN"]["ut_eval"] = 2.04450499704616e-05
eval_time["ddbar_uubar5g_co1"]["L-GATr"]["t_eval"] = 0.00290475122
eval_time["ddbar_uubar5g_co1"]["L-GATr"]["ut_eval"] = 0.0
eval_time["ddbar_uubar5g_co1"]["MLP"]["t_eval"] = 2.530469e-05
eval_time["ddbar_uubar5g_co1"]["MLP"]["ut_eval"] = 9.56808019180287e-07
eval_time["ddbar_uubar5g_co1"]["Transformer"]["t_eval"] = 0.00018315525
eval_time["ddbar_uubar5g_co1"]["Transformer"]["ut_eval"] = 2.5095491365953442e-05

# ddbar_uubar_co2
eval_time["ddbar_uubar2g_co2"]["GNN"]["t_eval"] = 0.0002789822
eval_time["ddbar_uubar2g_co2"]["GNN"]["ut_eval"] = 1.630083134959341e-05
eval_time["ddbar_uubar2g_co2"]["L-GATr"]["t_eval"] = 0.00208081383
eval_time["ddbar_uubar2g_co2"]["L-GATr"]["ut_eval"] = 0.0
eval_time["ddbar_uubar2g_co2"]["MLP"]["t_eval"] = 3.02494e-05
eval_time["ddbar_uubar2g_co2"]["MLP"]["ut_eval"] = 2.336262170969743e-06
eval_time["ddbar_uubar2g_co2"]["Transformer"]["t_eval"] = 0.00014117911
eval_time["ddbar_uubar2g_co2"]["Transformer"]["ut_eval"] = 7.878425289858841e-06

eval_time["ddbar_uubar3g_co2"]["GNN"]["t_eval"] = 0.00044155016000000004
eval_time["ddbar_uubar3g_co2"]["GNN"]["ut_eval"] = 1.6405865924364246e-05
eval_time["ddbar_uubar3g_co2"]["L-GATr"]["t_eval"] = 0.00234738799
eval_time["ddbar_uubar3g_co2"]["L-GATr"]["ut_eval"] = 0.0
eval_time["ddbar_uubar3g_co2"]["MLP"]["t_eval"] = 2.614352e-05
eval_time["ddbar_uubar3g_co2"]["MLP"]["ut_eval"] = 2.4171594732651143e-06
eval_time["ddbar_uubar3g_co2"]["Transformer"]["t_eval"] = 0.00015569342000000002
eval_time["ddbar_uubar3g_co2"]["Transformer"]["ut_eval"] = 7.919281439691317e-06

eval_time["ddbar_uubar4g_co2"]["GNN"]["t_eval"] = 0.00055285101
eval_time["ddbar_uubar4g_co2"]["GNN"]["ut_eval"] = 2.6698399038493625e-05
eval_time["ddbar_uubar4g_co2"]["L-GATr"]["t_eval"] = 0.00273138153
eval_time["ddbar_uubar4g_co2"]["L-GATr"]["ut_eval"] = 0.0
eval_time["ddbar_uubar4g_co2"]["MLP"]["t_eval"] = 2.5806309999999998e-05
eval_time["ddbar_uubar4g_co2"]["MLP"]["ut_eval"] = 7.026209244156472e-07
eval_time["ddbar_uubar4g_co2"]["Transformer"]["t_eval"] = 0.00016597393
eval_time["ddbar_uubar4g_co2"]["Transformer"]["ut_eval"] = 1.6083704684508725e-05

eval_time["ddbar_uubar5g_co2"]["GNN"]["t_eval"] = 0.00063327443
eval_time["ddbar_uubar5g_co2"]["GNN"]["ut_eval"] = 2.391690508936652e-05
eval_time["ddbar_uubar5g_co2"]["L-GATr"]["t_eval"] = 0.00291131845
eval_time["ddbar_uubar5g_co2"]["L-GATr"]["ut_eval"] = 0.0
eval_time["ddbar_uubar5g_co2"]["MLP"]["t_eval"] = 2.6871070000000002e-05
eval_time["ddbar_uubar5g_co2"]["MLP"]["ut_eval"] = 1.3096224171351333e-06
eval_time["ddbar_uubar5g_co2"]["Transformer"]["t_eval"] = 0.00019920267
eval_time["ddbar_uubar5g_co2"]["Transformer"]["ut_eval"] = 2.7889639786055027e-05

# # defaultdict(<function __main__.tree()>,
#             {'gg_4g': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00027374593,
#                                        'ut_eval': 1.304353782708682e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0020754428799999998,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 3.809707e-05,
#                                        'ut_eval': 1.6258811364727936e-05}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00014068771,
#                                        'ut_eval': 9.230064914700537e-06})}),
#              'gg_5g': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0004657274,
#                                        'ut_eval': 2.1959124416398493e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00243224499,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.541051e-05,
#                                        'ut_eval': 1.2306168697402293e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00015371613999999999,
#                                        'ut_eval': 9.900609692710462e-06})}),
#              'gg_6g': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00055583724,
#                                        'ut_eval': 2.9809942286706113e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0027323227100000003,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.5255709999999997e-05,
#                                        'ut_eval': 1.3829501109883684e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00017043659999999998,
#                                        'ut_eval': 1.1492902374822313e-05})}),
#              'gg_7g': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00065230378,
#                                        'ut_eval': 1.4188286230854858e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0029305662,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.376627e-05,
#                                        'ut_eval': 2.5033452617763535e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00017842856999999998,
#                                        'ut_eval': 2.2850282694247026e-05})}),
#              'gg_ddbar2g': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0002786068,
#                                        'ut_eval': 1.8940496686783975e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00209278594,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.508813e-05,
#                                        'ut_eval': 1.955096423326442e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00013607651999999999,
#                                        'ut_eval': 8.116615308722387e-06})}),
#              'gg_ddbar3g': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00043681163,
#                                        'ut_eval': 9.539257688135788e-06}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00235388749,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.602201e-05,
#                                        'ut_eval': 1.8420121244880649e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00017124281,
#                                        'ut_eval': 3.159064600855444e-05})}),
#              'gg_ddbar4g': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00052411588,
#                                        'ut_eval': 2.7979233060486214e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00267271929,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.6676510000000002e-05,
#                                        'ut_eval': 2.7208114639366148e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.000167509,
#                                        'ut_eval': 1.0727754967711286e-05})}),
#              'gg_ddbar5g': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0006654559,
#                                        'ut_eval': 2.366833694678674e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00288946958,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.606868e-05,
#                                        'ut_eval': 1.282126475859091e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00018999122,
#                                        'ut_eval': 1.4008876766795574e-05})}),
#              'dbard_4g': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00027851844,
#                                        'ut_eval': 1.2370089133892706e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00208454191,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.836278e-05,
#                                        'ut_eval': 4.121620651604239e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00013730325,
#                                        'ut_eval': 9.186721609634254e-06})}),
#              'dbard_5g': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00044653209,
#                                        'ut_eval': 1.8163602974572665e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00235333942,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.555362e-05,
#                                        'ut_eval': 1.8864278740168589e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00015348836000000002,
#                                        'ut_eval': 1.7299590734260165e-05})}),
#              'dbard_6g': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00054028385,
#                                        'ut_eval': 2.140696862108568e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00260886944,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.7047270000000002e-05,
#                                        'ut_eval': 4.1352117563826045e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00015907372,
#                                        'ut_eval': 1.7583661130753758e-05})}),
#              'dbard_7g': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0006845457899999999,
#                                        'ut_eval': 5.4899846127342295e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00292440336,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.597325e-05,
#                                        'ut_eval': 1.309762642668876e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00018972249999999998,
#                                        'ut_eval': 1.514206799441357e-05})}),
#              'gg_ddbaruubar0g_co1': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00026341982999999997,
#                                        'ut_eval': 1.8895816689018426e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00223855663,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.999614e-05,
#                                        'ut_eval': 3.2539294005375015e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00013420769999999998,
#                                        'ut_eval': 6.239490213136382e-06})}),
#              'gg_ddbaruubar1g_co1': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00043672351,
#                                        'ut_eval': 1.2623852449666823e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0023414837499999997,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.613017e-05,
#                                        'ut_eval': 2.6127031332815104e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00015279912,
#                                        'ut_eval': 1.883555522147848e-05})}),
#              'gg_ddbaruubar2g_co1': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00054006611,
#                                        'ut_eval': 2.7366382455098544e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0025954368799999995,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.001713e-05,
#                                        'ut_eval': 2.8941150945144147e-07}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00016437929,
#                                        'ut_eval': 1.9510447440015402e-05})}),
#              'gg_ddbaruubar3g_co1': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0006656855,
#                                        'ut_eval': 4.570267083712037e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00302478201,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.5190199999999998e-05,
#                                        'ut_eval': 1.42993215331886e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00018191706,
#                                        'ut_eval': 2.495191323586416e-05})}),
#              'gg_ddbaruubar0g_co2': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00028784512,
#                                        'ut_eval': 1.3728759835848656e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0020581584,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.7661699999999996e-05,
#                                        'ut_eval': 1.919337262056262e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00013513489,
#                                        'ut_eval': 5.696444613493113e-06})}),
#              'gg_ddbaruubar1g_co2': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00044462209,
#                                        'ut_eval': 1.108479402594728e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00243067619,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.290437e-05,
#                                        'ut_eval': 2.9554221891774903e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00015602927,
#                                        'ut_eval': 6.6904828904455505e-06})}),
#              'gg_ddbaruubar2g_co2': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0005574221,
#                                        'ut_eval': 2.2325253558665204e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00269145961,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.7574990000000002e-05,
#                                        'ut_eval': 8.000867566385807e-07}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00016438110000000002,
#                                        'ut_eval': 1.7357984519354404e-05})}),
#              'gg_ddbaruubar3g_co2': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0006451364300000001,
#                                        'ut_eval': 2.5075388270154776e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00303450419,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.398737e-05,
#                                        'ut_eval': 2.6408159146793248e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00018050283,
#                                        'ut_eval': 2.2163921877580588e-05})}),
#              'ddbar_uubar2g_co1': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00027667812,
#                                        'ut_eval': 1.041017129680976e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00208589477,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 3.130728e-05,
#                                        'ut_eval': 7.708654676820086e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00014084868999999998,
#                                        'ut_eval': 9.181726181331128e-06})}),
#              'ddbar_uubar3g_co1': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00044092512999999997,
#                                        'ut_eval': 2.0748493955385867e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00235013245,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.506331e-05,
#                                        'ut_eval': 1.20719029807218e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00016103549,
#                                        'ut_eval': 8.900321853348978e-06})}),
#              'ddbar_uubar4g_co1': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0005384510499999999,
#                                        'ut_eval': 2.3712625639834698e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0026774616300000004,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.4929389999999998e-05,
#                                        'ut_eval': 1.1194647466221383e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00017119021,
#                                        'ut_eval': 8.625956917636396e-06})}),
#              'ddbar_uubar5g_co1': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00065706869,
#                                        'ut_eval': 2.04450499704616e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00290475122,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.530469e-05,
#                                        'ut_eval': 9.56808019180287e-07}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00018315525,
#                                        'ut_eval': 2.5095491365953442e-05})}),
#              'ddbar_uubar2g_co2': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.0002789822,
#                                        'ut_eval': 1.630083134959341e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00208081383,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 3.02494e-05,
#                                        'ut_eval': 2.336262170969743e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00014117911,
#                                        'ut_eval': 7.878425289858841e-06})}),
#              'ddbar_uubar3g_co2': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00044155016000000004,
#                                        'ut_eval': 1.6405865924364246e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00234738799,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.614352e-05,
#                                        'ut_eval': 2.4171594732651143e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00015569342000000002,
#                                        'ut_eval': 7.919281439691317e-06})}),
#              'ddbar_uubar4g_co2': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00055285101,
#                                        'ut_eval': 2.6698399038493625e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00273138153,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.5806309999999998e-05,
#                                        'ut_eval': 7.026209244156472e-07}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00016597393,
#                                        'ut_eval': 1.6083704684508725e-05})}),
#              'ddbar_uubar5g_co2': defaultdict(<function __main__.tree()>,
#                          {'GNN': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00063327443,
#                                        'ut_eval': 2.391690508936652e-05}),
#                           'LGATr': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00291131845,
#                                        'ut_eval': 0.0}),
#                           'MLP': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 2.6871070000000002e-05,
#                                        'ut_eval': 1.3096224171351333e-06}),
#                           'Transformer': defaultdict(<function __main__.tree()>,
#                                       {'t_eval': 0.00019920267,
#                                        'ut_eval': 2.7889639786055027e-05})})})



unw_eff1["gg_4g"]["LC"] = 0.0108867
unw_eff1["gg_5g"]["LC"] = 0.0041933
unw_eff1["gg_6g"]["LC"] = 0.0020800
unw_eff1["gg_7g"]["LC"] = 0.0011925


unw_eff1["gg_ddbar2g"]["LC"] = 0.0094300
unw_eff1["gg_ddbar3g"]["LC"] = 0.0041267
unw_eff1["gg_ddbar4g"]["LC"] = 0.0018856
unw_eff1["gg_ddbar5g"]["LC"] = 0.0010642


unw_eff1["dbard_4g"]["LC"] = 0.0076000
unw_eff1["dbard_5g"]["LC"] = 0.0028900
unw_eff1["dbard_6g"]["LC"] = 0.0015500
unw_eff1["dbard_7g"]["LC"] = 0.0008900

unw_eff1["gg_ddbaruubar0g_co1"]["LC"] = 0.0092500
unw_eff1["gg_ddbaruubar1g_co1"]["LC"] = 0.0041750
unw_eff1["gg_ddbaruubar2g_co1"]["LC"] = 0.0020827
unw_eff1["gg_ddbaruubar3g_co1"]["LC"] = 0.0011394

unw_eff1["gg_ddbaruubar0g_co2"]["LC"] = 0.0086650
unw_eff1["gg_ddbaruubar1g_co2"]["LC"] = 0.0035950
unw_eff1["gg_ddbaruubar2g_co2"]["LC"] = 0.0017182
unw_eff1["gg_ddbaruubar3g_co2"]["LC"] = 0.0008967


unw_eff1["ddbar_uubar2g_co1"]["LC"] = 0.0109767
unw_eff1["ddbar_uubar3g_co1"]["LC"] = 0.0045850
unw_eff1["ddbar_uubar4g_co1"]["LC"] = 0.0021540
unw_eff1["ddbar_uubar5g_co1"]["LC"] = 0.0011450

unw_eff1["ddbar_uubar2g_co2"]["LC"] = 0.0098450
unw_eff1["ddbar_uubar3g_co2"]["LC"] = 0.0035850
unw_eff1["ddbar_uubar4g_co2"]["LC"] = 0.0017467
unw_eff1["ddbar_uubar5g_co2"]["LC"] = 0.0009633