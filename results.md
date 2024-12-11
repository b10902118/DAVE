# Total {'val': 82042, 'test': 78836}

test without big
#test MAE 8.372074422611531 RMSE 30.683231353759766

test semi ex_max=8 only
#test MAE 9.470036464909512 RMSE 32.04297637939453

## semi ex_max= min(..., 8) density>0.95 confidence>0.97

#test MAE 9.310379182449495 RMSE 32.21308898925781
Smallest 10 differences:  
Index: 1150, Difference: -164.17  
Index: 1069, Difference: -110.89  
Index: 524, Difference: -63.92  
Index: 250, Difference: -46.58  
Index: 1027, Difference: -42.65  
Index: 148, Difference: -39.6  
Index: 173, Difference: -39.01  
Index: 269, Difference: -36.65  
Index: 144, Difference: -32.39  
Index: 172, Difference: -31.72

Biggest 10 differences:  
Index: 210, Difference: 13.35  
Index: 630, Difference: 15.57  
Index: 121, Difference: 16.4  
Index: 1068, Difference: 17.73  
Index: 1102, Difference: 18.23  
Index: 73, Difference: 20.23  
Index: 267, Difference: 21.7  
Index: 917, Difference: 27.8  
Index: 1149, Difference: 40.77  
Index: 205, Difference: 50.91

## semi ex_max= min(n/5, 3) density>0.9 confidence>0.97

#test MAE 9.139985466645623 RMSE 31.802871704101562
Smallest 10 differences:  
Index: 1150, Difference: -139.57  
Index: 524, Difference: -63.92  
Index: 1069, Difference: -60.9  
Index: 1027, Difference: -47.05  
Index: 250, Difference: -46.58  
Index: 269, Difference: -31.54  
Index: 1033, Difference: -30.32  
Index: 875, Difference: -26.37  
Index: 148, Difference: -25.96  
Index: 144, Difference: -24.43

Biggest 10 differences:  
Index: 267, Difference: 12.62  
Index: 217, Difference: 13.8  
Index: 1102, Difference: 14.78  
Index: 630, Difference: 15.57  
Index: 121, Difference: 16.52  
Index: 205, Difference: 17.48  
Index: 73, Difference: 18.16  
Index: 1068, Difference: 22.08  
Index: 917, Difference: 26.0  
Index: 1149, Difference: 31.38

## semi ex_max= 1 density>0.9 confidence>0.97

#test MAE 8.796921033249157 RMSE 31.103540420532227
Smallest 10 differences:
Index: 524, Difference: -63.92
Index: 1150, Difference: -61.69
Index: 1027, Difference: -48.31
Index: 1069, Difference: -38.17
Index: 269, Difference: -26.73
Index: 172, Difference: -18.46
Index: 148, Difference: -18.24
Index: 1041, Difference: -17.75
Index: 278, Difference: -15.94
Index: 1033, Difference: -14.84

Biggest 10 differences:
Index: 217, Difference: 7.79
Index: 121, Difference: 7.95
Index: 78, Difference: 8.16
Index: 630, Difference: 8.27
Index: 1038, Difference: 8.55
Index: 73, Difference: 10.07
Index: 1068, Difference: 11.87
Index: 159, Difference: 11.96
Index: 1149, Difference: 22.43
Index: 917, Difference: 24.29

## semi ex_max= 1 density>0.9 confidence>0.95

#test MAE 8.83672499736953 RMSE 31.115009307861328
Smallest 10 differences:
Index: 524, Difference: -63.91
Index: 1150, Difference: -54.93
Index: 1027, Difference: -48.28
Index: 1069, Difference: -38.14
Index: 269, Difference: -26.72
Index: 172, Difference: -18.45
Index: 148, Difference: -18.25
Index: 1041, Difference: -17.75
Index: 144, Difference: -17.02
Index: 278, Difference: -15.94

Biggest 10 differences:
Index: 217, Difference: 7.79
Index: 121, Difference: 7.94
Index: 630, Difference: 8.26
Index: 1038, Difference: 8.58
Index: 73, Difference: 10.08
Index: 1068, Difference: 11.87
Index: 159, Difference: 11.94
Index: 1102, Difference: 18.85
Index: 1149, Difference: 22.42
Index: 917, Difference: 24.3

## semi ex_max=1 confidence>0.95 density>0.9 && min |1-density|

#test MAE 8.806463890204125 RMSE 31.04218864440918  
repredict: 1188  
Smallest 10 differences:  
Index: 524, Difference: -63.92  
Index: 1150, Difference: -61.69  
Index: 1027, Difference: -48.31  
Index: 1069, Difference: -38.17  
Index: 250, Difference: -28.62  
Index: 269, Difference: -26.73  
Index: 172, Difference: -18.46  
Index: 148, Difference: -18.24  
Index: 278, Difference: -15.94  
Index: 1033, Difference: -14.84

Biggest 10 differences:  
Index: 78, Difference: 8.16  
Index: 630, Difference: 8.27  
Index: 1038, Difference: 8.55  
Index: 73, Difference: 10.07  
Index: 835, Difference: 10.3  
Index: 1068, Difference: 11.87  
Index: 159, Difference: 11.96  
Index: 205, Difference: 16.75  
Index: 1149, Difference: 22.43  
Index: 917, Difference: 24.29

# val

## original without big

#val MAE 8.587250018324863 RMSE 27.24641227722168

## two exampler

#val MAE 8.998242340207193 RMSE 28.10320472717285

## semi ex_max=1 confidence>0.95 density>0.9 && min |1-density|

#val MAE 8.737543674257232 RMSE 28.810226440429688
repredict: 1279
Smallest 10 differences:
Index: 1140, Difference: -84.21
Index: 399, Difference: -51.86
Index: 315, Difference: -50.59
Index: 203, Difference: -30.37
Index: 603, Difference: -16.59
Index: 125, Difference: -15.53
Index: 281, Difference: -11.26
Index: 339, Difference: -10.76
Index: 1192, Difference: -9.62
Index: 754, Difference: -9.54

Biggest 10 differences:
Index: 924, Difference: 7.94
Index: 451, Difference: 9.5
Index: 432, Difference: 9.79
Index: 535, Difference: 10.47
Index: 385, Difference: 10.48
Index: 303, Difference: 11.12
Index: 26, Difference: 11.78
Index: 199, Difference: 13.45
Index: 1239, Difference: 21.61
Index: 1136, Difference: 23.15

## semi ex_max=1 confidence>0.95 density>0.9 && max confidence

#val MAE 8.752921288604378 RMSE 28.811981201171875
repredict: 1279
Smallest 10 differences:
Index: 1140, Difference: -84.21
Index: 399, Difference: -51.86
Index: 315, Difference: -50.59
Index: 203, Difference: -30.37
Index: 603, Difference: -16.59
Index: 125, Difference: -15.53
Index: 339, Difference: -15.4
Index: 281, Difference: -11.26
Index: 1192, Difference: -9.62
Index: 754, Difference: -9.54

Biggest 10 differences:
Index: 924, Difference: 7.94
Index: 451, Difference: 9.5
Index: 432, Difference: 9.79
Index: 535, Difference: 10.47
Index: 385, Difference: 10.48
Index: 303, Difference: 11.12
Index: 26, Difference: 11.78
Index: 199, Difference: 13.45
Index: 1239, Difference: 21.61
Index: 1136, Difference: 23.15

## semi ex_max=1 confidence>0.95 density>0.9 && max confidence, replace one exampler

#val MAE 8.588205965353792 RMSE 27.24730110168457

## density normalization

#val MAE 8.931573040192458 RMSE 28.194080352783203
#test MAE 8.952152540703782 RMSE 33.02273941040039
