from operator import index
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

# For inserting an image
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg


df = pd.read_csv("../data/AutoInsurSweden.txt", sep='[\t ]', header=None, names=['X', 'Y'])

predicted = " -0.29841  -0.276306  -0.232391  -0.165519 -0.0739481  0.0446861   0.193391   0.375825   0.596283   0.859644    1.17127    1.53686    1.96219    2.45284    3.01389    3.64956    4.36291    5.15557    6.02734     6.9756    7.99439    9.07328    10.1965    11.3433     12.489    13.6086    14.6794    15.6842    16.6121    17.4589    18.2258    18.9177    19.5418    20.1065    20.6203    21.0914    21.5272    21.9343    22.3186    22.6851    23.0378    23.3805    23.7161     24.047    24.3753    24.7025    25.0297    25.3577    25.6868    26.0168    26.3472    26.6768    27.0041    27.3271    27.6436    27.9514    28.2485    28.5339    28.8083    29.0763    29.3492    29.6513    30.0313     30.583    31.4764    32.9853     35.449    39.0558    43.4834     47.871    51.4121    53.8551    55.3871    56.3069    56.8554    57.1877    57.3952    57.5298    57.6208     57.685     57.732    57.7677    57.7956     57.818    57.8362    57.8514     57.864    57.8747    57.8838    57.8916    57.8983     57.904     57.909    57.9133     57.917    57.9202     57.923    57.9255    57.9276    57.9294    57.9311    57.9325    57.9337    57.9348    57.9358    57.9366    57.9374    57.9381    57.9387    57.9392    57.9397    57.9401    57.9406    57.9409    57.9413    57.9416    57.9419    57.9422    57.9426    57.9429    57.9432    57.9435    57.9439    57.9443    57.9447    57.9452    57.9457    57.9463     57.947    57.9478    57.9487    57.9498    57.9511    57.9526    57.9543    57.9564    57.9589    57.9618    57.9654    57.9696    57.9747    57.9808    57.9883    57.9973    58.0084    58.0221    58.0391    58.0603    58.0873    58.1217    58.1664    58.2253    58.3041    58.4118     58.562    58.7764    59.0905    59.5625    60.2907    61.4403    63.2848    66.2506    70.9069    77.7445    86.5835    96.0136    104.003    109.412    112.421     113.77    114.113    113.862    113.243    112.388    111.385    110.303    109.204    108.137    107.138     106.23    105.422    104.716    104.107    103.586    103.143    102.768    102.452    102.185    101.961    101.773    101.614    101.481    101.369    101.274    101.195    101.128    101.073    101.026    100.987    100.954    100.927    100.905    100.886    100.871     100.86     100.85    100.843    100.838    100.835    100.833    100.833    100.835    100.837    100.841    100.847    100.853    100.861     100.87    100.881    100.893    100.907    100.923     100.94     100.96    100.982    101.006    101.034    101.064    101.098    101.135    101.177    101.224    101.275    101.333    101.397    101.468    101.547    101.636    101.734    101.843    101.966    102.102    102.254    102.423    102.613    102.824     103.06    103.323    103.617    103.945     104.31    104.716    105.165    105.662    106.209    106.807    107.457    108.156      108.9    109.682     110.49    111.311    112.127    112.921    113.672    114.364    114.984    115.523    115.979    116.355    116.656    116.893    117.077    117.218    117.326    117.412    117.481    117.541    117.595    117.649    117.705    117.767    117.839    117.928    118.042    118.194    118.403      118.7    119.126     119.74    120.621    121.868    123.599     125.94    129.005     132.87    137.539    142.922    148.834    155.021    161.207    167.147    172.658    177.633    182.032    185.862    189.163    191.989    194.398    196.447     198.19    199.672    200.933    202.005    202.917    203.693    204.351    204.908    205.377    205.769    206.093    206.357    206.566    206.726    206.842    206.915     206.95    206.947    206.909    206.836    206.729    206.588    206.413    206.203    205.956    205.672    205.349    204.984    204.575    204.119    203.613    203.052    202.433     201.75    200.999    200.174    199.268    198.276    197.191    196.006    194.715    193.312    191.792     190.15    188.383    186.492    184.475     182.34    180.091    177.738    175.296    172.777    170.198    167.575    164.923    162.257    159.588    156.925    154.276    151.647    149.045    146.477    143.952    141.486    139.093    136.794    134.607    132.552    130.648    128.906    127.335    125.938    124.713    123.653    122.747    121.981    121.342    120.815    120.386    120.042    119.771    119.564    119.414    119.316    119.266    119.264    119.312    119.414    119.576    119.808    120.122    120.535    121.065    121.735    122.568    123.593    124.837    126.329    128.092    130.147    132.506     135.17    138.128    141.356    144.815    148.454    152.216    156.036    159.852    163.604    167.241     170.72    174.011    177.092    179.953     182.59    185.007    187.211    189.214    191.029     192.67    194.153    195.491    196.699    197.789    198.773    199.662    200.465    201.193    201.852    202.451    202.994    203.489    203.939     204.35    204.726     205.07    205.384    205.673    205.938    206.182    206.406    206.613    206.804     206.98    207.143    207.294    207.433    207.563    207.683    207.795    207.898    207.994    208.084    208.168    208.245    208.318    208.385    208.449    208.507    208.562    208.614    208.662    208.707    208.749    208.788    208.825     208.86    208.892    208.923    208.951    208.978    209.003    209.026    209.048    209.069    209.089    209.107    209.124     209.14    209.155     209.17    209.183    209.196    209.208    209.219     209.23    209.239    209.249    209.258    209.266    209.274    209.281    209.288    209.295    209.301    209.307    209.313    209.318    209.323    209.328    209.333    209.337    209.341    209.345    209.348    209.352    209.355    209.359    209.362    209.364    209.367     209.37    209.372    209.375    209.377     209.38    209.382    209.384    209.386    209.388     209.39    209.392    209.393    209.395    209.397    209.398      209.4    209.402    209.403    209.405    209.406    209.408    209.409     209.41    209.412    209.413    209.415    209.416    209.417    209.419     209.42    209.421    209.423    209.424    209.425    209.426    209.428    209.429     209.43    209.432    209.433    209.434    209.436    209.437    209.438     209.44    209.441    209.443    209.444    209.445    209.447    209.448     209.45    209.451    209.453    209.454    209.456    209.457    209.459     209.46    209.462    209.463    209.465    209.467    209.468     209.47    209.472    209.474    209.475    209.477    209.479    209.481    209.482    209.484    209.486    209.488     209.49    209.492    209.494    209.496    209.498      209.5    209.502    209.504    209.507    209.509    209.511    209.513    209.516    209.518     209.52    209.523    209.525    209.528     209.53    209.533    209.535    209.538     209.54    209.543    209.546    209.549    209.551    209.554    209.557     209.56    209.563    209.566    209.569    209.572    209.575    209.578    209.582    209.585    209.588    209.592    209.595    209.598    209.602    209.606    209.609    209.613    209.616     209.62    209.624    209.628    209.632    209.636     209.64    209.644    209.648    209.652    209.657    209.661    209.665     209.67    209.674    209.679    209.684    209.688    209.693    209.698    209.703    209.708    209.713    209.718    209.724    209.729    209.734     209.74    209.745    209.751    209.757    209.762    209.768    209.774     209.78    209.786    209.792    209.799    209.805    209.812    209.818    209.825    209.832    209.839    209.845    209.853     209.86    209.867    209.874    209.882    209.889    209.897    209.905    209.913    209.921    209.929    209.937    209.946    209.954    209.963    209.972     209.98    209.989    209.999    210.008    210.017    210.027    210.037    210.046    210.056    210.066    210.077    210.087    210.098    210.108    210.119     210.13    210.142    210.153    210.164    210.176    210.188      210.2    210.212    210.225    210.237     210.25    210.263    210.276     210.29    210.303    210.317    210.331    210.345    210.359    210.374    210.389    210.404    210.419    210.434     210.45    210.466    210.482    210.498    210.515    210.532    210.549    210.567    210.584    210.602     210.62    210.639    210.658    210.677    210.696    210.716    210.735    210.756    210.776    210.797    210.818     210.84    210.862    210.884    210.906    210.929    210.952    210.976        211    211.024    211.049    211.074    211.099    211.125    211.151    211.178    211.205    211.233     211.26    211.289    211.318    211.347    211.377    211.407    211.438    211.469    211.501    211.533    211.566    211.599    211.633    211.667    211.702    211.738    211.774    211.811    211.848    211.886    211.925    211.964    212.004    212.045    212.086    212.128    212.171    212.215    212.259    212.304     212.35    212.396    212.444    212.492    212.541    212.591    212.642    212.694    212.747      212.8    212.855    212.911    212.968    213.025    213.084    213.144    213.205    213.267    213.331    213.395    213.461    213.528    213.597    213.666    213.737     213.81    213.884    213.959    214.036    214.115    214.195    214.276     214.36    214.445    214.531     214.62     214.71    214.802    214.897    214.993    215.091    215.192    215.294    215.399    215.506    215.615    215.727    215.841    215.958    216.077    216.199    216.324    216.452    216.582    216.716    216.853    216.993    217.136    217.282    217.432    217.585    217.743    217.903    218.068    218.237     218.41    218.587    218.768    218.954    219.144    219.339    219.539    219.744    219.954    220.169     220.39    220.616    220.849    221.087    221.331    221.581    221.838    222.101    222.372    222.649    222.933    223.225    223.525    223.832    224.147    224.471    224.803    225.143    225.493    225.852     226.22"\
"    226.598    226.986    227.383    227.792    228.211     228.64    229.082    229.534    229.998    230.474    230.963    231.464    231.977    232.504    233.044    233.597    234.165    234.746    235.342    235.953    236.578    237.219    237.874    238.546    239.233    239.936    240.655    241.391    242.143    242.911    243.697    244.499    245.319    246.156     247.01    247.881    248.769    249.675    250.598    251.538    252.495     253.47    254.461     255.47    256.494    257.536    258.593    259.666    260.755    261.859    262.979    264.112     265.26    266.422    267.597    268.784    269.984    271.196    272.419    273.652    274.896    276.148     277.41    278.679    279.956     281.24    282.529    283.824    285.124    286.428    287.735    289.044    290.355    291.668    292.981    294.294    295.606    296.917    298.226    299.533    300.836    302.135     303.43    304.721    306.006    307.286     308.56    309.827    311.088    312.341    313.587    314.825    316.055    317.278    318.491    319.697    320.894    322.082    323.261    324.431    325.592    326.745    327.889    329.023    330.149    331.266    332.375    333.475    334.566    335.649    336.724     337.79    338.849      339.9    340.943    341.978    343.006    344.027     345.04    346.046    347.046    348.038    349.024    350.004    350.977    351.943    352.903    353.857    354.805    355.747    356.682    357.611    358.535    359.452    360.363    361.268    362.166    363.059    363.945    364.825    365.699    366.565    367.426     368.28    369.126    369.966    370.799    371.625    372.444    373.255    374.059    374.855    375.643    376.424    377.196    377.961    378.717    379.465    380.204    380.935    381.657    382.371    383.076    383.771    384.458    385.136    385.805    386.465    387.115    387.757    388.389    389.012    389.626     390.23    390.826    391.412    391.989    392.556    393.115    393.664    394.205    394.736    395.259    395.773    396.278    396.774    397.261     397.74    398.211    398.673    399.127    399.572     400.01    400.439    400.861    401.275    401.681    402.079     402.47    402.854    403.231      403.6    403.962    404.318    404.667    405.009    405.344    405.673    405.996    406.312    406.622    406.927    407.225    407.518    407.805    408.087    408.363    408.633    408.899    409.159    409.414    409.665     409.91    410.151    410.387    410.618    410.845    411.068    411.287    411.501    411.711    411.917    412.119    412.317    412.512    412.703     412.89    413.073    413.253     413.43    413.604    413.774    413.941    414.104    414.265    414.423    414.578     414.73    414.879    415.025    415.169     415.31    415.448    415.584    415.717    415.848    415.977    416.103    416.227    416.349    416.468    416.586    416.701    416.814    416.925    417.034    417.142    417.247     417.35    417.452    417.552     417.65    417.747    417.841    417.934    418.026    418.116    418.204    418.291    418.376     418.46    418.542    418.623    418.703    418.781    418.858    418.934    419.008    419.081    419.153    419.223    419.293    419.361    419.428    419.494    419.559    419.623    419.686    419.748    419.808    419.868    419.927    419.985    420.042    420.098    420.153    420.207     420.26    420.312    420.364    420.415    420.465    420.514    420.562     420.61    420.656    420.702    420.748    420.792    420.836    420.879    420.922    420.964    421.005    421.045    421.085    421.125    421.163    421.201    421.239    421.276    421.312    421.348    421.383    421.417    421.451    421.485    421.518    421.551    421.583    421.614    421.645    421.676    421.706    421.736    421.765    421.794    421.822     421.85    421.877    421.904    421.931    421.957    421.983    422.009    422.034    422.058    422.083    422.107     422.13    422.153    422.176    422.199    422.221    422.243    422.264    422.286    422.307    422.327    422.348    422.368    422.387    422.407    422.426    422.445    422.463    422.481    422.499    422.517    422.535    422.552    422.569    422.586    422.602    422.618    422.634     422.65    422.666    422.681    422.696    422.711    422.726     422.74    422.754"
predicted = [float(i) for i in predicted.split()]

X = df.loc[:, 'X'].astype(np.double)
Y = df.loc[:, 'Y'].apply(lambda x: x.replace(',' ,'.')).astype(np.double)

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.grid(alpha=0.4)
ax1.set_title("Fitting AutoInsurSweden.txt dataset with 2-hidden layer NN", size=25, pad=20)
ax1.text(20, 300, r"""The model has an input layer with 1 neurons + 1 bias, 2 hidden FC layers
with 30 neurons + 1 bias each, an output layer with 1 neuron.
Each layer, except the output one, is followed by sigmoid activation.
The Loss function:  $L = \frac{1}{m} \sum_{i=1}^{m}{L_i}$, where $L_i = \sum_{j=1}^{n}{(f(x^{(i)})_j - y^{(i)}_j)^2}$
(number of outputs neurons $n = 1$)
""", verticalalignment="center",
         size=13, alpha=1, color="black", weight=500, linespacing=2)
ax1.text(54, 80, r"""The model was fitted using Batch GD with $\alpha = 1\mathrm{e}{-5}$.
The final loss after 5 mln iterations $L \approx 829.0$.
""", verticalalignment="center",
         size=13, alpha=1, color="black", weight=500, linespacing=2)
ax1.set_xlabel("Number of claims", fontsize=13)
ax1.set_ylabel("Total payment, Swedish Kronor", fontsize=13)
ax1.set_xlim(np.min(X) - 5, np.max(X) + 5)
ax1.set_ylim(-5, np.max(Y) + 5)
ax1.plot(np.linspace(0, 130, 10 * 130 + 1), predicted, c='b', label="Fitted model")
ax1.scatter(X, Y, c='r', marker='.', label="AutoInsurSweden.txt data")
ax1.legend(fontsize=13)
ax1.yaxis.set_major_locator(MultipleLocator(50))
ax1.yaxis.set_minor_locator(MultipleLocator(25))
ax1.xaxis.set_major_locator(MultipleLocator(10))
ax1.xaxis.set_minor_locator(MultipleLocator(5))

# Add image
nn_img = mpimg.imread('./nn_arch_autoInsurSweden.png')
imagebox = OffsetImage(nn_img, zoom=0.25)
ab = AnnotationBbox(imagebox, (10, 300))
ax1.add_artist(ab)

plt.show()
