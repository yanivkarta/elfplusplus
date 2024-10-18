'''draw the scores for each label'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
import os
import re
import sys




#open symbolic_categorize_packages.csv
expected_labels = ['kernel','system','service','utility','graphics','database','network','development','security','library','resources','media','miscellaneous'] 

df = pd.read_csv('symbolic_categorize_packages.csv')

#draw the scores for each label
colors = sns.color_palette("hls", len(expected_labels))

df = df.melt(id_vars="package", value_vars=expected_labels, var_name="label", value_name="score") 

#remove the strings from the package column on the plot

sns.scatterplot(data=df, x="package", y="score", hue="label", palette=colors , legend="full", alpha=0.5) 

plt.savefig('sym_scores.png',bbox_inches='tight', dpi=300)