import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import plotly
import plotly.subplots
import plotly.io
import plotly.graph_objs as go

img1 = "traversals_fresh.png"
score1 = 0.962
img2 = "traversals_peachy.png"
score2 = 1
img3 = "traversals_gallant.png"
score3 = 1 
font = {'family' : 'normal',
        'size'   : 7}
plt.rc('font')

img1 = mpimg.imread(img1)
img2 = mpimg.imread(img2)
img3 = mpimg.imread(img3)


f, axarr = plt.subplots(1,3)
axarr[0].imshow(img1)
axarr[1].imshow(img2)
axarr[2].imshow(img3)
axarr[1].set_title(f"Disentanglement = {score2}", font)
axarr[0].set_title(f"Disentanglement = {score1}", font)
axarr[2].set_title(f"Disentanglement = {score3}", font)

axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')

f.savefig("qual_quant_contrast.png", dpi=400, bbox_inches='tight', pad_inches=0)



fig = plotly.io.read_json(open("pca_40_lowbeta_gt_latent.plotly.json", 'r'))
fig.update_layout(paper_bgcolor='rgb(255,255,255)', plot_bgcolor='rgb(255,255,255)',title={'text':"Ground truth latent traversal on dSprites", 'x':0.47, 
    'xanchor':'center', 'font_family':'DejaVu Sans', 'font_color':'black'})
fig.update_xaxes(**{'mirror':True, 'ticks':"outside", "showline":True, 'linecolor':'black'})
fig.update_yaxes(**{'mirror':True, 'ticks':"outside", "showline":True, 'linecolor':'black'})
fig.update_layout(margin=go.layout.Margin(l=0,r=0, b=10, t=30))

fig2 = plotly.io.read_json(open("pca_40_highbeta_gt_latent.plotly.json", 'r'))


whole_fig = plotly.subplots.make_subplots(rows=1, cols=2, subplot_titles=["title1", "title2"])
whole_fig.add_trace(fig['data'][0],1,1)
whole_fig.add_trace(fig2['data'][0], 1, 2)
whole_fig.update_layout(paper_bgcolor='rgb(255,255,255)', plot_bgcolor='rgb(255,255,255)', showlegend=False)
whole_fig.update_xaxes(**{'mirror':True, 'ticks':"outside", "showline":True, 'linecolor':'black', 'tickcolor':'black'})
whole_fig.update_yaxes(**{'mirror':True, 'ticks':"outside", "showline":True, 'linecolor':'black', 'tickcolor':'black'})
whole_fig.update_layout(margin=go.layout.Margin(l=0,r=10, b=10, t=30))

whole_fig.layout.annotations[0].update(text="Beta = 0.1", font_family="DejaVu Sans", font_color="black")
whole_fig.layout.annotations[1].update(text="Beta = 4", font_family="DejaVu Sans", font_color="black")
whole_fig

whole_fig.write_image("gt_latents.png")



# import wandb
# api = wandb.Api()

# # Project is specified by <entity/project-name>
# runs = api.runs("atml/atmlbetavae")
# summary_list = [] 
# config_list = [] 
# name_list = [] 
# for run in runs: 
#     # run.summary are the output key/values like accuracy.
#     # We call ._json_dict to omit large files 
#     summary_list.append(run.summary._json_dict) 

#     # run.config is the input metrics.
#     # We remove special values that start with _.
#     config = {k:v for k,v in run.config.items() if not k.startswith('_')}
#     config_list.append(config) 

#     # run.name is the name of the run.
#     name_list.append(run.name)       

# import pandas as pd 
# summary_df = pd.DataFrame.from_records(summary_list) 
# config_df = pd.DataFrame.from_records(config_list) 
# name_df = pd.DataFrame({'name': name_list}) 
# all_df = pd.concat([name_df, config_df,summary_df], axis=1)

# all_df.to_csv("project.csv")


import pandas as pd
import numpy as np
import re

data = pd.read_csv("wandb_export_Ls.csv")
data = data.T
garbage = ~np.logical_or(data.index.str.contains('MAX'), data.index.str.contains('MIN'))
data = data[garbage]
new_index = list(data.index.str.split('final'))
new_index[0] = ['beta', 'L']
new_index = new_index[1:]
data = data.iloc[1:]
for i in range(len(new_index)):
    
    new_index[i][0] = re.findall(r'[\d\.]+', new_index[i][0])[0]
    new_index[i][1] = re.findall(r'L\d+', new_index[i][1])[0][1:]

# index_sort = np.array([[float(elem) for elem in biglist] for biglist in new_index])

xs = [x[0] for x in new_index]
labels = [x[1] for x in new_index]
vals = data[data.columns[0]].values

all_sorted = sorted(list(zip([float(x) for x in xs], [float(val) for val in vals],
    [int(label) for label in labels] )), key = lambda x: (x[0], x[2]))
xs = [elem[2] for elem in all_sorted]
vals = [elem[1] for elem in all_sorted]
labels = [elem[0] for elem in all_sorted]

grouped = {}
# for L in [16,64,128,256]:
for L in [0.1, 1.0, 4.0, 10.0]:
    idxs = []
    for i in range(len(xs)):
        if labels[i] == L:
            idxs.append(i)
    grouped[L] = {"x":[xs[idx] for idx in idxs], "y":[vals[idx] for idx in idxs], 
        "labels":[labels[idx] for idx in idxs]}


# for L in [16,64,128,256]:
for L in [0.1, 1.0, 4.0, 10.0]:
    plt.plot(grouped[L]["x"], grouped[L]["y"], marker='o', label = f"Beta={L}")

plt.xlabel('Sample size L for the Higgins metric')
plt.ylabel('Disentanglement metric')
plt.legend()
plt.savefig('sample_size.png', dpi=400)
