from typing import Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import display
import plotly.colors
from plotly import subplots
from PIL import ImageColor
seq = px.colors.sequential

# Palette of colors
PALETTES = [seq.Blues_r, seq.Greens_r, seq.Oranges_r, seq.Purples_r, seq.Reds_r, seq.Greys_r, seq.RdPu_r ]



#This function allows to visualize the statistics related to the results
def visualize_stats(dfs: pd.DataFrame):

    thresholds = np.unique([ v[1] for v in dfs[0].columns]) #Set of noise values

    sorted = np.argsort(thresholds.astype(float))
    thresholds = thresholds[sorted]

    y_titles = [df.name for df in dfs]

    #Plot representations
    f1 = bar_with_slider(thresholds, dfs, x_title="data",y_titles=y_titles, slide_title="thresholds")

    display(f1)

def add_bar_plots(fig: go.Figure, df:pd.DataFrame, slider:List, methods:List, des_dict: Dict, color_range:np.ndarray, col, row):
    # Add traces, one for each slider step

    n_base = len(fig.data)
    showlegend = row == 1
    for slide in slider:
        # add traces, one for each method, for the current slider step
        for i,method in enumerate(methods):
            x = df.index.values
            y = df[(method,slide)].values

            des_name = method.split(" ")[0]

            color = get_color(PALETTES[list(des_dict.keys()).index(des_name)], color_range[des_dict[des_name].index(method)])
            
            # Add a scatter plot for the current slider value of the current method
            fig.add_trace(
                go.Bar(
                    visible=False,
                    marker_color=color,
                    #line=dict(color=colors[i], width=2),
                    name=method,
                    legendgroup=method,
                    x=x,
                    y=y,
                    showlegend=showlegend,
                ),
                row=row,
                col=col
            )

    # Make the first traces visible
    n_m = len(methods)
    for s in fig.data[n_base:n_base+n_m]:
        s.visible = True

    return fig

def build_layout(x_title:str,y_title:str,slide_title:str) -> go.Layout:
    # define the layout
    return  go.Layout(
        title=f"{y_title} sliding {slide_title}",
        xaxis=dict(
            title=x_title
        ),
        yaxis=dict(
            title=y_title
        ) )

# This function allows to display a the benchmark results in a 2D scatter plot with a slider to manually iterate over the third dimension
def bar_with_slider(slider,  # Set of values of the third dimension
                        dfs:List[pd.DataFrame],      # Dataframe
                        x_title, # title for the x axis of the 2D plot
                        y_titles:List[str],
                        slide_title # title for the slider
                       ) -> List[go.Figure]: 
    assert len(dfs) == len(y_titles)
       
    # number of methods
    methods = list(np.unique([ v[0] for v in dfs[0].columns]))
    methods.sort(key=lambda x : "".join(x.split(" ")[1:]))

    des_dict = dict()
    for entry in methods:
        splitted = entry.split(" ")
        des_name = splitted[0]
        des_set = entry

        if des_name in des_dict:
            des_dict[des_name].append(des_set)
        else:
            des_dict[des_name] = [des_set]

    max_sets = max([len(v) for v in des_dict.values()])
    color_range = np.linspace(0.2, 0.7, max_sets)

    # Create figures
    fig = subplots.make_subplots(
        rows=len(dfs),
        cols=1,
        print_grid=False,
        subplot_titles=y_titles)

    for i,df in enumerate(dfs):
        add_bar_plots(fig, df, slider=slider, methods=methods, des_dict= des_dict, color_range=color_range, col=1, row=i+1)

    # Create and add slider
    num_m = len(methods)
    num_traces = num_m*slider.shape[0]
    steps = []
    for i in range(slider.shape[0]):
        step = dict(
            method="restyle",
            label = str(slider[i]),
            # each step will hide all the plots
            args=[{"visible": [False] * len(fig.data)}] 
        )
        for j in range(len(dfs)):
            # then will make visible only the plots of the current step
            step["args"][0]["visible"][j*num_traces + i*num_m: j*num_traces + (i+1)*num_m] = [True]*num_m
        steps.append(step)

    # build the slider
    sliders = [dict(
        active=10,
        currentvalue={"prefix":  f"{slide_title}: "},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
            sliders=sliders,
            #scene = scene,
            barmode='group',
            # width=840,
            height=800
        )

    return fig


def get_color(colorscale_name, loc):
    from _plotly_utils.basevalidators import ColorscaleValidator
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...] 
    colorscale = cv.validate_coerce(colorscale_name)
    
    if hasattr(loc, "__iter__"):
        return [get_continuous_color(colorscale, x) for x in loc]
    return get_continuous_color(colorscale, loc)

def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
    
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )