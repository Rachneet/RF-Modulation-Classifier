import plotly.graph_objects as go
import plotly


def update_axis(fig, axis, tick0=None, dtick=0.1, range=None, title=None, fontsize=15):
    """
    :param fig: plotly figure object
    :param axis: axis to modify 'x' or 'y'
    :param dtick: integer value for tick spacing
    :param title: axis title
    :param fontsize: label font size
    :return: None
    """
    if axis == 'x':
        fig.update_xaxes(
            automargin=True,
            showline=True,
            ticks='inside',
            mirror=True,
            tickfont=dict(family="times new roman", size=15, color='black'),
            linecolor='black',
            linewidth=1,
            title=dict(
                font=dict(
                    # family="sans-serif",
                    size=fontsize,
                    color="black"
                ),
                text=title
            )
        )

    elif axis == 'y':
        fig.update_yaxes(
            automargin=True,
            showline=True,
            mirror=True,
            tickfont=dict(family="times new roman", size=15, color='black'),
            linecolor='black',
            tick0=tick0,
            dtick=dtick,
            range=range,
            linewidth=1,
            title=dict(
                font=dict(
                    # family="sans-serif",
                    size=fontsize,
                    color="black"
                ),
                text=title
            )
        )


def update_layout(fig,
                  title=None,
                  width=None,
                  height=None,
                  plotbg='rgba(0,0,0,0)',
                  showlegend=False,
                  legend_x=0,
                  legend_y=0,
                  legend_orientation='v'
                  ):
    """
    :param fig: plotly figure object
    :param title: title of the plot
    :param width: plot width
    :param height: plot height
    :param plotbg: plot background color
    :param showlegend: boolean value to show legend or hide it
    :param legend_x: int value for x axis of legend box
    :param legend_y: int value for y axis of legend box
    :return:
    """
    fig.update_layout(
        title_text=title,
        # margin=dict(b=260, l=0, r=150, t=20), # for small boxes
        # margin=dict(b=350, l=0, r=200, t=20),  # normal
        margin=go.layout.Margin(
            l=0,  # left margin
            r=110,  # right margin  130
            b=160,  # bottom margin
            t=20,  # top margin
        ),
        title_x=0.50,
        # title_y=0.90,
        paper_bgcolor='white',
        plot_bgcolor=plotbg,
        xaxis={"mirror": "all"},
        showlegend=showlegend,
        legend=dict(
            # bordercolor='black',
            # borderwidth=1,
            bgcolor='rgba(0,0,0,0)',
            orientation=legend_orientation,
            itemsizing='constant',
            x=legend_x,
            y=legend_y,
            font=dict(
                # family="sans-serif",
                size=10,
                color="black"
            ),
            traceorder='normal'
        ),
        width=width,
        height=height
    )


def save_fig(fig, figname, width=None, height=None):
    """
    :param fig: plotly figure object
    :param figname: name of saved figure
    :param width: figure width
    :param height: figure height
    :return:
    """
    plotly.offline.plot(figure_or_data=fig, image_width=width, image_height=height, filename=figname, image='svg')


def draw_grid_lines(fig, line_ax, color='grey', x_len=4.3):
    """
    :param fig: plotly figure object
    :param line_ax: list of y axis points where you need the grid lines
    :param color: grid line color
    :param x_len: len of grid line in x axis
    :return: None
    """
    for i in line_ax:
        fig.add_shape(type="line", x0=-0.3, y0=i, x1=x_len, y1=i,  # change
                      line=dict(
                          color=color,
                          width=1,
                          dash="dash",
                      ),
        )


def annotate_plot(fig, text, x_axis, y_axis):
    """
    :param fig: plotly figure object
    :param text: text
    :param x_axis: text x coordinate
    :param y_axis: text 7 coordinate
    :return: None
    """
    fig.add_annotation(dict(font=dict(color="black", size=15,family='times new roman'),
                        x=x_axis,
                        y=y_axis,
                        showarrow=False,
                        text=text,
                        xref="paper",
                        yref="paper"))