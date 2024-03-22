import plotly.graph_objects as go


def plot_roc_train_test(plot_data):
    labels = list(plot_data.keys())
    all_dfs = [plot_data[label]['train']['df_result'] for label in labels] + [plot_data[label]['test']['df_result'] for label in labels]
    trace_names = [f'{label}_train=>auc:{plot_data[label]["train"]["auc"]:.4f}' for label in labels] + \
                  [f'{label}_test=>auc:{plot_data[label]["test"]["auc"]:.4f}' for label in labels]

    traces = [go.Scatter(x=(1 - df['Spe']), y=df['Sen'], mode="lines+markers", 
                         name=name, visible=False, marker=dict(size=4)) for name, df in zip(trace_names, all_dfs)]
    traces[0].visible = True
    traces[len(labels)].visible = True

    visible_list = [[label in trace.name for trace in traces] for label in labels]
    visible_list = visible_list + [[True] * len(visible_list[0])]

    var_type = labels + ["all"]
    buttons = [{'label': var_type[i], 'method': 'update', 'args': [{'visible': visible_list[i]}]} for i in range(len(visible_list))]

    updatemenus = [dict(
        type="dropdown",
        active=-1,
        x=0.0,
        xanchor='left',
        y=1.33,
        yanchor='top',
        direction='down',
        buttons=buttons,
    )]

    layout = go.Layout(
        xaxis=dict(dtick=0.1),
        yaxis=dict(dtick=0.1),
        height=800,
        width=800,
        updatemenus=updatemenus,
        title="<b><br>ROC<b>",
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()