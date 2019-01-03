import plotly
from plotly import tools
import plotly.offline as py
import plotly.io as pio
import plotly.graph_objs as go
import datetime
from commons.constants import config

images_path="C:/Users/borja/Desktop/master_thesis/imagenes_thesis"
#plotly.io.orca.config.executable = 'C:/Users/borja/Anaconda3/orca_app/orca.exe'

def commission_analysis(data_02, data_05, data_1):

    trace02 = go.Box(
        y=data_02,
        name = "0.2%"
    )
    trace05 = go.Box(
        y=data_05,
        name = "0.5%"
    )
    trace1 = go.Box(
        y=data_1,
        name="1%"
    )
    data = [trace02, trace05, trace1]

    layout = go.Layout(
        title="Commissions impact"
    )

    fig = go.Figure(data=data, layout=layout)

    pio.write_image(fig, images_path + "/commissions_analysis_" + config['instrument'] + '_' + str(config['c'])+ '_' + str(config['n_layers']) + ".pdf")
    py.plot(fig, filename='{}.html'.format("../results/commissions_analysis_" + config['instrument'] + '_' + str(config['c'])+ '_' + str(config['n_layers'])))


def price_rewards_actions_utility_plot(date_string, dates, data, rewards, decisions, sharpe):

    data_trace = go.Scatter(
        x=[x for x in dates] if date_string else [datetime.datetime.utcfromtimestamp(int(x) / 1000) for x in dates],
        y=[round(float(x), 4) for x in data],
        name="price ({})".format(config['instrument'])
    )

    rewards_trace = go.Scatter(
        x=[x for x in dates] if date_string else [datetime.datetime.utcfromtimestamp(int(x) / 1000) for x in dates],
        y=[round(float(x), 4) for x in rewards],
        name="rewards",
    )

    decisions_trace = go.Scatter(
        x=[x for x in dates] if date_string else [datetime.datetime.utcfromtimestamp(int(x) / 1000) for x in dates],
        y=[round(float(x), 4) for x in decisions],
        name="decisions",
    )

    sharpe_trace = go.Scatter(
        x=[x for x in dates] if date_string else [datetime.datetime.utcfromtimestamp(int(x) / 1000) for x in dates],
        y=[round(float(x), 4) for x in sharpe],
        name="sharpe",
    )

    fig = tools.make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.001)

    fig.append_trace(data_trace, 1, 1)
    fig.append_trace(rewards_trace, 2, 1)
    fig.append_trace(decisions_trace, 3, 1)
    fig.append_trace(sharpe_trace, 4, 1)

    fig['layout'].update(height=800, width=1000, title='deep-trader {}'.format(config['instrument']))

    pio.write_image(fig,images_path + "/" + config['instrument'] + '_' + str(config['c']) + '_' + str(config['n_layers']) + ".pdf")
    py.plot(fig, filename='{}.html'.format("../results/" + config['instrument'] + '_' + str(config['c'])+ '_' + str(config['n_layers'])))

def convergence_plot(rew_epochs):

    rewards_trace = go.Scatter(
        x=[str(x) for x in range(config['epochs'])],
        y=[round(float(x), 4) for x in rew_epochs],
        name="rewards",
    )

    data = [rewards_trace]

    # Edit the layout
    layout = dict(title='Average Sharpe by Epoch',
                  xaxis=dict(title='Epochs'),
                  yaxis=dict(title='Sharpe'),
                  )

    fig = dict(data=data, layout=layout)

    pio.write_image(fig,images_path + "/convergence_" + config['instrument'] + '_' + str(config['c']) + '_' + str(config['n_layers']) + ".pdf")
    py.plot(fig, filename='{}.html'.format("../results/convergence_" + config['instrument'] + '_' + str(config['c'])+ '_' + str(config['n_layers'])))