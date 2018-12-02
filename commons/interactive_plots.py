# -*- coding: utf-8 -*-
"""
Created on 01/08/2018

@author: Borja
"""

import os
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
import datetime
import yaml


dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, "../config/config.yaml")
config = yaml.load(open(filename, 'r'))


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

    py.plot(fig, filename='{}.html'.format("../results/" + config['instrument'] + '_' + str(config['c'])+ '_' + str(config['n_layers'])))