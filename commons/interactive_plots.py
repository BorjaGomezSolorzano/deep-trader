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

path = os.path.abspath(os.path.dirname(__file__))

def plotly_interactive_decisions(date_string, dates, data, rewards, decisions, sharpe, config):

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

    py.plot(fig, filename='{}.html'.format("../results/" + config['instrument'] + '_' + str(config['c'])))