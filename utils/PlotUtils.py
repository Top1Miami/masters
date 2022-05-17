import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_quantiles(df, path, score_field):
    melif_df = df.loc[df['model'] == 'melif']
    mq1_plot = []
    mq3_plot = []
    m_mean_plot = []

    pmelif_df = df.loc[df['model'] == 'pmelif']
    pmq1_plot = []
    pmq3_plot = []
    pm_mean_plot = []

    for i in range(1, 11):
        melif_scores = melif_df.loc[melif_df['features_number'] == i][score_field]
        mq1_plot.append(melif_scores.quantile(0.25))
        mq3_plot.append(melif_scores.quantile(0.75))
        m_mean_plot.append(melif_scores.mean())

        pmelif_scores = pmelif_df.loc[pmelif_df['features_number'] == i][score_field]
        pmq1_plot.append(pmelif_scores.quantile(0.25))
        pmq3_plot.append(pmelif_scores.quantile(0.75))
        pm_mean_plot.append(pmelif_scores.mean())

    ax = plt.gca()

    plt.plot(range(1, 11), mq1_plot, label='melif quantile 1', color='lightcoral')
    plt.plot(range(1, 11), mq3_plot, label='melif qunatile 3', color='darkred')
    plt.plot(range(1, 11), m_mean_plot, label='melif mean', color='red')
    plt.plot(range(1, 11), pmq1_plot, label='pmelif quantile 1', color='lightblue')
    plt.plot(range(1, 11), pmq3_plot, label='pmelif quantile 3', color='darkblue')
    plt.plot(range(1, 11), pm_mean_plot, label='pmelif mean', color='blue')

    ax.legend()
    ax.set_xticks(range(1, 11))
    ax.set_ylim([0.0, None])
    ax.set_xlim([1.0, None])
    plt.grid()
    plt.savefig(path)
    plt.close()


def plot_std(df, path, score_field, max_features_select, palette=None):
    if palette is None:
        palette = ['red', 'blue']
    sns.set_style('whitegrid')
    ax = sns.lineplot(x='features_number', y=score_field, hue='model',
                      data=df, ci='sd', palette=palette)
    ax.set_xticks(range(1, max_features_select + 1))
    ax.set_ylim([0.0, None])
    ax.set_xlim([1.0, None])
    plt.grid()
    plt.savefig(path)
    plt.close()


def plot_no_std(df, path, score_field, max_features_select, palette=None):
    if palette is None:
        palette = ['red', 'blue']
    ax = sns.lineplot(x='features_number', y=score_field, hue='model',
                      data=df, ci=None, palette=palette)
    ax.set_xticks(range(1, max_features_select + 1))
    ax.set_ylim([0.0, None])
    ax.set_xlim([1.0, None])
    plt.grid()
    plt.legend(bbox_to_anchor=(0.85, 1), loc=2, borderaxespad=0.)
    plt.savefig(path)
    plt.close()


def plot_single(df, path, score_field):
    ax = sns.lineplot(x='alpha', y=score_field,
                      data=df, ci=None)
    ax.set_xticks(np.arange(0, 1, 0.05))
    ax.set_ylim([0.0, None])
    ax.set_xlim([0.1, None])
    plt.grid()
    plt.legend(bbox_to_anchor=(0.85, 1), loc=2, borderaxespad=0.)
    plt.savefig(path)
    plt.close()


def plot_no_std_no_color(df, path, score_field, max_features_select):
    ax = sns.lineplot(x='features_number', y=score_field, hue='model',
                      data=df, ci=None)
    ax.set_xticks(range(1, max_features_select + 1))
    ax.set_ylim([0.0, None])
    ax.set_xlim([1.0, None])
    plt.grid()
    plt.legend(bbox_to_anchor=(0.85, 1), loc=2, borderaxespad=0.)
    plt.savefig(path)
    plt.close()
