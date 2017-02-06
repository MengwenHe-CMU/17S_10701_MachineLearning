import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy
import os


def safe_batting_average(at_bats, hits):
    divisor = numpy.where(at_bats != 0, at_bats, numpy.ones_like(at_bats))
    return hits.astype(float) / divisor


def visualize_batting_averages(train_stats, test_stats, estimated_batting_averages, axes, title):

    true_batting_averages = safe_batting_average(test_stats['at_bats'], test_stats['hits'])
    train_at_bats = train_stats['at_bats']

    indicator_low_data = train_at_bats < 5

    # plot the estimates vs. the true average for data with more training examples
    axes.scatter(
        estimated_batting_averages[~indicator_low_data], true_batting_averages[~indicator_low_data], color='blue')
    best_fit_high = numpy.polyfit(
        estimated_batting_averages[~indicator_low_data], true_batting_averages[~indicator_low_data], 1)
    axes.plot(
        estimated_batting_averages[~indicator_low_data],
        best_fit_high[0] * estimated_batting_averages[~indicator_low_data] + best_fit_high[1], '-', color='blue')

    # plot the estimates vs. the true average for data with few training examples
    axes.scatter(
        estimated_batting_averages[indicator_low_data], true_batting_averages[indicator_low_data], color='red')
    best_fit_low = numpy.polyfit(
        estimated_batting_averages[indicator_low_data], true_batting_averages[indicator_low_data], 1)
    axes.plot(estimated_batting_averages[indicator_low_data],
              best_fit_low[0] * estimated_batting_averages[indicator_low_data] + best_fit_low[1], '-', color='red')

    # plot the true vs. true line
    axes.plot(true_batting_averages, true_batting_averages, '-', color='green')

    axes.set_title(title)
    axes.set_ylabel('True Batting Averages')
    axes.set_xlabel('Estimated Batting Averages')


def visualize_better_estimator(at_bats_of_interest, test_stats, mle_estimates, map_estimates, axes, title):

    true_batting_averages = safe_batting_average(test_stats['at_bats'], test_stats['hits'])
    map_better = numpy.abs(map_estimates - true_batting_averages) < numpy.abs(mle_estimates - true_batting_averages)
    at_bats = at_bats_of_interest.astype(int)
    range1 = range(0, 11, 1);
    range2 = range(min(at_bats[at_bats > 10]), max(at_bats) + 1, 5);
    bins =  list(range1) + list(range2);
    bin_assignments = numpy.digitize(at_bats, bins=bins)
    fraction_map_better = numpy.zeros(len(bins), dtype=float)
    for index_bin in range(len(bins)):
        count_bin = numpy.count_nonzero(index_bin == bin_assignments)
        if count_bin > 0:
            fraction_map_better[index_bin] = float(
                numpy.count_nonzero(map_better[index_bin == bin_assignments])) / count_bin
    axes.vlines(bins, 0, fraction_map_better)
    axes.set_xlabel('At Bats')
    axes.set_ylabel('MAP Better Fraction')
    axes.set_title(title)


def visualize(pre_season_stats, mid_season_stats, second_half_stats, mle_estimates, map_estimates, output_dir):

    figure_width = 20
    figure_height = 20

    fig = plt.figure(figsize=(figure_width, figure_height))
    grid = gridspec.GridSpec(2, 2)

    visualize_batting_averages(mid_season_stats, second_half_stats, mle_estimates, fig.add_subplot(grid[0, 0]), 'MLE')
    visualize_batting_averages(mid_season_stats, second_half_stats, map_estimates, fig.add_subplot(grid[0, 1]), 'MAP')
    visualize_better_estimator(
        pre_season_stats['at_bats'], second_half_stats, mle_estimates, map_estimates,
        fig.add_subplot(grid[1, 0]), 'Fraction MAP is better vs. pre-season at-bats')
    visualize_better_estimator(
        mid_season_stats['at_bats'], second_half_stats, mle_estimates, map_estimates,
        fig.add_subplot(grid[1, 1]), 'Fraction MAP is better vs. mid-season at-bats')

    output_file = os.path.join(output_dir, 'baseball_visualizations.png')

    fig.savefig(output_file, bbox_inches='tight')
    fig.show()
    # plt.close(fig)


def load_data(path):
    with open(path, 'rt') as data_file:
        field_names = None
        values = list()
        for line in data_file:
            current_values = line.strip().split('\t')
            if field_names is None:
                field_names = list()
                for value in current_values:
                    field_names.append(value)
                    values.append(list())
            else:
                for index_value, value in enumerate(current_values):
                    values[index_value].append(float(value))
        return dict(zip(field_names, [numpy.array(v) for v in values]))