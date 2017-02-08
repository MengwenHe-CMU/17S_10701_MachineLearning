import hw1_baseball
import numpy

# Question A.1
def mleFromMidSeason(midData):
    return hw1_baseball.safe_batting_average(midData.get('at_bats'), midData.get('hits'))


# Question A.2
def mapFromPreMidSeason(preData, midData):
    numerate=preData.get('hits')+midData.get('hits')
    denominator=preData.get('at_bats')+midData.get('at_bats')
    denominator = numpy.where(denominator > 0, denominator, numpy.ones_like(denominator))
    return numerate.astype(float)/denominator


if __name__ == "__main__":
    preData = hw1_baseball.load_data('pre_season.txt')
    midData = hw1_baseball.load_data('mid_season.txt')
    endData = hw1_baseball.load_data('end_season.txt')

    # Question A.1
    MLE = mleFromMidSeason(midData)

    # Question A.2
    MAP = mapFromPreMidSeason(preData, midData)

    # Question A.3
    hw1_baseball.visualize(preData, midData, endData, MLE, MAP, './')
