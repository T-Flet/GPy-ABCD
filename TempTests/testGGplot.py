import pandas as pd
from plotnine import ggplot, geom_line, aes
import numpy as np
from GPy.util.squashers import sigmoid


def tanhSigmoid(x):
    return (1 + np.tanh(x)) / 2
def tanhSigTwoLocIndicatorHeight(location, stop_location, slope):
    return 2 * tanhSigmoid(((stop_location - location) / 2) / slope) - 1


# points = pd.DataFrame({'points': np.linspace(-3, 3, 100)})
#
# print(ggplot(points) +
#         # geom_line(aes('points', 'np.cosh(points)'), color='blue') +
#         geom_line(aes('points', 'tanhSigmoid((points - 0) / 1)'), color='blue') +
#         geom_line(aes('points', '1 - tanhSigmoid((points - 0) / 1)'), color='blue') +
#         # geom_line(aes('points', 'sigmoid(1 - points)'), color='red') +
#         #geom_line(aes('points', 'sigmoid(points) * sigmoid(points)'), color='orange') +
#         geom_line(aes('points', '4 * tanhSigmoid(points) * (1 - tanhSigmoid(points))'), color='green') +
#         geom_line(aes('points', '1 - 4 * tanhSigmoid(points) * (1 - tanhSigmoid(points))'), color='green') #+
#         # geom_line(aes('points', '4 * sigmoid(points) * (1 - sigmoid(points))'), color='yellow') +
#         # geom_line(aes('points', '1 - 4 * sigmoid(points) * (1 - sigmoid(points))'), color='yellow')
#       )




widerPoints = pd.DataFrame({'points': np.linspace(-60, 60, 100)})

print(ggplot(widerPoints) +
        geom_line(aes('points', 'tanhSigmoid((points + 8) / 10) + tanhSigmoid((points - 5) / (-10)) - 1'), color='black') +
        geom_line(aes('points', 'tanhSigTwoLocIndicatorHeight(-8, 5, 10)'), color='gray') +
        geom_line(aes('points', '(tanhSigmoid((points + 8) / 10) + tanhSigmoid((points - 5) / (-10)) - 1) / tanhSigTwoLocIndicatorHeight(-8, 5, 10)'), color='orange') +
        # geom_line(aes('points', '4 * tanhSigmoid((points - 5) / 10) * (1 - tanhSigmoid((points - 5) / 10))'), color='orange') +
        # geom_line(aes('points', 'tanhSigmoid((points + 25) / 10) * (1 - tanhSigmoid((points - 35) / 10))'), color='blue') +
        # geom_line(aes('points', 'tanhSigmoid((points + 25) / 10) + tanhSigmoid((points - 35) / (-10)) - 1'), color='blue') +
        # geom_line(aes('points', '1 - (tanhSigmoid((points + 25) / 10) + tanhSigmoid((points - 35) / (-10)) - 1)'), color='green') +
        # geom_line(aes('points', 'tanhSigmoid((points + 25) / 10) + (1 - tanhSigmoid((points - 35) / 10)) - 1'), color='red') +
        geom_line(aes('points', 'tanhSigmoid((points - 35) / 10) + (1 - tanhSigmoid((points + 25) / 10)) - 1 + 1'), color='yellow') #+
        # geom_line(aes('points', '1 - (tanhSigmoid((points + 25) / 10) + (1 - tanhSigmoid((points - 35) / 10)) - 1)'), color='green') #+
        # geom_line(aes('points', 'tanhSigmoid((points - 5) / 10) * (1 - tanhSigmoid((points - 5) / 10)) * 10 + np.random.randn(100) * 0.2'), color='purple')
      )