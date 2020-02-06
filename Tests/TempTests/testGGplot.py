import pandas as pd
from plotnine import ggplot, aes, geom_line, geom_vline
import numpy as np
from GPy.util.squashers import sigmoid


def tanhSigmoid(x):
    return (1 + np.tanh(x)) / 2
def tanhSigmoidHalfWidth(slope, y = 1 - 0.01): # Distance from peak to when y is reached
    return - slope * np.arctanh(1 - 2 * y)

def tanhSigTwoLocIndicatorHeight(location, stop_location, slope):
    return 2 * tanhSigmoid(((stop_location - location) / 2) / slope) - 1

def tanhSigIndicatorWithWidthHeight(width, slope):
    return 2 * tanhSigmoid((width / 2) / slope) - 1

def tanhSigOneLocIndicatorHalfWidth(slope, y = 0.01): # Distance from peak to when y is reached
    return slope * np.arccosh(1 / np.sqrt(y))
def tanhSigOneLocIndicatorLocations(location, slope, y = 0.01):
    width = tanhSigOneLocIndicatorHalfWidth(slope, y)
    return location - width, location + width


# points = pd.DataFrame({'points': np.linspace(-3, 3, 100)})
#
# print(ggplot(points) +
#           # geom_line(aes('points', 'np.cosh(points)'), color='blue') +
#           geom_line(aes('points', 'tanhSigmoid((points - 0) / 0.5)'), color='blue') +
#           geom_line(aes('points', '1 - tanhSigmoid((points - 0) / 0.5)'), color='blue') #+
#           # geom_line(aes('points', 'sigmoid(1 - points)'), color='red') +
#           #geom_line(aes('points', 'sigmoid(points) * sigmoid(points)'), color='orange') +
#           # geom_line(aes('points', '4 * tanhSigmoid(points) * (1 - tanhSigmoid(points))'), color='green') +
#           # geom_line(aes('points', '1 - 4 * tanhSigmoid(points) * (1 - tanhSigmoid(points))'), color='green') #+
#           # geom_line(aes('points', '4 * sigmoid(points) * (1 - sigmoid(points))'), color='yellow') +
#           # geom_line(aes('points', '1 - 4 * sigmoid(points) * (1 - sigmoid(points))'), color='yellow')
# )




widerPoints = pd.DataFrame({'points': np.linspace(-60, 60, 100)})

print(ggplot(widerPoints) +
        geom_line(aes('points', 'tanhSigmoid((points - 5) / 10)'), color='blue') +
        geom_vline(aes(xintercept = '5 + tanhSigmoidHalfWidth(10, 1 - 0.01)'), color='blue') +
        # geom_line(aes('points', '1 - tanhSigmoid((points - 5) / 10)'), color='blue')  # +
        # geom_line(aes('points', '4 * tanhSigmoid((points - 5) / 10) * (1 - tanhSigmoid((points - 5) / 10))'), color='orange') +
        # geom_vline(aes(xintercept = '5 + tanhSigOneLocIndicatorHalfWidth(10, 0.01)'), color='orange')
        # geom_line(aes('points', 'tanhSigmoid((points + 8) / 10) + tanhSigmoid((points - 5) / (-10)) - 1'), color='black') +
        # geom_line(aes('points', 'tanhSigTwoLocIndicatorHeight(-8, 5, 10)'), color='black') #+
        # geom_line(aes('points', '(tanhSigmoid((points + 8) / 10) + tanhSigmoid((points - 5) / (-10)) - 1) / tanhSigTwoLocIndicatorHeight(-8, 5, 10)'), color='orange') #+
        geom_line(aes('points', 'tanhSigmoid((points - 5) / 5) + tanhSigmoid((points - (5 + 6)) / (-5)) - 1'), color='black') +
        geom_line(aes('points', 'tanhSigIndicatorWithWidthHeight(6, 5)'), color='black') +
        geom_line(aes('points', '(tanhSigmoid((points - 5) / 5) + tanhSigmoid((points - (5 + 6)) / (-5)) - 1) / tanhSigIndicatorWithWidthHeight(6, 5)'), color='orange') #+
        # geom_line(aes('points', 'tanhSigmoid((points + 25) / 10) * (1 - tanhSigmoid((points - 35) / 10))'), color='blue') +
        # geom_line(aes('points', 'tanhSigmoid((points + 25) / 10) + tanhSigmoid((points - 35) / (-10)) - 1'), color='blue') +
        # geom_line(aes('points', '1 - (tanhSigmoid((points + 25) / 10) + tanhSigmoid((points - 35) / (-10)) - 1)'), color='green') +
        # geom_line(aes('points', 'tanhSigmoid((points + 25) / 10) + (1 - tanhSigmoid((points - 35) / 10)) - 1'), color='red') +
        # geom_line(aes('points', 'tanhSigmoid((points - 35) / 10) + (1 - tanhSigmoid((points + 25) / 10)) - 1 + 1'), color='yellow') #+
        # geom_line(aes('points', '1 - (tanhSigmoid((points + 25) / 10) + (1 - tanhSigmoid((points - 35) / 10)) - 1)'), color='green') #+
        # geom_line(aes('points', 'tanhSigmoid((points - 5) / 10) * (1 - tanhSigmoid((points - 5) / 10)) * 10 + np.random.randn(100) * 0.2'), color='purple')
)