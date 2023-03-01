'''
Brief argparse example.
'''

import argparse
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("deg", type=float, help="Degree of monomial.")
parser.add_argument("-y", "--yshift", type=float, default=0,
                    help="Vertical shift of plot.")
parser.add_argument("-x", "--xshift", type=float, default=0,
                    help="Horizontal shift of plot.")
parser.add_argument("-s", "--stretch", type=float, default=1,
                    help="Stretch coefficient.")

def plot_monomial(deg,yshift=0,xshift=0,stretch=1):
    '''
    Plot a monomial from -10 to 10 with the given degree, shift and stretch.

    Arguments:
        deg: float
            degree of the polynomial
        vshift: float
            constant to add to the monomial
        hshift: float
            replace x with (x-hshift)
        stretch: float
            multiply stretch*(x-hshift)
    '''

    xmesh = np.linspace(-10,10,1000)
    vals = yshift + stretch*(xmesh - xshift)**deg
    plt.plot(xmesh, vals)
    if yshift == 0:
        yshift = ''
    else:
        yshift = '{:.2g}+'.format(yshift)
    if xshift == 0:
        xshift = ''
    else:
        xshift = '-{:.2g})'.format(xshift)
    if stretch == 1:
        stretch = ''
    else:
        stretch = '{:.2g}'.format(stretch)
    if xshift != '':
        plt.title('Plot of '+yshift+stretch+'(x$'+xshift+'^{:.2g}$'.format(deg))
    else:
        plt.title('Plot of '+yshift+stretch+'$x^{:.2g}$'.format(deg))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    plot_monomial(args.deg, args.yshift, args.xshift, args.stretch)
