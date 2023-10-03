import numpy as np
import visgeom as vg
import matplotlib
import matplotlib.pyplot as plt
from pylie import SO3, SE3

"""Exercise 1 - Linear interpolation of poses on the manifold"""


def interpolate_lie_element(alpha, X_1: SE3, X_2: SE3):
    """Perform linear interpolation on the manifold

    :param alpha: A scalar interpolation factor in [0, 1]
    :param X_1: First element
    :param X_2: Second element
    :return: The interpolated element
    """
    # TODO 1: Implement the interpolation (see Example 4.8 in the compendium for hints).
    X = X_1 + alpha * (X_2 - X_1)
    return X

def interpolate_lie_element_seperatly(alpha, X_1: SE3, X_2: SE3):
    """Perform linear interpolation on the manifold

    :param alpha: A scalar interpolation factor in [0, 1]
    :param X_1: First element
    :param X_2: Second element
    :return: The interpolated element
    """
    # TODO 1: Implement the interpolation (see Example 4.8 in the compendium for hints).
    R_1, t_1 = X_1.rotation, X_1.translation
    R_2, t_2 = X_2.rotation, X_2.translation

    # Interpolate rotation
    R = R_1.compose(SO3.Exp(alpha * SO3.Log(R_1.inverse().compose(R_2))))

    # Interpolate translation
    t = (1 - alpha) * t_1 + alpha * t_2

    return SE3(pose_tuple=(R, t))


def main():
    # Define the first pose.
    T_1 = SE3((SO3.from_roll_pitch_yaw(np.pi / 4, 0, np.pi / 2), np.array([[1, 1, 1]]).T))

    # Define the second pose.
    T_2 = SE3((SO3.from_roll_pitch_yaw(-np.pi / 6, np.pi / 4, np.pi / 2), np.array([[1, 4, 2]]).T))

    # Plot the interpolation.
    # Use Qt 5 backend in visualisation.
    matplotlib.use('qt5agg')

    # Create figure and axis.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot the poses.
    vg.plot_pose(ax, T_1.to_tuple())
    vg.plot_pose(ax, T_2.to_tuple())

    # Plot the interpolated poses.
    for alpha in np.linspace(0, 2, 40):
        T = interpolate_lie_element(alpha, T_1, T_2)
        vg.plot_pose(ax, T.to_tuple(), alpha=0.1)

    # Show figure where translation and rotation are interpolated seperatly.
    vg.plot.axis_equal(ax)
    plt.show()

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot the poses.
    vg.plot_pose(ax, T_1.to_tuple())
    vg.plot_pose(ax, T_2.to_tuple())

    # Plot the interpolated poses.
    for alpha in np.linspace(0, 2, 40):
        T = interpolate_lie_element_seperatly(alpha, T_1, T_2)
        vg.plot_pose(ax, T.to_tuple(), alpha=0.1)

    # Show figure where translation and rotation are interpolated seperatly.
    vg.plot.axis_equal(ax)
    plt.show()

if __name__ == "__main__":
    main()
