from ex2_mean_pose import draw_random_poses
import numpy as np
import visgeom as vg
import matplotlib
import matplotlib.pyplot as plt
from pylie import SO3, SE3

"""Exercise 3 - Propagating uncertainty in backprojection"""


def backproject(u, z, f, c):
    """The inverse camera model backprojects a pixel u with depth z out to a 3D point in the camera coordinate frame.

    :param u: Pixel points, a 2xn matrix of pixel position column vectors
    :param z: Depths, an n-vector of depths.
    :param f: Focal lengths, a 2x1 column vector [fu, fv].T
    :param c: Principal point, a 2x1 column vector [cu, cv].T
    :return: A 3xn matrix of column vectors representing the backprojected points
    """
    # TODO 1: Implement backprojection.
    x_c = z * np.vstack(((u - c) / f, np.ones((1, u.shape[1]))))
    return x_c


def main():
    # Define camera parameters.
    f = np.array([[100, 100]]).T  # "Focal lengths"  [fu, fv].T
    c = np.array([[50, 50]]).T  # Principal point [cu, cv].T

    # Define camera pose distribution.
    R_w_c = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    t_w_c = np.array([[1, 0, 0]]).T
    mean_T_w_c = SE3((SO3(R_w_c), t_w_c))
    Sigma_T_w_c = np.diag(np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01]) ** 2)

    # Define pixel position distribution.
    mean_u = np.array([[50, 50]]).T
    Sigma_u = np.diag(np.array([1, 1]) ** 2)

    # Define depth distribution.
    mean_z = 10
    var_z = 0.1 ** 2

    # TODO 2: Approximate the distribution of the 3D point in world coordinates:
    x_c = backproject(mean_u, mean_z, f, c)
    # TODO 2: Propagate the expected point.
    x_w = mean_T_w_c * x_c

    # TODO 2: Propagate the uncertainty.
    Jf_T_hat_wc = mean_T_w_c.jac_action_Xx_wrt_X(x_c)
    Jf_u = mean_T_w_c.jac_action_Xx_wrt_x() @ np.array([[mean_z / f[0, 0], 0], [0, mean_z / f[1, 0]], [0, 0]])
    Jf_z = mean_T_w_c.jac_action_Xx_wrt_x() @ backproject(mean_u, 1, f, c)
    cov_x_w = Jf_T_hat_wc @ Sigma_T_w_c @ Jf_T_hat_wc.T + Jf_u @ Sigma_u @ Jf_u.T + Jf_z * var_z * Jf_z.T
    print(cov_x_w)

    # Simulate points from the true distribution.
    num_draws = 1000
    random_poses = draw_random_poses(mean_T_w_c, Sigma_T_w_c, num_draws)
    random_u = np.random.multivariate_normal(mean_u.flatten(), Sigma_u, num_draws).T
    random_z = np.random.normal(mean_z, np.sqrt(var_z), num_draws)
    rand_x_c = backproject(random_u, random_z, f, c)
    rand_x_w = np.zeros((3, num_draws))
    for i in range(num_draws):
        rand_x_w[:, [i]] = random_poses[i] * rand_x_c[:, [i]]

    # Plot result.
    # Use Qt 5 backend in visualisation.
    matplotlib.use('qt5agg')

    # Create figure and axis.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot camera poses.
    vg.plot_pose(ax, mean_T_w_c.to_tuple())

    # Plot simulated points.
    ax.plot(rand_x_w[0, :], rand_x_w[1, :], rand_x_w[2, :], 'k.', alpha=0.1)

    # Plot the estimated mean pose.
    vg.plot_covariance_ellipsoid(ax, x_w, cov_x_w)

    # Show figure.
    vg.plot.axis_equal(ax)
    plt.show()


if __name__ == "__main__":
    main()
