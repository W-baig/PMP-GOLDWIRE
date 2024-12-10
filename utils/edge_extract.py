from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np

# pts = np.random.rand(30, 2)
# hull = ConvexHull(pts)
# plt.plot(pts[:,0], pts[:,1], 'o')
# for i in hull.simplices:
#     plt.plot(pts[i, 0], pts[i, 1], 'k-')
def convix_hull(pts) :

    hull = ConvexHull(pts)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='b')
        # # for i in hull.simplices:
        # #     ax.plot_trisurf(pts[i, 0], pts[i, 1], pts[i,2], alpha=0.5)
        # for simplex in hull.simplices:
        #     tri = pts[simplex]
        #     ax.plot_trisurf(tri[:, 0], tri[:, 1], tri[:, 2], alpha=0.5, color='orange', edgecolor='k')
        # plt.savefig("/home/wanghao/Projects/PMP-Net-main-JRS/exp/output/test_convix/test_convix.png")

    hull_points = pts[hull.vertices]

    return hull_points

def alpha_shape(pts) :
    return pts