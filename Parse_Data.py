import numpy as np

from Island import Island, Vertex


def parse_data(island_data_fp, sigma_fp, vertex_fp):
    island_data = np.genfromtxt(island_data_fp, delimiter=',')
    sigma_data = np.genfromtxt(sigma_fp, delimiter=',')
    vertex_data = np.genfromtxt(vertex_fp, delimiter=',', dtype=np.uint8)

    sigma_data = sigma_data.reshape((len(sigma_data), 1))
    complete_data = np.hstack([island_data, sigma_data])

    islands = [
        Island(
            np.array([float(row[0]), float(row[1])]),
            0, 0, float(row[2]), sigma=int(row[3])
        )
        for row in complete_data
    ]

    vertices = [
        Vertex(row) for row in vertex_data[:-1]
    ]
    vertices.append(Vertex(vertex_data[-1][:4], is_center=True))

    for i, vertex in enumerate(vertices[:-1]):
        island0 = islands[vertex.ids[0]]
        island1 = islands[vertex.ids[1]]
        island2 = islands[vertex.ids[2]]
        island3 = islands[vertex.ids[3]]

        adj_center1 = island1.center - island0.center
        adj_center2 = island3.center - island2.center

        theta1 = np.arctan(adj_center1[1] / adj_center1[0])
        if adj_center1[0] < 0:
            theta1 += np.pi
        theta1 %= (2*np.pi)

        theta2 = np.arctan(adj_center2[1] / adj_center2[0])
        if adj_center2[0] < 0:
            theta2 += np.pi
        theta2 %= (2*np.pi)

        adj_th0 = island0.theta % (2 * np.pi)
        adj_th1 = (island1.theta + np.pi) % (2 * np.pi)
        adj_th2 = island2.theta % (2 * np.pi)
        adj_th3 = (island3.theta + np.pi) % (2 * np.pi)


        if abs(theta1 - (adj_th0)) < (np.pi / 2):
            if island0.sigma == 1:
                vertex.inout[0] = 'in'
            else:
                vertex.inout[0] = 'out'
        else:
            if island0.sigma == 1:
                vertex.inout[0] = 'out'
            else:
                vertex.inout[0] = 'in'

        if abs(theta1 - adj_th1) < (np.pi / 2):
            if island1.sigma == 1:
                vertex.inout[1] = 'in'
            else:
                vertex.inout[1] = 'out'
        else:
            if island1.sigma == 1:
                vertex.inout[1] = 'out'
            else:
                vertex.inout[1] = 'in'

        if abs(theta2 - (adj_th2)) < (np.pi / 2):
            if island2.sigma == 1:
                vertex.inout[2] = 'in'
            else:
                vertex.inout[2] = 'out'
        else:
            if island2.sigma == 1:
                vertex.inout[2] = 'out'
            else:
                vertex.inout[2] = 'in'

        if abs(theta2 - adj_th3 + np.pi) < (np.pi / 2):
            if island3.sigma == 1:
                vertex.inout[3] = 'in'
            else:
                vertex.inout[3] = 'out'
        else:
            if island3.sigma == 1:
                vertex.inout[3] = 'out'
            else:
                vertex.inout[3] = 'in'

    for i, vertex in enumerate(vertices):
        print("{}: {} {}".format(i, vertex.ids, vertex.inout))

    return

# parse_data(r'./islandcoordinates/single10.txt', r'./savedresults/result.txt', r'./vertices/single10.txt')
