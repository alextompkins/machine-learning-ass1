from matplotlib import pyplot, lines


def draw_line(p1, p2, **kwargs):
    ax = pyplot.gca()
    xmin, xmax = ax.get_xbound()

    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmax - p1[0])
        ymin = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmin - p1[0])

    line = lines.Line2D([xmin, xmax], [ymin, ymax], **kwargs)
    ax.add_line(line)
    return line
