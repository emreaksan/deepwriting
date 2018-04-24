import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import svgwrite

def draw_stroke_cv2(strokes_abs, size=(120,1200), margin=(10,20), pen_width=4, factor=1, output_path=None):
    """
    Draws strokes that are used in network training. The pen information consists of 2 possible values namely 0 and 1.

    Args:
        strokes_abs:
        size:
        margin:
        pen_width:
        factor:
        output_path:

    Returns:
    """
    if size is None:
        width = int(strokes_abs[:,0].max() * factor + margin[1])
        factor_w = factor
        height = int(strokes_abs[:,1].max() * factor + margin[0])
        factor_h = factor
    else:
        # Scale between 0 and 1 first.
        max_ = strokes_abs[:,0:2].max(0)
        min_ = strokes_abs[:,0:2].min(0)
        strokes_abs[:, 0:2] = (strokes_abs[:,0:2] - min_) / (max_ - min_)
        width = size[1]
        factor_w = size[1]-2*margin[1]
        height = size[0]
        factor_h = size[0]-2*margin[0]

    image = np.zeros((height, width, 1), np.uint8)
    image[:, :] = 255

    new_line = False
    for idx in range(strokes_abs.shape[0]):
        x = strokes_abs[idx,0]
        y = strokes_abs[idx,1]
        pen = round(strokes_abs[idx, 2])

        if (idx == 0) and (pen == 0):
            start_p = (int(x * factor_w) + margin[1], int(y * factor_h) + margin[0])
        elif pen == 1: # Start a new line.
            new_line = True
        elif pen == 0 and new_line:
            start_p = (int(x * factor_w) + margin[1], int(y * factor_h) + margin[0])
            new_line = False
        elif pen == 0:
            end_p = (int(x * factor_w) + margin[1], int(y * factor_h) + margin[0])
            cv2.line(image, start_p, end_p, 0, pen_width, 8)
            start_p = end_p

    if output_path:
        cv2.imwrite(output_path, image)

    return image

def draw_stroke_cv2_colored(strokes_abs, size=(120,1200), margin=(10,20), pen_width=4, factor=1, color_labels=None, black_zero=True, output_path=None):
    """
    Draws strokes that are used in network training. The pen information consists of 2 possible values namely 0 and 1.

    Args:
        strokes_abs:
        size:
        margin:
        pen_width:
        factor:
        output_path:

    Returns:
    """
    if size is None:
        width = int(strokes_abs[:,0].max() * factor + margin[1])
        factor_w = factor
        height = int(strokes_abs[:,1].max() * factor + margin[0])
        factor_h = factor
    else:
        # Scale between 0 and 1 first.
        max_ = strokes_abs[:,0:2].max(0)
        min_ = strokes_abs[:,0:2].min(0)
        strokes_abs[:, 0:2] = (strokes_abs[:,0:2] - min_) / (max_ - min_)
        width = size[1]
        factor_w = size[1]-2*margin[1]
        height = size[0]
        factor_h = size[0]-2*margin[0]

    image = np.zeros((height, width, 3), np.uint8)
    image[:, :] = 255

    new_line = False
    prev_color_label = color_labels[0]
    color_code = (int(np.random.randint(0, 255, 1)), int(np.random.randint(0, 255, 1)), int(np.random.randint(0, 255, 1)))
    for idx in range(strokes_abs.shape[0]):
        if black_zero and color_labels[idx] == 0:
            color_code = (0,0,0)
        elif prev_color_label != color_labels[idx]:
            color_code = (int(np.random.randint(0, 255, 1)), int(np.random.randint(0, 255, 1)), int(np.random.randint(0, 255, 1)))

        prev_color_label = color_labels[idx]
        x = strokes_abs[idx,0]
        y = strokes_abs[idx,1]
        pen = round(strokes_abs[idx, 2])

        if (idx == 0) and (pen == 0):
            start_p = (int(x * factor_w) + margin[1], int(y * factor_h) + margin[0])
        elif pen == 1: # Start a new line.
            new_line = True
        elif pen == 0 and new_line:
            start_p = (int(x * factor_w) + margin[1], int(y * factor_h) + margin[0])
            new_line = False
        elif pen == 0:

            end_p = (int(x * factor_w) + margin[1], int(y * factor_h) + margin[0])
            cv2.line(image, start_p, end_p, color_code, pen_width, 8)
            start_p = end_p

    if output_path:
        cv2.imwrite(output_path, image)

    return image

def draw_stroke_svg(data, factor=0.01, svg_filename='sample.svg'):
    """
    Renders stroke sequence as an SVG image.

    Args:
        data:
        factor:
        svg_filename:

    Returns:

    """

    def get_bounds(data, factor):
        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0

        abs_x = 0
        abs_y = 0
        for i in range(len(data)):
            x = float(data[i, 0])/factor
            y = float(data[i, 1])/factor
            abs_x += x
            abs_y += y
            min_x = min(min_x, abs_x)
            min_y = min(min_y, abs_y)
            max_x = max(max_x, abs_x)
            max_y = max(max_y, abs_y)

        return (min_x, max_x, min_y, max_y)

    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50+max_x-min_x, 50+max_y-min_y)

    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    lift_pen = 1

    abs_x = 25-min_x
    abs_y = 25-min_y
    p = "M%s,%s "%(abs_x, abs_y)

    command = "m"

    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i, 0])/factor
        y = float(data[i, 1])/factor
        lift_pen = data[i, 2]
        p += command+str(x)+","+str(y)+" "

    the_color = "black"
    stroke_width = 2

    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))

    dwg.save()
    #display(SVG(dwg.tostring()))

    return dwg