import numpy as np
import cv2
import matplotlib
import svgwrite
matplotlib.use('Agg')


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


def get_bounds(data, factor):
    """
    Auxiliary function for draw_stroke_svg (Original source https://github.com/hardmaru/write-rnn-tensorflow).
    """
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

    return min_x, max_x, min_y, max_y


def draw_stroke_svg(data, factor=0.01, svg_filename='sample.svg', color_labels=None):
    """
    Renders stroke sequence as an SVG image. (Original source https://github.com/hardmaru/write-rnn-tensorflow)

    Args:
        data:
        factor:
        svg_filename:
        color_labels:

    Returns:
    """
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)

    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    abs_x = 25 - min_x
    abs_y = 25 - min_y

    lift_pen = 1
    stroke_width = 2

    if color_labels is None:
        color_labels = np.zeros(data.shape[0])
    prev_color_label = color_labels[0]
    color = svgwrite.rgb(0, 0, 0)

    for idx in range(data.shape[0]):
        prev_x = abs_x
        prev_y = abs_y

        x = float(data[idx, 0])/factor
        y = float(data[idx, 1])/factor

        abs_x += x
        abs_y += y

        if lift_pen == 1:
            p = "M " + str(abs_x) + "," + str(abs_y) + " "
        else:
            p = "M +" + str(prev_x) + "," + str(prev_y) + " L " + str(abs_x) + "," + str(abs_y) + " "

        lift_pen = data[idx, 2]

        if prev_color_label != color_labels[idx]:
            r, g, b = int(np.random.randint(0, 255, 1)), int(np.random.randint(0, 255, 1)), int(
                np.random.randint(0, 255, 1))
            color = svgwrite.rgb(r, g, b)
        prev_color_label = color_labels[idx]

        dwg.add(dwg.path(p).stroke(color, stroke_width).fill(color))

    dwg.save()
    return dwg


def visualize_sample(args):
    factor = 0.001

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    dataset = Dataset(args.data_file)

    for sample_id in args.sample_ids:
        _, stroke_sample, valid_stroke_target = dataset.fetch_sample(sample_id)
        # Prepare the sample and its reconstruction for visualization.
        sample = dataset.undo_normalization(stroke_sample[0])
        svg_path = os.path.join(args.out_dir, "real_" + str(sample_id) + '.svg')
        draw_stroke_svg(sample, factor=factor, svg_filename=svg_path)


if __name__ == '__main__':
    import argparse
    import os
    from dataset_hw import HandWritingDatasetConditional as Dataset

    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument('-D', '--data_file', action="store", dest="data_file", required=True, help='Data file path.')
    parser.add_argument('-O', '--out_dir', action="store", dest="out_dir", required=True, help='Output directory to save images.')
    parser.add_argument('-S', '--sample_ids', nargs='+', type=int, dest="sample_ids", help='List of sample indices to visualize', required=True)
    args = parser.parse_args()

    visualize_sample(args)
