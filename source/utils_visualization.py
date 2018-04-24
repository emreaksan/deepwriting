import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import PIL

"""
Functions to save plots, matrices as image.
"""

def plot_and_get_image(plot_data, fig_height=8, fig_width=12, axis_off=False):
    fig = plt.figure()
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    plt.plot(plot_data)
    if axis_off:
        plt.axis('off')

    img = fig_to_img(fig)
    plt.close(fig)
    return img


def plot_matrix_and_get_image(plot_data, fig_height=8, fig_width=12, axis_off=False, colormap="jet"):
    fig = plt.figure()
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    plt.matshow(plot_data, fig.number)

    if fig_height < fig_width:
        plt.colorbar(orientation="horizontal")
    else:
        plt.colorbar(orientation="vertical")

    plt.set_cmap(colormap)
    if axis_off:
        plt.axis('off')

    img = fig_to_img(fig)
    plt.close(fig)
    return img


def plot_matrices(plot_data, title_data={}, row_colorbar=True, fig_height=8, fig_width=12, show_plot=False):
    """
    Args:
        plot_data: A dictionary with positional index keys such as "00", "01", "10" and "11" where each entry is a two
            dimensional matrix.
        title_data: A dictionary with positional index keys such as "00", "01", "10" and "11" where each entry is plot
            title.
        row_colorbar: Use a common colorbar for two matrices in the same row. If False, each matrix has its own colorbar.
        fig_height:
        fig_width:
        colormap:

    Returns:

    """
    num_latent_neurons, seq_len = plot_data['00'].shape
    aspect_ratio = max(round((seq_len/num_latent_neurons)/2), 1)
    yticks = np.int32(np.linspace(start=0, stop=num_latent_neurons-1, num=min(16, num_latent_neurons)))

    nrows = 1
    if "10" in plot_data.keys():
        nrows = 2

    if row_colorbar:
        fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(fig_width, fig_height), gridspec_kw={"width_ratios": [1, 1, 0.05]})
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(fig_width, fig_height), gridspec_kw={"width_ratios": [1, 0.05, 1, 0.05]})

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    fig.tight_layout()
    plt.setp(axes, yticks=yticks)

    # first row: 00
    if row_colorbar:
        min_value = min(plot_data['00'].min(), plot_data['01'].min())
        max_value = max(plot_data['00'].max(), plot_data['01'].max())

        axes[0,0].imshow(plot_data['00'], vmin=min_value, vmax=max_value, aspect=aspect_ratio)
        axes[0,0].set_title(title_data.get('00', ""))

        colorbar_ref = axes[0,1].imshow(plot_data['01'], vmin=min_value, vmax=max_value, aspect=aspect_ratio)
        axes[0,1].set_title(title_data.get('01', ""))

        ip = InsetPosition(axes[0,1], [1.05, 0, 0.05, 1])
        axes[0,2].set_axes_locator(ip)
        fig.colorbar(colorbar_ref, cax=axes[0,2], ax=[axes[0,0], axes[0,1]])
    else:
        colorbar_ref= axes[0,0].imshow(plot_data['00'], vmin=plot_data['00'].min(), vmax=plot_data['00'].max(), aspect=aspect_ratio)
        axes[0,0].set_title(title_data.get('00', ""))
        ip = InsetPosition(axes[0,0], [1.05, 0, 0.05, 1])
        axes[0,1].set_axes_locator(ip)
        fig.colorbar(colorbar_ref, cax=axes[0,1], ax=axes[0,0])

        colorbar_ref = axes[0,2].imshow(plot_data['01'], vmin=plot_data['01'].min(), vmax=plot_data['01'].max(), aspect=aspect_ratio)
        axes[0,2].set_title(title_data.get('01', ""))
        ip = InsetPosition(axes[0,2], [1.05, 0, 0.05, 1])
        axes[0,3].set_axes_locator(ip)
        fig.colorbar(colorbar_ref, cax=axes[0,3], ax=axes[0,2])


    img = fig_to_img(fig)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return img


def plot_latent_variables(plot_data, fig_height=8, fig_width=12, show_plot=False):
    """

    Args:
        plot_data: a dictionary with keys "q_mu", "q_sigma", "p_mu" and "p_sigma" where each field is a two dimensional
            matrix with size of (num_latent_neurons, seq_len)
        fig_height:
        fig_width:
        colormap:

    Returns:

    """
    num_latent_neurons, seq_len = plot_data['p_mu'].shape
    aspect_ratio = max(round((seq_len/num_latent_neurons)/2), 1)
    yticks = np.int32(np.linspace(start=0, stop=num_latent_neurons-1, num=min(16, num_latent_neurons)))

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(fig_width, fig_height), gridspec_kw={"width_ratios": [1, 1, 0.05]})
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    fig.tight_layout()
    plt.setp(axes, yticks=yticks)

    # mu plots
    mu_min = min(plot_data['q_mu'].min(), plot_data['p_mu'].min())
    mu_max = max(plot_data['q_mu'].max(), plot_data['p_mu'].max())

    axes[0, 0].imshow(plot_data['q_mu'], vmin=mu_min, vmax=mu_max, aspect=aspect_ratio)
    axes[0, 0].set_title("q_mu")
    #axes[0, 0].set_yticks(range(plot_data['q_mu'].shape[0]))

    im_p_mu = axes[0, 1].imshow(plot_data['p_mu'], vmin=mu_min, vmax=mu_max, aspect=aspect_ratio)
    axes[0, 1].set_title("p_mu")
    #axes[0, 1].axis('off')

    ip = InsetPosition(axes[0, 1], [1.05, 0, 0.05, 1])
    axes[0, 2].set_axes_locator(ip)
    fig.colorbar(im_p_mu, cax=axes[0, 2], ax=[axes[0, 0], axes[0, 1]])

    # sigma plots
    mu_min = min(plot_data['q_sigma'].min(), plot_data['p_sigma'].min())
    mu_max = max(plot_data['q_sigma'].max(), plot_data['p_sigma'].max())

    axes[1, 0].imshow(plot_data['q_sigma'], vmin=mu_min, vmax=mu_max, aspect=aspect_ratio)
    axes[1, 0].set_title("q_sigma")
    #axes[1, 0].set_yticks(range(plot_data['q_mu'].shape[0]))

    im_p_sigma = axes[1, 1].imshow(plot_data['p_sigma'], vmin=mu_min, vmax=mu_max, aspect=aspect_ratio)
    axes[1, 1].set_title("p_sigma")
    #axes[1, 1].axis('off')

    ip = InsetPosition(axes[1, 1], [1.05, 0, 0.05, 1])
    axes[1, 2].set_axes_locator(ip)
    fig.colorbar(im_p_sigma, cax=axes[1, 2], ax=[axes[1, 0], axes[1, 1]])

    img = fig_to_img(fig)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return img


def plot_latent_categorical_variables(plot_data, fig_height=8, fig_width=12, show_plot=False):
    """

    Args:
        plot_data: a dictionary with keys "q_pi" and "p_pi" where each field is a two dimensional matrix with size of
            (num_latent_neurons, seq_len)
        fig_height:
        fig_width:
        colormap:

    Returns:

    """
    num_latent_neurons, seq_len = plot_data['q_pi'].shape
    aspect_ratio = max(round((seq_len/num_latent_neurons)/2), 1)
    yticks = np.int32(np.linspace(start=0, stop=num_latent_neurons-1, num=min(16, num_latent_neurons)))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(fig_width, fig_height), gridspec_kw={"width_ratios": [1, 1, 0.05]})
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    fig.tight_layout()
    plt.setp(axes, yticks=yticks)

    min_val = min(plot_data['q_pi'].min(), plot_data['p_pi'].min())
    max_val = max(plot_data['q_pi'].max(), plot_data['p_pi'].max())

    axes[0].imshow(plot_data['q_pi'], vmin=min_val, vmax=max_val, aspect=aspect_ratio)
    axes[0].set_title("q_pi")
    #axes[0, 0].set_yticks(range(plot_data['q_mu'].shape[0]))

    im_p_mu = axes[1].imshow(plot_data['p_pi'], vmin=min_val, vmax=max_val, aspect=aspect_ratio)
    axes[1].set_title("p_pi")
    #axes[1].axis('off')

    ip = InsetPosition(axes[1], [1.05, 0, 0.05, 1])
    axes[2].set_axes_locator(ip)
    fig.colorbar(im_p_mu, cax=axes[2], ax=[axes[0], axes[1]])

    img = fig_to_img(fig)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return img


def fig_to_img (fig):
    """
    Convert a Matplotlib figure to an image in numpy array format.
    Args:
        fig:a matplotlib figure

    Returns:

    """
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1]+(3,))

    return img


def fig_to_img_pil (fig):
    """
    Convert a Matplotlib figure to a PIL Image in RGBA format and return as numpy array.

    Args:
        fig: a matplotlib figure

    Returns: a numpy array of Python Imaging Library ( PIL ) image.

    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    w, h, d = buf.shape

    return np.array(PIL.Image.frombytes("RGBA", (w, h), buf.tostring()))