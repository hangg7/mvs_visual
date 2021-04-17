#!/usr/bin/env python3
#
# File   : mvs_visual.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 04/15/2021
#
# Distributed under terms of the MIT license.

from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import cv2
import numpy as np
import plotly.graph_objects as go
import trimesh
from plotly.subplots import make_subplots


class PerspectiveCamera(NamedTuple):
    # properties in wh order.
    focal_length: Tuple[float]
    image_size: Tuple[int]
    # in opengl format.
    c2w: np.ndarray  # (4, 4)
    principal_point: Optional[Tuple[float]] = None


class AxisArgs(NamedTuple):
    showgrid: bool = True
    zeroline: bool = True
    showline: bool = True
    ticks: str = 'outside'
    showticklabels: bool = True
    backgroundcolor: str = 'rgb(230, 230, 250)'
    showaxeslabels: bool = False


class Lighting(NamedTuple):
    ambient: float = 0.8
    diffuse: float = 1.0
    fresnel: float = 0.0
    specular: float = 0.0
    roughness: float = 0.5
    facenormalsepsilon: float = 1e-6
    vertexnormalsepsilon: float = 1e-12


class Segment(NamedTuple):
    points: np.ndarray  # (J, 3)
    parents: Optional[np.ndarray] = None  # (J,)
    colors: Optional[Union[List, np.ndarray]] = None  # (3,) or (J, 3)


def get_camera_wireframe(scale: float = 0.3):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    a = 0.5 * np.array([-2, 1.5, -4])
    up1 = 0.5 * np.array([0, 1.5, -4])
    up2 = 0.5 * np.array([0, 2, -4])
    b = 0.5 * np.array([2, 1.5, -4])
    c = 0.5 * np.array([-2, -1.5, -4])
    d = 0.5 * np.array([2, -1.5, -4])
    C = np.zeros(3)
    F = np.array([0, 0, -3])
    camera_points = [a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C, F]
    lines = np.stack([x for x in camera_points]) * scale
    return lines


def get_plane_pts(image_size, camera_scale=0.3, scale_factor=1 / 4):
    Z = -2 * camera_scale
    X0, Y0, X1, Y1 = (
        -camera_scale,
        1.5 / 2 * camera_scale,
        camera_scale,
        -1.5 / 2 * camera_scale,
    )

    # scale image to plane such that it can go outside of the x0x1 range.
    W, H = X1 - X0, Y0 - Y1
    w, h = image_size
    ratio = min(w / W, h / H)
    oW, oH = w / ratio, h / ratio

    X0, Y0, X1, Y1 = -oW / 2, oH / 2, oW / 2, -oH / 2
    wsteps, hsteps = int(w * scale_factor), int(h * scale_factor)
    Ys, Xs = np.meshgrid(
        np.linspace(Y0, Y1, num=hsteps),
        np.linspace(X0, X1, num=wsteps),
        indexing='ij',
    )
    Zs = np.ones_like(Xs) * Z
    plane_pts = np.stack([Xs, Ys, Zs], axis=-1)
    return plane_pts


def get_plane_pts_from_camera(camera, camera_scale=1, scale_factor=1 / 4):
    focal_length = camera.focal_length
    #  principal_point = camera.principal_point
    image_size = camera.image_size
    Z = -(focal_length[0] + focal_length[1]) / 2 * camera_scale
    X0, Y0, X1, Y1 = (
        -image_size[0] / 2 * camera_scale,
        image_size[1] / 2 * camera_scale,
        image_size[0] / 2 * camera_scale,
        -image_size[1] / 2 * camera_scale,
    )

    # scale image to plane such that it can go outside of the x0x1 range.
    W, H = X1 - X0, Y0 - Y1
    w, h = image_size
    ratio = min(w / W, h / H)
    oW, oH = w / ratio, h / ratio

    X0, Y0, X1, Y1 = -oW / 2, oH / 2, oW / 2, -oH / 2
    wsteps, hsteps = int(w * scale_factor), int(h * scale_factor)
    Ys, Xs = np.meshgrid(
        np.linspace(Y0, Y1, steps=hsteps),
        np.linspace(X0, X1, steps=wsteps),
        indexing='ij',
    )
    Zs = np.ones_like(Xs) * Z
    plane_pts = np.stack([Xs, Ys, Zs], dim=-1)
    return plane_pts


def c2w_to_eye_at_up(c2w):
    # in opengl format.
    eye_at_up_c = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    eye_at_up_w = (c2w[None, :3, :3] @ eye_at_up_c[..., None])[..., 0] + c2w[
        None, :3, -1
    ]
    eye, at, up_plus_eye = (eye_at_up_w[0], eye_at_up_w[1], eye_at_up_w[2])
    up = up_plus_eye - eye
    return eye, at, up


def plot_scene(
    plots: Dict[
        str,
        Dict[
            str,
            Union[
                trimesh.Trimesh,
                PerspectiveCamera,
                Tuple[PerspectiveCamera, np.ndarray],
                Segment,
            ],
        ],
    ],
    *,
    viewpoint_c2w: Optional[np.ndarray] = None,
    ncols: int = 1,
    camera_scale: float = 0.3,
    **kwargs,
):
    subplots = list(plots.keys())
    fig = _gen_fig_with_subplots(len(subplots), ncols, subplots)
    axis_args_dict = kwargs.get('axis_args', AxisArgs())._asdict()
    lighting_dict = kwargs.get('lighting', Lighting())._asdict()
    mesh_opacity = kwargs.get('mesh_opacity', 1)
    camera_image_opacity = kwargs.get('camera_img_opacity', 1)
    camera_image_scale_size = kwargs.get('camera_img_scale_size', 1 / 4)
    show_camera_wireframe = kwargs.get('show_camera_wireframe', True)
    marker_size = kwargs.get('marker_size', 2)
    segment_size = kwargs.get('segment_size', 1)
    show_segment = kwargs.get('show_segment', True)

    # Set axis arguments to defaults defined at the top of this file
    x_settings = {**axis_args_dict}
    y_settings = {**axis_args_dict}
    z_settings = {**axis_args_dict}

    # Update the axes with any axis settings passed in as kwargs.
    x_settings.update(**kwargs.get('xaxis', {}))
    y_settings.update(**kwargs.get('yaxis', {}))
    z_settings.update(**kwargs.get('zaxis', {}))

    # in opengl format.
    viewpoint_camera_dict = {'up': {'x': 0, 'y': 1, 'z': 0}}
    viewpoint_eye_at_up = None
    if viewpoint_c2w is not None:
        viewpoint_eye_at_up = c2w_to_eye_at_up(viewpoint_c2w)

    for subplot_idx in range(len(subplots)):
        subplot_name = subplots[subplot_idx]
        traces = plots[subplot_name]
        for trace_name, struct in traces.items():
            if isinstance(struct, trimesh.Trimesh):
                _add_mesh_trace(
                    fig,
                    struct,
                    trace_name,
                    subplot_idx,
                    ncols,
                    lighting=lighting_dict,
                    opacity=mesh_opacity,
                )
            elif isinstance(struct, PerspectiveCamera):
                _add_camera_trace(
                    fig,
                    struct,
                    trace_name,
                    subplot_idx,
                    ncols,
                    camera_scale,
                    show_camera_wireframe=show_camera_wireframe,
                )
            elif isinstance(struct, Segment):
                _add_segment_trace(
                    fig,
                    struct,
                    trace_name,
                    subplot_idx,
                    ncols,
                    marker_size=marker_size,
                    segment_size=segment_size,
                    show_segment=show_segment,
                )
            elif isinstance(struct, tuple):
                camera, image = struct
                _add_camera_trace(
                    fig,
                    camera,
                    trace_name,
                    subplot_idx,
                    ncols,
                    camera_scale,
                    image=image,
                    image_scale_size=camera_image_scale_size,
                    show_camera_wireframe=show_camera_wireframe,
                    image_opacity=camera_image_opacity,
                )
            else:
                raise ValueError(
                    (
                        'struct {} is not a PerspectiveCamera, Segment, '
                        'Trimesh or a Tuple[PerspectiveCamera, Image] object'
                    ).format(struct)
                )

        # Ensure update for every subplot.
        plot_scene = 'scene' + str(subplot_idx + 1)
        current_layout = fig['layout'][plot_scene]
        xaxis = current_layout['xaxis']
        yaxis = current_layout['yaxis']
        zaxis = current_layout['zaxis']

        # Update the axes with our above default and provided settings.
        xaxis.update(**x_settings)
        yaxis.update(**y_settings)
        zaxis.update(**z_settings)

        # cubify the view space.
        x_range = xaxis['range']
        y_range = yaxis['range']
        z_range = zaxis['range']
        ranges = np.array([x_range, y_range, z_range])
        center = ranges.mean(1)
        max_len = (ranges[:, 1] - ranges[:, 0]).max() * 1.1
        ranges = np.stack(
            [center - max_len / 2, center + max_len / 2], axis=0
        ).T.tolist()
        xaxis['range'] = ranges[0]
        yaxis['range'] = ranges[1]
        zaxis['range'] = ranges[2]

        # update camera viewpoint if provided
        if viewpoint_eye_at_up:
            eye, at, up = viewpoint_eye_at_up
            eye_x, eye_y, eye_z = eye.tolist()
            at_x, at_y, at_z = at.tolist()
            up_x, up_y, up_z = up.tolist()

            # scale camera eye to plotly [-1, 1] ranges
            eye_x = _scale_camera_to_bounds(eye_x, x_range, True)
            eye_y = _scale_camera_to_bounds(eye_y, y_range, True)
            eye_z = _scale_camera_to_bounds(eye_z, z_range, True)

            at_x = _scale_camera_to_bounds(at_x, x_range, True)
            at_y = _scale_camera_to_bounds(at_y, y_range, True)
            at_z = _scale_camera_to_bounds(at_z, z_range, True)

            up_x = _scale_camera_to_bounds(up_x, x_range, False)
            up_y = _scale_camera_to_bounds(up_y, y_range, False)
            up_z = _scale_camera_to_bounds(up_z, z_range, False)

            viewpoint_camera_dict['eye'] = {'x': eye_x, 'y': eye_y, 'z': eye_z}
            viewpoint_camera_dict['center'] = {'x': at_x, 'y': at_y, 'z': at_z}
            viewpoint_camera_dict['up'] = {'x': up_x, 'y': up_y, 'z': up_z}

        current_layout.update(
            {
                'xaxis': xaxis,
                'yaxis': yaxis,
                'zaxis': zaxis,
                'aspectmode': 'cube',
                'camera': viewpoint_camera_dict,
            }
        )

    return fig


def _add_mesh_trace(
    fig: go.Figure,  # pyre-ignore[11]
    mesh: trimesh.Trimesh,
    trace_name: str,
    subplot_idx: int,
    ncols: int,
    lighting: Dict = Lighting()._asdict(),
    opacity: float = 1,
):
    """
    Adds a trace rendering a Trimesh object to the passed in figure, with
    a given name and in a specific subplot.

    Args:
        fig: plotly figure to add the trace within.
        mesh: Trimesh object to render.
        trace_name: name to label the trace with.
        subplot_idx: identifies the subplot, with 0 being the top left.
        ncols: the number of subplots per row.
    """

    verts = mesh.vertices
    faces = mesh.faces
    # If mesh has vertex colors defined as texture, use vertex colors
    # for figure, otherwise use plotly's default colors.
    verts_rgb = None
    if mesh.visual.vertex_colors is not None:
        verts_rgb = np.asarray(mesh.visual.vertex_colors[:, :3])

    # Reposition the unused vertices to be "inside" the object
    # (i.e. they won't be visible in the plot).
    verts_used = np.zeros(verts.shape[0], dtype=np.bool)
    verts_used[np.unique(faces)] = True
    verts_center = verts[verts_used].mean(0)
    verts[~verts_used] = verts_center

    row, col = subplot_idx // ncols + 1, subplot_idx % ncols + 1
    fig.add_trace(
        go.Mesh3d(  # pyre-ignore[16]
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            vertexcolor=verts_rgb,
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            lighting=lighting,
            name=trace_name,
            opacity=opacity,
            showlegend=True,
        ),
        row=row,
        col=col,
    )

    # Access the current subplot's scene configuration
    plot_scene = 'scene' + str(subplot_idx + 1)
    current_layout = fig['layout'][plot_scene]

    # update the bounds of the axes for the current trace
    max_expand = (verts.max(0) - verts.min(0)).max()
    _update_axes_bounds(verts_center, max_expand, current_layout)


def _add_camera_trace(
    fig: go.Figure,
    camera: PerspectiveCamera,
    trace_name: str,
    subplot_idx: int,
    ncols: int,
    camera_scale: float,
    image: Optional[np.ndarray] = None,
    marker_size: int = 1,
    image_scale_size: float = 1 / 4,
    show_camera_wireframe: bool = True,
    image_opacity: float = 1,
):
    """
    Adds a trace rendering a PerspectiveCamera object to the passed in figure,
    with a given name and in a specific subplot.

    Args:
        fig: plotly figure to add the trace within.
        camera: the PerspectiveCamera object to render.
        trace_name: name to label the trace with.
        subplot_idx: identifies the subplot, with 0 being the top left.
        ncols: the number of sublpots per row.
        camera_scale: the size of the wireframe used to render the
            PerspectiveCamera object.
    """
    cam_wires = get_camera_wireframe(camera_scale)
    c2w = camera.c2w
    cam_wires_trans = (c2w[:3, :3] @ cam_wires[..., None])[..., 0] + c2w[
        None, :3, -1
    ]
    x, y, z = cam_wires_trans.T

    row, col = subplot_idx // ncols + 1, subplot_idx % ncols + 1
    if show_camera_wireframe:
        fig.add_trace(
            go.Scatter3d(  # pyre-ignore [16]
                x=x,
                y=y,
                z=z,
                marker={'size': 1},
                name=trace_name,
                showlegend=False,
                legendgroup=trace_name,
            ),
            row=row,
            col=col,
        )
    if image is not None:
        H, W = image.shape[:2]
        if show_camera_wireframe:
            plane_pts = get_plane_pts(
                (W, H),
                camera_scale=camera_scale * 1.1,
                scale_factor=image_scale_size,
            )
        else:
            plane_pts = get_plane_pts_from_camera(
                camera, camera_scale=camera_scale, scale_factor=image_scale_size
            )
        h, w = plane_pts.shape[:2]
        plane_pts_trans = (
            (
                (c2w[:3, :3] @ (plane_pts.reshape(-1, 3)[..., None]))[..., 0]
                + c2w[None, :3, -1]
            )
        ).reshape(h, w, 3)
        images_sample = cv2.resize(
            image, None, fx=image_scale_size, fy=image_scale_size
        )
        fig.add_trace(
            go.Scatter3d(  # pyre-ignore[16]
                x=plane_pts_trans[..., 0].reshape(-1),
                y=plane_pts_trans[..., 1].reshape(-1),
                z=plane_pts_trans[..., 2].reshape(-1),
                marker={
                    'color': images_sample.reshape(-1, 3),
                    'size': marker_size,
                },
                mode='markers',
                name=trace_name,
                opacity=image_opacity,
                legendgroup=trace_name,
            ),
            row=row,
            col=col,
        )

    # Access the current subplot's scene configuration
    plot_scene = 'scene' + str(subplot_idx + 1)
    current_layout = fig['layout'][plot_scene]

    flattened_wires = cam_wires_trans
    if not show_camera_wireframe:
        flattened_wires = plane_pts_trans.reshape(-1, 3)
    points_center = flattened_wires.mean(0)
    max_expand = (flattened_wires.max(0) - flattened_wires.min(0)).max()
    _update_axes_bounds(points_center, max_expand, current_layout)


def _add_segment_trace(
    fig: go.Figure,
    segment: Segment,
    trace_name: str,
    subplot_idx: int,
    ncols: int,
    marker_size: int,
    segment_size: Optional[int] = None,
    show_segment: bool = True,
):
    """
    Adds a trace rendering a segment object to the passed in figure, with
    a given name and in a specific subplot.
    Args:
        fig: plotly figure to add the trace within.
        segment: Segment object to render.
        trace_name: name to label the trace with.
        subplot_idx: identifies the subplot, with 0 being the top left.
        ncols: the number of subplots per row.
        marker_size: the size of the rendered points
    """
    points = segment.points
    parents = segment.parents
    colors = segment.colors

    if colors is not None:
        if isinstance(colors, List):
            colors = np.array(colors, dtype=np.uint8)[None].repeat(
                points.shape[0], axis=0
            )
        if colors.shape[1] == 3:
            template = 'rgb(%d, %d, %d)'
            colors = [template % (r, g, b) for r, g, b in colors]
        else:
            raise NotImplementedError('Only support RGB segments right now.')

    row = subplot_idx // ncols + 1
    col = subplot_idx % ncols + 1
    if not show_segment:
        # just show the point cloud.
        fig.add_trace(
            go.Scatter3d(  # pyre-ignore[16]
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                marker={'color': colors, 'size': marker_size},
                mode='markers',
                name=trace_name,
            ),
            row=row,
            col=col,
        )
    else:
        if segment_size is None:
            segment_size = int(np.ceil(marker_size * 0.5))
        if parents is not None:
            fig.add_trace(
                go.Scatter3d(  # pyre-ignore[16]
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    marker={'color': colors, 'size': marker_size},
                    mode='markers',
                    name=trace_name,
                    legendgroup=trace_name,
                ),
                row=row,
                col=col,
            )
            for p, j in zip(parents[1:], np.arange(parents.shape[0])[1:]):
                fig.add_trace(
                    go.Scatter3d(  # pyre-ignore[16]
                        x=points[[p, j], 0],
                        y=points[[p, j], 1],
                        z=points[[p, j], 2],
                        line={
                            'color': [colors[p], colors[j]],
                            'width': segment_size,
                        },
                        showlegend=False,
                        mode='lines',
                        legendgroup=trace_name,
                    ),
                    row=row,
                    col=col,
                )
        else:
            fig.add_trace(
                go.Scatter3d(  # pyre-ignore[16]
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    marker={'color': colors, 'size': marker_size},
                    line={'color': colors, 'width': segment_size},
                    name=trace_name,
                ),
                row=row,
                col=col,
            )

    # Access the current subplot's scene configuration
    plot_scene = 'scene' + str(subplot_idx + 1)
    current_layout = fig['layout'][plot_scene]

    # update the bounds of the axes for the current trace
    points_center = points.mean(0)
    max_expand = (points.max(0) - points.min(0)).max()
    _update_axes_bounds(points_center, max_expand, current_layout)


def _gen_fig_with_subplots(
    batch_size: int, ncols: int, subplot_titles: List[str]
):
    """
    Takes in the number of objects to be plotted and generate a plotly figure
    with the appropriate number and orientation of titled subplots.
    Args:
        batch_size: the number of elements in the batch of objects to be
            visualized.
        ncols: number of subplots in the same row.
        subplot_titles: titles for the subplot(s). list of strings of length
            batch_size.

    Returns:
        Plotly figure with ncols subplots per row, and batch_size subplots.
    """
    fig_rows = batch_size // ncols
    if batch_size % ncols != 0:
        fig_rows += 1  # allow for non-uniform rows
    fig_cols = ncols
    fig_type = [{'type': 'scene'}]
    specs = [fig_type * fig_cols] * fig_rows
    # subplot_titles must have one title per subplot
    fig = make_subplots(
        rows=fig_rows,
        cols=fig_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        column_widths=[1.0] * fig_cols,
    )
    return fig


def _update_axes_bounds(
    verts_center: np.array,
    max_expand: float,
    current_layout: go.Scene,  # pyre-ignore[11]
):
    """
    Takes in the vertices' center point and max spread, and the current plotly
    figure layout and updates the layout to have bounds that include all traces
    for that subplot.
    Args:
        verts_center: tensor of size (3) corresponding to a trace's vertices'
            center point.
        max_expand: the maximum spread in any dimension of the trace's vertices.
        current_layout: the plotly figure layout scene corresponding to the
            referenced trace.
    """
    verts_min = verts_center - max_expand
    verts_max = verts_center + max_expand
    bounds = np.stack([verts_min, verts_max], axis=-1)

    # Ensure that within a subplot, the bounds capture all traces
    old_xrange, old_yrange, old_zrange = (
        current_layout['xaxis']['range'],
        current_layout['yaxis']['range'],
        current_layout['zaxis']['range'],
    )
    x_range, y_range, z_range = bounds
    if old_xrange is not None:
        x_range[0] = min(x_range[0], old_xrange[0])
        x_range[1] = max(x_range[1], old_xrange[1])
    if old_yrange is not None:
        y_range[0] = min(y_range[0], old_yrange[0])
        y_range[1] = max(y_range[1], old_yrange[1])
    if old_zrange is not None:
        z_range[0] = min(z_range[0], old_zrange[0])
        z_range[1] = max(z_range[1], old_zrange[1])

    xaxis = {'range': x_range}
    yaxis = {'range': y_range}
    zaxis = {'range': z_range}
    current_layout.update({'xaxis': xaxis, 'yaxis': yaxis, 'zaxis': zaxis})


def _scale_camera_to_bounds(
    coordinate: float, axis_bounds: Tuple[float, float], is_position: bool
):
    """
    We set our plotly plot's axes' bounding box to [-1,1]x[-1,1]x[-1,1]. As
    such, the plotly camera location has to be scaled accordingly to have its
    world coordinates correspond to its relative plotted coordinates for
    viewing the plotly plot.
    This function does the scaling and offset to transform the coordinates.

    Args:
        coordinate: the float value to be transformed
        axis_bounds: the bounds of the plotly plot for the axis which
            the coordinate argument refers to
        is_position: If true, the float value is the coordinate of a position,
            and so must be moved in to [-1,1]. Otherwise it is a component of a
            direction, and so needs only to be scaled.
    """
    scale = (axis_bounds[1] - axis_bounds[0]) / 2
    if not is_position:
        return coordinate / scale
    offset = (axis_bounds[1] / scale) - 1
    return coordinate / scale - offset
