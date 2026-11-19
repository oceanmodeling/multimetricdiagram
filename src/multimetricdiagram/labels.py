"""Markerstick label de-overlapping: adjust_markerstick_labels() and its geometry helpers.

Originally built on textalloc (Christoffer Kjellson, https://github.com/ckjellson/textalloc).
"""

import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D

_LEADER_GID = 'multimetricdiagram-leader'  # marks leader lines, so a repeat call can replace them
_OBSTACLE_GID = 'multimetricdiagram-obstacle'  # fixed artists that labels avoid
_RINGS = 12  # rings of candidate spots around a marker, from min_distance to max_distance
_DIRECTIONS = 32  # candidate spots per ring, at equally spaced angles


def adjust_markerstick_labels(
    ax: Any,
    markerstick_results: list,
    *,
    leader_color: str = '#404040',
    leader_linewidth: float = 0.7,
    leader_alpha: float = 0.4,
    leader_zorder: float | None = None,
    margin: float = 0.008,
    min_distance: float = 0.013,
    max_distance: float = 0.2,
) -> list[tuple[Any, Line2D | None]]:
    """
    Move the markerstick labels that overlap something to clear spots, with leader lines to
    their markers; labels that overlap nothing stay where add_markerstick() put them.

    The fixed elements are the polar axes' visible tick labels and every artist on ax with gid
    'multimetricdiagram-obstacle', which make_fig_axes() (either diagram) gives its star and
    the text it draws on and around the polar axes (and the multimetric diagram its markerstick
    keys); tag other artists the same way to keep labels off them. Lines with a linestyle are
    avoided as drawn, other artists by their bounding box; gridlines and shading are not
    avoided.

    Every label, kept or moved, must lie wholly inside the plotted sector of the polar axes
    (between its theta limits and within its r-limit), with the same margin.

    Labels are checked in order (markerstick_results order, then each markerstick's labels).
    A label stays, with no leader line, if its rotated text box, with margin around it, is
    inside the sector and meets no marker or stick as drawn, fixed element, or label already
    kept; its own marker and stick need only not overlap it, since its offsets are gaps from its
    marker's edge. Of two labels that overlap only each other, the first stays. Kept labels
    become obstacles for the rest. The others, in order, each take a clear spot (checked the
    same way, but with the margin from their own marker and stick too, and against kept and
    earlier moved labels) among spots tried in rings around the marker, from min_distance to
    max_distance out, else anywhere in the axes: first a spot whose leader line crosses no
    other marker, stick, label or earlier leader line and keeps clear of its own stick (by half
    the stick's linewidth plus the margin, beyond the margin around its marker), next one whose
    leader line at least keeps clear of its own stick, then the one with the shortest leader
    line, then the nearest. If no spot is clear, a label takes the one that overlaps the fewest
    things, and one UserWarning names it. Where a moved label ends up
    depends on its size, not on its 'position' or offsets. Labels keep their rotation, and a
    moved label gets a leader line if its final box does not contain its marker. Call once,
    after all add_markerstick() calls (it needs every marker, stick and label at once); a repeat
    call first returns every label to its add_markerstick() position, so it gives the same
    result, and replaces the leader lines.

    The placement approach was originally built on textalloc by Christoffer Kjellson
    (ckjellson), https://github.com/ckjellson/textalloc.

    Args:
        ax: The polar axes all add_markerstick() calls used.
        markerstick_results: List of add_markerstick() return values, one per call:
                results = []
                for ...:
                    results.append(add_markerstick(ax, metrics, ...))
                adjust_markerstick_labels(ax, results)
            Markersticks without labels still act as obstacles. Each label of a multi-label
            markerstick is placed independently, anchored to that markerstick's marker.
        leader_color, leader_linewidth, leader_alpha: Leader line style.
        leader_zorder: Default 0.1 below the lowest marker zorder, so leader lines sit just
            beneath markers, sticks and labels.
        margin: Gap kept around each label's box: this fraction of the polar axes' width, or
            of its height times the figure's width/height ratio, whichever is larger.
        min_distance, max_distance: Inner and outer rings of the spots first tried around a
            moved label's marker: the gap from the marker to the label's box, as fractions of
            the axes' width (horizontally) and of its height times the figure's width/height
            ratio (vertically).

    Returns:
        [(label, leader_line), ...]: one pair per moved label, in label order; leader_line is
        its Line2D, or None if the label's box contains its marker. Labels that stay are not
        listed, so [] means nothing moved. A repeat call lists only its own moves (the earlier
        leader lines are removed).

    Warns:
        UserWarning: no clear spot anywhere for some label(s) (named in the message); each
            takes the spot that overlaps the fewest things.
    """
    for line in list(ax.lines):  # leader lines from an earlier call
        if line.get_gid() == _LEADER_GID:
            line.remove()

    stick_thetas, stick_rs, stick_markers, stick_lines = [], [], [], []
    labels, label_stick_idx = [], []
    for stick_idx, (marker_artist, stick_line, labels_artists) in enumerate(markerstick_results):
        theta, r = marker_artist.get_offsets()[0]
        stick_thetas.append(float(theta))
        stick_rs.append(float(r))
        stick_markers.append(marker_artist)
        stick_lines.append(stick_line)
        for label in labels_artists:
            if not len(label.get_path().vertices):  # empty text: nothing to place
                continue
            labels.append(label)
            label_stick_idx.append(stick_idx)

    if not labels:
        return []
    for label in labels:  # back to the preset position, undoing an earlier call
        label._set_offset_inches(0.0, 0.0)  # pylint: disable=protected-access

    fig = ax.get_figure()
    fig.draw_without_rendering()  # finalize layout so ax's transform is current
    renderer = fig.canvas.get_renderer()

    stick_anchor_xy = ax.transData.transform(np.column_stack([stick_thetas, stick_rs]))
    obstacle_boxes, obstacle_lines = _obstacles(ax, renderer)

    if leader_zorder is None:
        leader_zorder = min(m.get_zorder() for m in stick_markers) - 0.1

    axes_bbox = ax.get_window_extent(renderer)
    aspect_ratio = fig.get_size_inches()[0] / fig.get_size_inches()[1]
    xdiff = axes_bbox.x1 - axes_bbox.x0
    ydiff = axes_bbox.y1 - axes_bbox.y0
    xmargin, ymargin = xdiff * margin, ydiff * margin * aspect_ratio

    scene = _scene(
        ax,
        stick_markers,
        stick_lines,
        obstacle_boxes,
        obstacle_lines,
        max(xmargin, ymargin),
        renderer,
    )
    # keep labels that are clear where add_markerstick() put them, in input order
    clear, kept_quads = _clear_labels(labels, label_stick_idx, scene)
    labels = [label for label, keep in zip(labels, clear) if not keep]
    label_stick_idx = [si for si, keep in zip(label_stick_idx, clear) if not keep]
    if not labels:
        return []

    anchor_xy = np.array([stick_anchor_xy[si] for si in label_stick_idx])

    gap_min = (xdiff * min_distance, ydiff * min_distance * aspect_ratio)
    gap_max = (xdiff * max_distance, ydiff * max_distance * aspect_ratio)
    # each moved label, in order, takes the best clear spot (see _best_spot()) around its marker,
    # else anywhere in the axes, else the spot that overlaps the fewest things
    final_box = {}
    placed, no_room = list(kept_quads), []
    leaders = []  # (owner, segment) of the earlier moved labels' leader lines
    for ind, label in enumerate(labels):
        si, anchor = label_stick_idx[ind], anchor_xy[ind]
        quad, bbox = _label_quad(label), label.get_window_extent(renderer)
        near, grid = _candidate_shifts(bbox, anchor, axes_bbox, gap_min, gap_max)
        shift = _best_spot(quad, si, anchor, scene, placed, near, leaders)
        if shift is None:
            shift = _best_spot(quad, si, anchor, scene, placed, grid, leaders)
        if shift is None:
            no_room.append(getattr(label, '_text', '?'))
            shift = _fewest_overlaps(quad, scene, placed, np.vstack([near, grid]))
        final_box[ind] = (bbox.x0 + shift[0], bbox.y0 + shift[1], bbox.width, bbox.height)
        if _has_leader(final_box[ind], anchor):
            start = _leader(quad + shift, anchor, si, scene, placed, leaders)[2]
            leaders.append((si, np.array([start, anchor])))
        placed.append(quad + shift)
    if no_room:
        warnings.warn(
            f"adjust_markerstick_labels(): no clear spot for label(s) {', '.join(no_room)}; "
            "each placed where it overlaps the fewest things",
            UserWarning,
            stacklevel=2,
        )

    moved_quads = placed[len(kept_quads) :]
    drawn = []  # (owner, segment) of the leader lines drawn so far, in label order
    moved = []  # (label, leader line or None), returned
    for ind, label in enumerate(labels):
        x_coord, y_coord, _w, _h = final_box[ind]
        bbox = label.get_window_extent(renderer)
        label._set_offset_inches(  # pylint: disable=protected-access
            (x_coord - bbox.x0) / renderer.dpi, (y_coord - bbox.y0) / renderer.dpi
        )
        si, anchor = label_stick_idx[ind], anchor_xy[ind]
        if not _has_leader(final_box[ind], anchor):
            moved.append((label, None))
            continue
        others = kept_quads + [q for j, q in enumerate(moved_quads) if j != ind]
        _crosses, _length, start = _leader(moved_quads[ind], anchor, si, scene, others, drawn)
        drawn.append((si, np.array([start, anchor])))
        polar_pts = ax.transData.inverted().transform([start, anchor])
        (leader,) = ax.plot(
            polar_pts[:, 0],
            polar_pts[:, 1],
            color=leader_color,
            linewidth=leader_linewidth,
            alpha=leader_alpha,
            clip_on=False,
            zorder=leader_zorder,
            gid=_LEADER_GID,
        )
        moved.append((label, leader))
    return moved


def _obstacles(ax, renderer):
    """(boxes, lines) labels avoid on ax: display boxes (x0, y0, x1, y1) of the visible tick
    labels and of the artists with gid _OBSTACLE_GID, except Line2Ds with a drawn line, which
    are returned as lines."""
    boxes, lines = [], []
    for artist in ax.get_children():
        if artist.get_gid() != _OBSTACLE_GID or not artist.get_visible():
            continue
        if (
            isinstance(artist, Line2D)
            and len(artist.get_xdata()) > 1
            and artist.get_linestyle() not in ('None', '')
        ):
            lines.append(artist)
        else:
            boxes.append(artist.get_window_extent(renderer))
    boxes += [
        t.get_window_extent(renderer)
        for t in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels()
        if t.get_visible() and t.get_text()
    ]
    return [tuple(b.extents) for b in boxes], lines


class _Scene(NamedTuple):
    """What a label's rotated box is tested against: shapes [(owner, convex polygon, growth)],
    their bounding boxes grown by growth + margin (for a quick prefilter), the margin (px),
    outside(quad), true where a quad grown by the margin leaves the plotted sector, and own,
    {owner: (marker outline, its growth + margin, stick zones)} for _along_own_stick()."""

    shapes: list
    boxes: np.ndarray
    margin: float
    outside: Callable
    own: dict | None = None


def _scene(ax, markers, sticks, obstacle_boxes, obstacle_lines, margin, renderer):
    """The _Scene of ax: markers and sticks as drawn (owner: markerstick index), obstacle lines
    and boxes (owner None), and the polar axes' plotted sector, all in display coordinates."""
    px_per_pt = renderer.dpi / 72
    shapes, own = [], {}
    for owner, m in enumerate(markers):
        point = m.get_offset_transform().transform(m.get_offsets())[0]
        path = m.get_paths()[0].transformed(Affine2D(m.get_transforms()[0]).translate(*point))
        polygons = path.to_polygons(closed_only=False)
        outline, growth = np.array([point]), 0.0
        if polygons:
            outline = _convex_hull(np.concatenate(polygons))
            growth = m.get_linewidths()[0] * px_per_pt / 2
            shapes.append((owner, outline, growth))
        own[owner] = (outline, growth + margin, [])
    for owner, line in [*enumerate(sticks), *((None, line) for line in obstacle_lines)]:
        xy = line.get_transform().transform(line.get_xydata())
        hw = line.get_linewidth() * px_per_pt / 2
        segments = [xy[i : i + 2] for i in range(len(xy) - 1)]
        shapes += [(owner, seg, hw) for seg in segments]
        if owner is not None:  # its own stick as drawn (projecting caps), plus the margin
            own[owner][2].extend(
                _grown(seg[[0, 1, 1, 0]], hw + margin) for seg in segments if np.ptp(seg, 0).any()
            )
    for x0, y0, x1, y1 in obstacle_boxes:
        shapes.append((None, np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), 0.0))
    boxes = np.array(
        [
            (*(poly.min(axis=0) - g - margin), *(poly.max(axis=0) + g + margin))
            for _owner, poly, g in shapes
        ]
    ).reshape(-1, 4)
    return _Scene(shapes, boxes, margin, _sector_test(ax, margin), own)


def _sector_test(ax, margin):
    """outside(quad): whether a quad (display corners), grown by margin, reaches beyond the polar
    axes' r-limit or outside its theta limits."""
    thetamin, thetamax = ax.get_xlim()
    r_max = ax.get_ylim()[1]
    span = thetamax - thetamin
    to_data = ax.transData.inverted()
    excluded = None
    if span > np.pi:  # the plotted sector is not convex, but the part left out is
        thetas = np.linspace(thetamax, thetamin + 2 * np.pi, 2 + int(np.degrees(2 * np.pi - span)))
        far = np.column_stack([thetas, np.full_like(thetas, 2 * r_max)])
        excluded = ax.transData.transform(np.vstack([[(thetamin, 0.0)], far]))

    def outside(quad):
        grown = _grown(quad, margin)
        theta, r = to_data.transform(grown).T
        if np.any(r > r_max):
            return True
        if excluded is None:
            return bool(np.any(np.mod(theta - thetamin, 2 * np.pi) > span + 1e-9))
        return _convex_overlap(grown, excluded)

    return outside


def _label_quad(label):
    """Display corners (4 x 2, in order) of a text_polar() label's rotated ink box."""
    x0, y0, x1, y1 = label.get_path().get_extents().extents
    return label.get_transform().transform([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _hits(quad, owner, scene, placed):
    """Whether a label's rotated box (display corners), grown by the scene's margin, leaves the
    sector or meets a shape or a placed label's box. The marker and stick of owner (a label at
    its preset position, whose offset is measured from its marker's edge) count without the
    margin; owner None (a moved label) gives every shape the margin."""
    return any(_overlaps(quad, owner, scene, placed))


def _overlaps(quad, owner, scene, placed):
    """Yields True once for each thing a label's rotated box overlaps, as in _hits(): leaving
    the sector, each shape and each placed label's box it meets."""
    if scene.outside(quad):
        yield True
    (x0, y0), (x1, y1) = quad.min(axis=0), quad.max(axis=0)
    b = scene.boxes
    near = np.nonzero((b[:, 0] <= x1) & (b[:, 2] >= x0) & (b[:, 1] <= y1) & (b[:, 3] >= y0))[0]
    for i in near:
        own, poly, growth = scene.shapes[i]
        gap = 0 if owner is not None and own == owner else scene.margin
        if _convex_overlap(_grown(quad, growth + gap), poly):
            yield True
    grown = _grown(quad, scene.margin)
    for other in placed:
        if _convex_overlap(grown, other):
            yield True


def _fewest_overlaps(quad, scene, placed, shifts):
    """The first of shifts at which a moved label's box overlaps the fewest things (see
    _overlaps(), with the margin from its own marker and stick too)."""
    counts = [sum(_overlaps(quad + d, None, scene, placed)) for d in shifts]
    return shifts[int(np.argmin(counts))]


def _clear_labels(labels, label_stick_idx, scene):
    """(clear, kept quads): per label, in order, whether it is clear where it is (see _hits(),
    earlier clear labels included); and the rotated boxes of the clear ones."""
    clear, kept = [], []
    for label, si in zip(labels, label_stick_idx):
        quad = _label_quad(label)
        clear.append(not _hits(quad, si, scene, kept))
        if clear[-1]:
            kept.append(quad)
    return clear, kept


def _candidate_shifts(bbox, anchor, axes_bbox, gap_min, gap_max):
    """(near, grid): translations (k x 2) moving a label's box (bbox) to candidate spots, each
    nearest first: rings around the anchor (see _ring_shifts()); and a grid over the axes."""
    w, h = bbox.width, bbox.height
    xs = np.arange(axes_bbox.x0, axes_bbox.x1 - w, max(w / 2, 2.0))
    ys = np.arange(axes_bbox.y0, axes_bbox.y1 - h, max(h, 2.0))
    grid = np.array([(x, y) for x in xs for y in ys]).reshape(-1, 2)

    centers = grid + (w / 2, h / 2)
    order = np.argsort(np.hypot(*(centers - anchor).T), kind='stable')
    return _ring_shifts(bbox, anchor, gap_min, gap_max), grid[order] - (bbox.x0, bbox.y0)


def _ring_shifts(bbox, anchor, gap_min, gap_max):
    """Translations (k x 2, nearest first) moving a label's box (bbox) to spots around anchor:
    box centers on _RINGS rectangles about anchor, each as far out as leaves a gap (x, y) from
    the anchor to the box, stepped from gap_min to gap_max, at _DIRECTIONS equal angles."""
    half = np.array([bbox.width, bbox.height]) / 2
    angles = 2 * np.pi * np.arange(_DIRECTIONS) / _DIRECTIONS
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    gap_min, gap_max = np.asarray(gap_min, float), np.asarray(gap_max, float)
    centers = []
    for f in np.linspace(0.0, 1.0, _RINGS):
        a, b = half + gap_min + f * (gap_max - gap_min)  # the rectangle's half sides
        reach = 1 / np.maximum(np.abs(unit[:, 0]) / a, np.abs(unit[:, 1]) / b)
        centers.append(anchor + unit * reach[:, None])
    centers = np.concatenate(centers)
    order = np.argsort(np.hypot(*(centers - anchor).T), kind='stable')
    return centers[order] - half - (bbox.x0, bbox.y0)


def _best_spot(quad, owner, anchor, scene, placed, shifts, leaders=()):
    """The best of shifts (nearest first) that leaves the label clear (see _hits(), margin from
    its own marker and stick too): one whose leader line crosses no other markerstick's marker
    or stick, no placed label and no other markerstick's leader in leaders ((owner, segment)
    pairs) and keeps clear of its own stick (see _along_own_stick()), next one whose leader
    at least keeps clear of its own stick, then the shortest leader line, then the nearest spot;
    None if no shift is clear."""
    best, best_key = None, None
    center = quad.mean(axis=0)
    half_diag = 0.5 * np.hypot(*np.ptp(quad, axis=0))
    for d in shifts:
        dist = np.hypot(*(center + d - anchor))
        # pylint: disable-next=unsubscriptable-object  # best_key is a tuple once set
        if best_key is not None and not best_key[0] and dist - half_diag > best_key[2]:
            break  # leaders from here on are longer than the best clean one
        moved = quad + d
        if _hits(moved, None, scene, placed):  # a moved label keeps the margin from its own too
            continue
        crosses, length, start = _leader(moved, anchor, owner, scene, placed, leaders)
        along = crosses and _along_own_stick(np.array([start, anchor]), owner, scene)
        key = (crosses, along, length, dist)
        if best_key is None or key < best_key:
            best, best_key = d, key
    return best


def _has_leader(box, anchor):
    """Whether a moved label's final box (x0, y0, w, h) gets a leader line: unless it contains
    its marker at anchor."""
    x0, y0, w, h = box
    return not (x0 <= anchor[0] <= x0 + w and y0 <= anchor[1] <= y0 + h)


def _leader(quad, anchor, owner, scene, others, leaders=()):
    """(crosses, length, start) of the leader line from a label's rotated box (display
    corners) to its marker at anchor: start is the nearest point of the box, a corner or an edge
    midpoint, the shortest whose line crosses nothing (see _leader_crosses()), else (then
    crosses is True) the shortest that keeps clear of its own stick, else the shortest."""
    center = quad.mean(axis=0)
    u, v = (quad[1] - quad[0]) / 2, (quad[3] - quad[0]) / 2
    lu, lv = np.hypot(*u), np.hypot(*v)
    uh = u / lu if lu else np.array([1.0, 0.0])
    vh = v / lv if lv else np.array([-uh[1], uh[0]])
    rel = np.asarray(anchor) - center
    nearest = center + np.clip(rel @ uh, -lu, lu) * uh + np.clip(rel @ vh, -lv, lv) * vh
    starts = [nearest, *quad, *((quad + np.roll(quad, -1, axis=0)) / 2)]
    starts.sort(key=lambda p: np.hypot(*(anchor - p)))
    fallback = None
    for start in starts:
        segment = np.array([start, anchor])
        if not _leader_crosses(segment, owner, scene, others, leaders):
            return False, float(np.hypot(*(anchor - start))), start
        if fallback is None and not _along_own_stick(segment, owner, scene):
            fallback = start
    start = starts[0] if fallback is None else fallback
    return True, float(np.hypot(*(anchor - start))), start


def _leader_crosses(segment, owner, scene, others, leaders=()):
    """Whether a leader segment (start, anchor; display) crosses another markerstick's marker or
    stick (not owner's), a label box in others, or another markerstick's leader in leaders
    ((owner, segment) pairs; a markerstick's own leaders meet at its marker), or runs along
    owner's own stick (see _along_own_stick())."""
    if _along_own_stick(segment, owner, scene):
        return True
    (x0, y0), (x1, y1) = segment.min(axis=0), segment.max(axis=0)
    b = scene.boxes
    near = np.nonzero((b[:, 0] <= x1) & (b[:, 2] >= x0) & (b[:, 1] <= y1) & (b[:, 3] >= y0))[0]
    for i in near:
        own, poly, _growth = scene.shapes[i]
        if own is not None and own != owner and _convex_overlap(segment, poly):
            return True
    if any(_convex_overlap(segment, seg) for own, seg in leaders if own != owner):
        return True
    return any(_convex_overlap(segment, q) for q in others)


def _along_own_stick(segment, owner, scene):
    """Whether a leader segment (start, anchor; display) comes within half the linewidth plus
    the margin of owner's own stick, beyond its marker's drawn extent grown by the margin."""
    outline, growth, zones = (scene.own or {}).get(owner, (None, 0.0, []))
    if not zones:
        return False
    start, anchor = segment
    t = _exit_fraction(anchor, start - anchor, outline, growth)
    if t >= 1:  # the whole leader is within the marker's extent
        return False
    part = np.array([anchor + t * (start - anchor), start])
    return any(_convex_overlap(part, zone) for zone in zones)


def _exit_fraction(point, d, outline, g):
    """Fraction of d at which the segment point -> point + d leaves a convex outline
    (counterclockwise) with each side moved out by g; an outline of under 3 vertices counts as
    the circle about point through its farthest vertex, plus g. 0 if point is outside."""
    if len(outline) < 3:
        length = np.hypot(*d)
        return (np.hypot(*(outline - point).T).max() + g) / length if length else np.inf
    edges = np.roll(outline, -1, axis=0) - outline
    normals = np.column_stack([edges[:, 1], -edges[:, 0]]) / np.hypot(*edges.T)[:, None]
    toward = normals @ d
    room = g - np.einsum('ij,ij->i', normals, point - outline)
    leaving = toward > 0
    if not leaving.any():
        return np.inf
    return max(float(np.min(room[leaving] / toward[leaving])), 0.0)


def _grown(quad, g):
    """A rectangle's 4 corners (in order) with every side moved out by g."""
    center = quad.mean(axis=0)
    u, v = (quad[1] - quad[0]) / 2, (quad[3] - quad[0]) / 2
    lu, lv = np.hypot(*u), np.hypot(*v)
    uh = u / lu if lu else np.array([1.0, 0.0])
    vh = v / lv if lv else np.array([-uh[1], uh[0]])
    a, b = lu + g, lv + g
    return np.array(
        [center + su * a * uh + sv * b * vh for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    )


def _convex_hull(points):
    """Convex hull of 2-D points, counterclockwise (monotone chain)."""
    pts = sorted(set(map(tuple, np.round(points, 9))))
    if len(pts) < 3:
        return np.array(pts)

    def half(seq):
        hull = []
        for p in seq:
            while (
                len(hull) >= 2
                and (
                    (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
                    - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])
                )
                <= 0
            ):
                hull.pop()
            hull.append(p)
        return hull[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def _convex_overlap(p, q):
    """Whether convex polygons p and q (vertex arrays in order; a segment is 2 vertices)
    overlap, by the separating axis test."""
    for poly in (p, q):
        if len(poly) < 2:
            continue
        edges = np.roll(poly, -1, axis=0) - poly
        for nx, ny in np.column_stack([-edges[:, 1], edges[:, 0]]):
            if nx == 0 and ny == 0:
                continue
            a, b = p @ (nx, ny), q @ (nx, ny)
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True
