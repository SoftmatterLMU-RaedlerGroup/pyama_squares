#! /usr/bin/env python3
DOC = """© 2020 Daniel Woschée <daniel.woschee@physik.lmu.de>

Create squared bounding boxes from cell contours.
"""
import os
import pickle
import sys

import numpy as np


AREAS = {
        # Named areas in µm²
        "Q20": 20**2,
        "Q25": 25**2,
        "Q30": 30**2,
    }

RESOLUTIONS = {
        # Microscope resolutions in px/µm
        "10x_UNikon": 1.527,
        "10x_Nikon": 1.541,
        "10x_Zeiss": 1.546,
    }


def read_bboxes(fn):
    """Read bounding boxes from a pickled file.

    The pickled files must contain a dict with ROI names as keys
    and a dict as values. The ROI names are currently ignored.
    A special ROI name is `None`, which must contain the items:
        * 'n_frames': int, number of frames
        * 'width': int, image width in pixels
        * 'height': int, image height in pixels
    All other ROI dicts must have the entries:
        'x_min', 'x_max', 'y_min', 'y_max', 'x_mean', 'y_mean', 'area'
    They all have corresponding int or float values; unit is 'px'.
    """
    with open(fn, 'rb') as f:
        bboxes = pickle.load(f)
    return bboxes


def make_empty_bboxes(coords, area):
    """Create bounding boxes at given positions.

    `coords` is an iterable of coordinates of the bounding box centers.
    The coordinates have the x-coordinate as first and the y-coordinate
    as second argument.
    `area` is the adhesion site area.
    
    Returns a dict as required by the parameter `bboxes` of
    `varying_margins_centered`.
    """
    w = np.rint(np.sqrt(area))
    h = np.rint(area / w)
    w2 = w // 2
    h2 = h // 2
    area = w * h
    
    bboxes = {}
    for i, c in enumerate(coords):
        x1 = np.rint(c[0] - w2)
        y1 = np.rint(c[1] - h2)
        x2 = x1 + w
        y2 = y1 + h
        bboxes[i] = {None: dict(
            x_min=x1,
            x_max=x2,
            x_mean=(x1+x2)/2,
            y_min=y1,
            y_max=y2,
            y_mean=(y1+y2)/2,
            area=area,
        )}
    return bboxes


def insert_bb(stack, bb, bb_width=None, bb_height=None, frame=None, weighted=False, ignore_borders=False):
    """Insert a bounding box into a stack.

    stack -- the numpy stack in which the bounding box is inserted; must have shape (n_frames, 1, 1, height, width)
    bb -- dict of the bounding box; must have entries 'x_min', 'x_max', 'y_min', 'y_max'
    bb_width, bb_height -- int, optional arguments to overwrite bounding box size
    frame -- int, frame of `stack` to insert the bounding box; set `None` for all frames
    weighted -- bool, center new bounding box in old bounding box (False) or at centroid of old ROI (True)
    ignore_borders -- bool, draw (True) or omit (False) ROIs that hit an image border

    Writes 1 if the bounding box fits into the stack and 2 if it is cropped.
    """
    w = bb['x_max'] - bb['x_min'] + 1
    h = bb['y_max'] - bb['y_min'] + 1

    if bb_width is None:
        bb_width = w
    if bb_height is None:
        bb_height = h
    if frame is None or frame is ...:
        frame = slice(None)

    if weighted:
        x1 = np.rint(bb['x_mean']).astype(int) - bb_width // 2
        x2 = x1 + bb_width
        y1 = np.rint(bb['y_mean']).astype(int) - bb_height // 2
        y2 = y1 + bb_height
    else:
        x1 = bb['x_min'] - (bb_width - w) // 2
        x2 = x1 + bb_width
        y1 = bb['y_min'] - (bb_height - h) // 2
        y2 = y1 + bb_height

    height, width = stack.shape[-2:]
    if x1 < 0 or x2 > width or y1 < 0 or y2 > height:
        if isinstance(frame, slice):
            frame_name = ':'
        elif isinstance(frame, type(...)):
            frame_name = '...'
        else:
            frame_name = frame + 1
        if ignore_borders:
            msg = "INCLUDE"
        else:
            msg = "EXCLUDE"
        print(f"{msg} box (x={x1}:{x2}, y={y1}:{y2})[{frame_name}] that hits border (w={bb_width}, h={bb_height})", file=sys.stderr)
        if not ignore_borders:
            return
        if x1 < 0:
            x1 = 0
        if x2 > width:
            x2 = width
        if y1 < 0:
            y1 = 0
        if y2 > height:
            y2 = height
        val = 2
    else:
        val = 1
    stack[frame, ..., y1:y2, x1:x2] = val


def varying_margins_centered(area, *bboxes, margins=(0,), ignore_borders=False):
    """Make binary npz stack with centered ROI and varying margins"""
    n_frames = bboxes[0][None]['n_frames']
    width = bboxes[0][None]['width']
    height = bboxes[0][None]['height']
    stacks = {}
    for margin in margins:
        stacks[margin] = np.zeros((n_frames, height, width), dtype=np.uint8)

    for k, tr in (it for bbx in bboxes if bbx for it in bbx.items()):
        if k is None:
            continue

        for margin, stack in stacks.items():
            x = 1 + margin
            bb_length = np.rint(np.sqrt(x * area)).astype(int)

            for fr, bb in tr.items():
                if fr is ...:
                    continue
                insert_bb(stack, bb, bb_width=bb_length, bb_height=bb_length, frame=fr, weighted=True, ignore_borders=ignore_borders)

    return stacks


def export_squares(stacks, fn, outdir=None, verbose=False):
    """Write squares to npz files.

    stacks -- dict of stacks as generated by `varying margins_centered`
    fn -- str, path of input file, used for generating output file names
    outdir -- str, optional output directory other than input directory
    verbose -- bool, display file names?
    """
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.dirname(fn)
    name = os.path.splitext(os.path.basename(fn))[0]

    for margin, stack in stacks.items():
        stack_name = f"{os.path.join(outdir, name)}_sqcenter_{1+margin :.0%}.npz"
        if verbose:
            print(f"Writing: {stack_name}")
        np.savez_compressed(stack_name, stack)


def make_epilog():
    epilog = []

    epilog.append("Predefined areas:")
    for k, v in AREAS.items():
        epilog.append(f"\t\"{k}\" (={v} µm²)")
    epilog.append("")

    epilog.append("Predefined microscope resolutions:")
    for k, v in RESOLUTIONS.items():
        epilog.append(f"\t\"{k}\" (={v} px/µm)")

    return "\n".join(epilog)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
            description=DOC,
            epilog=make_epilog(),
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('area', metavar="AREA", help="Area of adhesion site (by default in px²) or named area (see below). When specifying the area in µm² or a named area, specify resolution using '-r'.")
    parser.add_argument('path', type=str, nargs='+', metavar="PATH", help="Path(s) of the pickled file(s) to with the cell contours")
    parser.add_argument('-m', '--margin', default='100', help="Relative area of the ROI, inclusive adhesion site and margin, in percent of the adhesion site area. Default is 100 (adhesion area without margin). Multiple areas can be specified as comma-separated list.")
    parser.add_argument('-r', '--resolution', default=None, metavar="RES", help="Resolution of the area in px/µm or named microscope resolution (see below).")
    parser.add_argument('-b', '--ignore-borders', action='store_true', help="Do not exclude ROIs that hit the border.")
    parser.add_argument('-o', '--outdir', default=None, help="Output directory, if result files should not be written to same directory as input file.")
    parser.add_argument('-g', '--glob', action='store_true', help="Treat the given path(s) as Unix filename glob. May be helpful on Windows, where filenames are not expanded.")
    parser.add_argument('-s', '--silent', action='store_true', help="Suppress status output.")
    parser.add_argument('-x', '--empty', action='append', default=[], help="Insert a ROI centered at the given coordinate, e.g. to analyze an empty adhesion site. Specify the center as comma-separated list, e.g. '100,200' for x=100 and y=200 (in pixels). Multiple ROIs can be inserted by separating their coordinates with a semicolon or by specifying this option multiple times.")
    args = parser.parse_args()

    argdict = {}

    if args.glob:
        import glob
        argdict['path'] = [g for p in args.path for g in glob.iglob(p)]
    else:
        argdict['path'] = args.path

    argdict['margin'] = [float(x)/100-1 for x in args.margin.split(',') if x]

    area_invalid = False
    resolution_required = False
    try:
        area = float(args.area)
    except (ValueError, TypeError):
        resolution_required = True
        try:
            area = AREAS[args.area]
        except KeyError:
            area_invalid = True
    if area_invalid:
        raise ValueError(f"Invalid area '{args.area}'. Area must be a numeric value or a name of a predefined area.")
    if args.resolution is not None or resolution_required:
        res_invalid = False
        try:
            res = float(args.resolution)
        except (ValueError, TypeError):
            try:
                res = RESOLUTIONS[args.resolution]
            except KeyError:
                res_invalid = True
        if res_invalid:
            if resolution_required and args.resolution is None:
                raise ValueError("No resolution given. Resolution must be specified when specifying a named area.")
            else:
                raise ValueError("Invalid resolution '{args.resolution}'. Resolution must be a numeric value or a name of a predefined resolution.")
        else:
            area *= res**2
    argdict['area'] = area
    
    argdict['empty'] = []
    for coord in (c for C in args.empty for c in C.split(';') if c):
        cc = []
        for csp in coord.split(','):
            try:
                cc.append(int(csp.strip()))
            except Exception:
                cc = None
                break
        else:
            if len(cc) == 2:
                argdict['empty'].append(cc)
            else:
                cc = None
        if not cc:
            raise ValueError(f"Invalid coordinate '{coord}'. Coordinates must consist of two integers separated by a comma.")

    argdict['ignore_borders'] = args.ignore_borders
    argdict['outdir'] = args.outdir
    argdict['verbose'] = not args.silent

    return argdict


def main(path, area, margin, outdir=None, verbose=True, ignore_borders=False, empty=None):
    for p in path:
        bboxes = read_bboxes(p)
        if empty:
            empty = make_empty_bboxes(empty, area)
        stacks = varying_margins_centered(area, bboxes, empty, margins=margin, ignore_borders=ignore_borders)
        export_squares(stacks, p, outdir=outdir, verbose=verbose)


if __name__ == '__main__':
    main(**parse_args())
