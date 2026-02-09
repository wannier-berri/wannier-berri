import numpy as np


def flatten_path(nodes, segments, direction=None):
    """Flattens a path defined by nodes and segments. If direction is given, then the path is flattened along this direction, otherwise the path is flattened along the direction of the first segment"""
    if direction is None:
        return nodes, segments
    print(f"Flattening path along direction {direction}")
    print(f"Original nodes: {nodes}")
    print(f"Original segments: {segments}")
    nodes_flat = {k: np.array(v) for k, v in nodes.items() if abs(v[direction]) < 1e-7}
    flatten_map = {}
    for k, v in nodes.items():
        for kf, vf in nodes_flat.items():
            diff = v - vf
            diff[direction] = 0
            if np.linalg.norm(diff) < 1e-7:
                flatten_map[k] = kf
                break
    print(f"Flatten map: {flatten_map}")
    segments_flat = [(flatten_map[s[0]], flatten_map[s[1]]) for s in segments]
    print(f"Segments after flattening: {segments_flat}")
    segments_flat = [seg for seg in segments_flat if seg[0] != seg[1]]  # remove vertical lines
    print(f"Segments after removing vertical lines: {segments_flat}")
    repeated = np.zeros(len(segments_flat), dtype=bool)
    for i, seg in enumerate(segments_flat):
        for j in range(i):
            if not repeated[j]:
                seg2 = segments_flat[j]
                if (seg == seg2) or (seg == (seg2[1], seg2[0])):
                    repeated[i] = True
                    break
    segments_flat = [seg for i, seg in enumerate(segments_flat) if not repeated[i]]  # remove repeated lines
    segments_flat = connect_segments(segments_flat)
    return nodes_flat, segments_flat



def connect_segments(segments):
    """Connects segments into a path. 

    Parameters
    ----------
    segments : list of tuples
         list of segments defined by tuples of node indices (start, end). start and end may be any hashable objects, but they should be consistent across segments. 

    Returns
    -------
    segments : list of tuples
         list of segments connected into a path. The order of segments is changed and some of them are reversed so that the end of one segment is the start of the next one. 

    Notes
    -----
    The segments that cannot be connected are left unchanged in the end of the list. The segments that are connected are moved to the beginning of the list.
    """
    if len(segments) == 0:
        return segments
    segment_loops = []
    segments_taken = np.zeros(len(segments), dtype=bool)
    while not np.all(segments_taken):
        i = np.where(~segments_taken)[0][0]
        loop = [segments[i][0], segments[i][1]]
        segments_taken[i] = True
        loop_closed = False
        # connect to end of loop whatever possible
        while True:
            success = False
            for j, seg in enumerate(segments):
                if not segments_taken[j]:
                    if loop[-1] in seg or loop[0] in seg:
                        success = True
                        segments_taken[j] = True
                        if loop[-1] in seg:
                            print(f"Segment {seg} can be connected to the end of the loop {loop}")
                            if seg[0] == loop[-1]:
                                loop.append(seg[1])
                            else:
                                loop.append(seg[0])  # reverse the segment
                        elif loop[0] in seg:
                            print(f"Segment {seg} can be connected to the beginning of the loop {loop}")
                            if seg[0] == loop[0]:
                                loop.insert(0, seg[1])  # reverse the segment
                            else:
                                loop.insert(0, seg[0])  # reverse the segment
                    else:
                        print(f"Segment {seg} cannot be connected to the loop {loop}")
                    if loop[0] == loop[-1]:  # check if the loop is closed or cannot be extended
                        loop_closed = True
                        print(f"Loop closed: {loop}")
                        break  # loop while
            if not success or loop_closed:
                print(f"Loop cannot be extended: {loop}, segments left: {[s for i, s in enumerate(segments) if not segments_taken[i]]}")
                break
        segment_loops.append(loop)
    print(f"Segment loops: {segment_loops}")
    segment_loops = insert_all_closed_loops(segment_loops)
    print(f"Segment loops after inserting closed loops: {segment_loops}")
    segment_loops = insert_all_unclosed_to_closed(segment_loops)
    print(f"Segment loops after inserting unclosed loops into closed loops: {segment_loops}")
    return [(s1, s2) for loop in segment_loops for s1, s2 in zip(loop[:-1], loop[1:])]



def insert_closed_loop(loops):
    """Tries to insert a closed loop into another loop.

    Parameters
    ----------
    loops : list of lists
        list of loops defined by lists of node indices (start, ..., end). start and end may be any hashable objects, but they should be consistent across loops. 
    closed_loops : list of bool
        list of booleans indicating whether the corresponding loop is closed or not. The length of this list should be equal to the length of the loops list.

    Returns
    -------
    loops : list of lists
        the modified list of loops.
    success : bool
        whether the insertion was successful or not. 
    """
    for i, loop1 in enumerate(loops):  # the loop which is to be inserted
        if loop1[0] != loop1[-1]:  # if the loop is not closed
            continue
        for j, loop2 in enumerate(loops):  # the loop to be inserted into
            if i == j:
                continue
            for k, point in enumerate(loop1):
                if point in loop2:
                    loop1roll = loop1[k:-1] + loop1[:k]  # roll the loop so that the common point is at the beginning, but do not include last point
                    idx = loop2.index(point)
                    loop2_modified = loop2[:idx] + loop1roll + loop2[idx:]  # insert the loop1 into loop2
                    # print (f"{loop1=} is inserted into {loop2=} at point {idx=} {point=},\n{k=}, {loop1roll=} \nresulting in {loop2_modified=}")
                    loops_new = []
                    for l, loop in enumerate(loops):
                        if l == i:
                            continue
                        elif l == j:
                            loops_new.append(loop2_modified)
                        else:
                            loops_new.append(loop)
                    return loops_new, True
    return loops, False


def insert_all_closed_loops(loops):
    """Tries to insert all closed loops into other loops. The order of loops is changed in the process. The closed loops that cannot be inserted are left unchanged at the end of the list.

    Parameters
    ----------
    loops : list of lists
        list of loops defined by lists of node indices (start, ..., end). start and end may be any hashable objects, but they should be consistent across loops. 

    Returns
    -------
    loops : list of lists
        the modified list of loops.
    """
    while True:
        loops, success = insert_closed_loop(loops)
        if not success:
            break
    return loops


def insert_unclosed_to_closed(loops):
    """Tries to insert an unclosed loop into a closed loop. The order of loops is changed in the process. The unclosed loops that cannot be inserted are left unchanged at the end of the list.

    Parameters
    ----------
    loops : list of lists
        list of loops defined by lists of node indices (start, ..., end). start and end may be any hashable objects, but they should be consistent across loops. 

    Returns
    -------
    loops : list of lists
        the modified list of loops.
    success : bool
        whether the insertion was successful or not.
    """
    for i, loop1 in enumerate(loops):  # the loop which is to be inserted
        if loop1[0] == loop1[-1]:  # if the loop is closed
            continue
        for j, loop2 in enumerate(loops):  # the loop to be inserted into
            if i == j:
                continue
            if loop2[0] != loop2[-1]:  # if the loop is not closed
                continue
            for k, point in enumerate(loop2):
                if loop1[0] == point or loop1[-1] == point:
                    loop2roll = loop2[k:-1] + loop2[:k + 1]  # roll the loop so that the common point is at the beginning
                    if loop1[0] == point:
                        loop2_modified = loop1[-1::-1] + loop2roll[1:]  # insert the reversed loop1 into loop2
                    else:
                        loop2_modified = loop1 + loop2roll[1:]  # insert the loop1 into loop2
                    return [loop2_modified if l == j else loop for l, loop in enumerate(loops) if l != i], True
    return loops, False


def insert_all_unclosed_to_closed(loops):
    """Tries to insert all unclosed loops into closed loops. The order of loops is changed in the process. The unclosed loops that cannot be inserted are left unchanged at the end of the list.

    Parameters
    ----------
    loops : list of lists
        list of loops defined by lists of node indices (start, ..., end). start and end may be any hashable objects, but they should be consistent across loops. 

    Returns
    -------
    loops : list of lists
        the modified list of loops.
    """
    while True:
        loops, success = insert_unclosed_to_closed(loops)
        if not success:
            return loops
