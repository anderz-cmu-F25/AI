# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    head, tail = alien.get_head_and_tail()
    wid = alien.get_width()

    # Not ball
    if head != tail:
        magnitude = ((head[1] - tail[1])**2 + (head[0] - tail[0])**2)**0.5
        assert magnitude != 0, "line 38"
        tail_left = -(head[1] - tail[1])/magnitude*wid + tail[0], (head[0] - tail[0])/magnitude*wid + tail[1]
        tail_right = (head[1] - tail[1])/magnitude*wid + tail[0], -(head[0] - tail[0])/magnitude*wid + tail[1]
        head_left = -(head[1] - tail[1])/magnitude*wid + head[0], (head[0] - tail[0])/magnitude*wid + head[1]
        head_right = (head[1] - tail[1])/magnitude*wid + head[0], -(head[0] - tail[0])/magnitude*wid + head[1]

        for i in walls:
            wall = ((i[0], i[1]), (i[2], i[3]))
            if do_segments_intersect((tail_left, tail_right), wall) or do_segments_intersect((tail_left, head_left), wall) or \
            do_segments_intersect((head_right, tail_right), wall) or do_segments_intersect((head_left, head_right), wall) or \
            point_segment_distance(head, wall) <= wid or point_segment_distance(tail, wall) <= wid or \
            is_point_in_polygon(wall[0], (tail_left, head_left, head_right, tail_right)) or \
            is_point_in_polygon(wall[1], (tail_left, head_left, head_right, tail_right)):
                return True
    else:
        for i in walls:
            wall = ((i[0], i[1]), (i[2], i[3]))
            if point_segment_distance(head, wall) <= wid:
                return True

    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    x, y = alien.get_centroid()
    shape = alien.get_shape()
    len = alien.get_length() / 2
    wid = alien.get_width()
    lr, ud = wid, len + wid
    if shape == "Horizontal":
        lr, ud = len + wid, wid
    elif shape == "Ball":
        lr = ud = wid
    left = x - lr
    right = x + lr
    up = y - ud
    down = y + ud
    # print(f"{shape} pos: {x} {y} len {len} wid {wid} {left, right, up, down} {window}")
    return left > 0 and right < window[0] and up > 0 and down < window[1]


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    a, b, c, d = polygon
    ab = vectorize(a, b)
    ap = vectorize(a, point)
    dc = vectorize(d, c)
    dp = vectorize(d, point)
    ad = vectorize(a, d)
    bc = vectorize(b, c)
    bp = vectorize(b, point)
    one_one = cross_product(ab, ap)
    one_two = cross_product(dc, dp)
    two_one = cross_product(ad, ap)
    two_two = cross_product(bc, bp)
    if cross_product(ab, ad) == 0 and one_one == 0:
        return point[0] >= min(a[0], b[0], c[0], d[0]) and point[0] <= max(a[0], b[0], c[0], d[0]) and \
               point[1] >= min(a[1], b[1], c[1], d[1]) and point[1] <= max(a[1], b[1], c[1], d[1])
    return (one_one == 0 or one_two == 0 or ((one_one > 0) != (one_two > 0))) and (two_one == 0 or two_two == 0 or ((two_one > 0) != (two_two > 0)))


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    wid = alien.get_width()
    shape = alien.get_shape()
    pos = tuple(alien.get_centroid())

    if shape == "Ball":
        for i in walls:
            wall = (i[0], i[1]), (i[2], i[3])
            if segment_distance((pos, waypoint), wall) <= wid:
                return True
    else:
        head, tail = alien.get_head_and_tail()
        len = alien.get_length() / 2
        if shape == "Horizontal":
            dst_head, dst_tail = (waypoint[0] + len, waypoint[1]), (waypoint[0] - len, waypoint[1])
        elif shape == "Vertical":
            dst_head, dst_tail = (waypoint[0], waypoint[1] - len), (waypoint[0], waypoint[1] + len)
            
        for i in walls:
            wall = (i[0], i[1]), (i[2], i[3])
            if segment_distance((head, tail), wall) <= wid or segment_distance((dst_head, dst_tail), wall) <= wid or \
            segment_distance((head, dst_head), wall) <= wid or segment_distance((tail, dst_tail), wall) <= wid or \
            is_point_in_polygon(wall[0], (head, dst_head, dst_tail, tail)) or \
            is_point_in_polygon(wall[1], (head, dst_head, dst_tail, tail)):
                return True
            
    return False


def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    a, b = s

    if a == b:
        return euc_dist_sq(p, a)**0.5

    ap = (p[0] - a[0], p[1] - a[1])
    ab = (b[0] - a[0], b[1] - a[1])
    mag_ab_sq = euc_dist_sq(a, b)
    
    assert a != b, "right?"

    norm_comp = (ap[0]*ab[0] + ap[1]*ab[1]) / mag_ab_sq

    if norm_comp > 1:
        return euc_dist_sq(b, p)**0.5
    elif norm_comp < 0:
        return euc_dist_sq(a, p)**0.5
    else:
        q_0 = tuple(norm_comp * i for i in ab)
        q = (q_0[0] + a[0], q_0[1] + a[1])
        return euc_dist_sq(q, p)**0.5


def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    a, b = s1
    p, q = s2
    ab = (b[0] - a[0], b[1] - a[1])
    ap = (p[0] - a[0], p[1] - a[1])
    aq = (q[0] - a[0], q[1] - a[1])
    pq = (q[0] - p[0], q[1] - p[1])
    pa = (a[0] - p[0], a[1] - p[1])
    pb = (b[0] - p[0], b[1] - p[1])

    res = points_opposite(ab, ap, aq)

    if res == -1:
        x = (a[0] >= min(p[0], q[0]) and a[0] <= max(p[0], q[0])) or (b[0] >= min(p[0], q[0]) and b[0] <= max(p[0], q[0])) or (p[0] >= min(a[0], b[0]) and p[0] <= max(a[0], b[0]))
        y = (a[1] >= min(p[1], q[1]) and a[1] <= max(p[1], q[1])) or (b[1] >= min(p[1], q[1]) and b[1] <= max(p[1], q[1])) or (p[1] >= min(a[1], b[1]) and p[1] <= max(a[1], b[1]))
        return x and y
    else:
        return res and points_opposite(pq, pa, pb)


def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(s1, s2):
        return 0
    else:
        a, b = s1
        p, q = s2
        d1 = point_segment_distance(a, s2)
        d2 = point_segment_distance(b, s2)
        d3 = point_segment_distance(p, s1)
        d4 = point_segment_distance(q, s1)
        return min(d1, d2, d3, d4)

def euc_dist_sq(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (x2 - x1)**2 + (y2 - y1)**2

def points_opposite(base, target1, target2):
    sign1 = base[0]*target1[1] - base[1]*target1[0]
    sign2 = base[0]*target2[1] - base[1]*target2[0]

    if sign1 == 0 and sign2 == 0:
        return -1
    else:
        return sign1 == 0 or sign2 == 0 or (sign1 > 0 and sign2 < 0) or (sign1 < 0 and sign2 > 0)
    
def cross_product(v1, v2):
    return v1[0]*v2[1] - v1[1]*v2[0]
    
def vectorize(a, b):
    return (b[0] - a[0], b[1] - a[1])


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()
        wid = alien.get_width()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} and width {wid} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
