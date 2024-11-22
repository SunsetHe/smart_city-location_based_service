import pdb
from math import radians, cos, sin, asin, sqrt, atan, pi, log, tan, exp, atan2, degrees, fabs
import numpy as np
from scipy import interpolate
from haversine import haversine, Unit
from pyproj import Transformer






def utm_to_wgs84(lng, lat):
    # WGS84 = Proj(init='EPSG:4326')
    # p = Proj(init="EPSG:32650")
    # x,y = lng, lat
    # return transform(p, WGS84, x, y)
    transformer = Transformer.from_crs("epsg:32650", "epsg:4326")
    a = transformer.transform(lng, lat)
    print(a)
    return [a[1], a[0]]

# coordinate tool
def _transform_lat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * sqrt(fabs(lng))
    ret += (20.0 * sin(6.0 * lng * pi) + 20.0 * sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * sin(lat * pi) + 40.0 * sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * sin(lat / 12.0 * pi) + 320 * sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * sqrt(fabs(lng))
    ret += (20.0 * sin(6.0 * lng * pi) + 20.0 * sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * sin(lng * pi) + 40.0 * sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * sin(lng / 12.0 * pi) + 300.0 * sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def out_of_china(lng, lat):
    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    return not (73.66 < lng < 135.05 and 3.86 < lat < 53.55)


def wgs84_to_gcj02(lng, lat):
    """
    WGS84转GCJ02(火星坐标系)
    :param lng:WGS84坐标系的经度
    :param lat:WGS84坐标系的纬度
    :return:
    """
    if out_of_china(lng, lat):  # 判断是否在国内
        return [lng, lat]
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = sin(radlat)
    magic = 1 - 0.00669342162296594323 * magic * magic
    sqrtmagic = sqrt(magic)
    dlat = (dlat * 180.0) / ((6378245.0 * (1 - 0.00669342162296594323)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (6378245.0 / sqrtmagic * cos(radlat) * pi)
    return [lng + dlng, lat + dlat]


def gcj02_to_wgs84(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    if out_of_china(lng, lat):
        return [lng, lat]
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = 1 - 0.00669342162296594323 * sin(radlat) * sin(radlat)
    sqrtmagic = sqrt(magic)
    dlat = (dlat * 180.0) / ((6378245.0 * (1 - 0.00669342162296594323)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (6378245.0 / sqrtmagic * cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def wgs84_to_mercator(lon, lat):
    x = lon * 20037508.342789 / 180
    y = log(tan((90 + lat) * pi / 360)) / (pi / 180)
    y = y * 20037508.34789 / 180
    return [x, y]


def mercator_to_wgs84(x, y):
    lon = x / 20037508.34 * 180
    lat = y / 20037508.34 * 180
    lat = 180 / pi * (2 * atan(exp(lat * pi / 180)) - pi / 2)
    return [lon, lat]


def transform_points_wgs84_to_mercator(coordinates):
    temp_result = []
    for i, item in enumerate(coordinates):
        temp_result.append(wgs84_to_mercator(item[0], item[1]))
    return temp_result


def transform_points_mercator_to_wgs84(coordinates):
    temp_result = []
    for i, item in enumerate(coordinates):
        temp_result.append(mercator_to_wgs84(item[0], item[1]))
    return temp_result


def transform_points_wgs84_to_gcj02(coordinates):
    temp_result = []
    for i, item in enumerate(coordinates):
        temp_result.append(wgs84_to_gcj02(item[0], item[1]))
    return temp_result


def transform_points_gcj02_to_wgs84(coordinates):
    temp_result = []
    for i, item in enumerate(coordinates):
        temp_result.append(gcj02_to_wgs84(item[0], item[1]))
    return temp_result

import math
# angel tool    参考三面角余弦定理，书签博客
def get_angle_to_north(x_point_s, y_point_s, x_point_e, y_point_e):
    # lat1_radians = radians(lat1)    # radians 角度转弧度
    # lon1_radians = radians(lon1)
    # lat2_radians = radians(lat2)
    # lon2_radians = radians(lon2)
    # lon_difference = lon2_radians - lon1_radians
    # y = sin(lon_difference) * cos(lat2_radians)
    # x = cos(lat1_radians) * sin(lat2_radians) - sin(lat1_radians) * cos(lat2_radians) * cos(lon_difference)
    # return (degrees(atan2(y, x)) + 360) % 360   # atan2 反正切   degrees 弧度转角度

    angle = 0
    y_se = y_point_e - y_point_s
    x_se = x_point_e - x_point_s
    if x_se == 0 and y_se > 0:
        angle = 360
    if x_se == 0 and y_se < 0:
        angle = 180
    if y_se == 0 and x_se > 0:
        angle = 90
    if y_se == 0 and x_se < 0:
        angle = 270
    if x_se > 0 and y_se > 0:
        angle = math.atan(x_se / y_se) * 180 / pi
    elif x_se < 0 and y_se > 0:
        angle = 360 + math.atan(x_se / y_se) * 180 / pi
    elif x_se < 0 and y_se < 0:
        angle = 180 + math.atan(x_se / y_se) * 180 / pi
    elif x_se > 0 and y_se < 0:
        angle = 180 + math.atan(x_se / y_se) * 180 / pi
    return angle



def calculate_angle_diff(angel_diff):
    abs_angel_diff = fabs(angel_diff)
    if abs_angel_diff > 180:
        return 360 - abs_angel_diff
    else:
        return abs_angel_diff


def get_segment_dir_change(segment1, segment2):
    dir_segment1 = get_angle_to_north(segment1[0], segment1[1], segment1[2], segment1[3])
    dir_segment2 = get_angle_to_north(segment2[0], segment2[1], segment2[2], segment2[3])
    return calculate_angle_diff(dir_segment1 - dir_segment2)

def get_cos_value(net, link):
    vector_a = np.mat(net)
    vector_b = np.mat(link)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    return num / denom


def detect_u_turn(shape, u_turn_angle=130):
    start_angle = get_angle_to_north(shape[0][0], shape[0][1], shape[1][0], shape[1][1])
    end_angle = get_angle_to_north(shape[-2][0], shape[-2][1], shape[-1][0], shape[-1][1])
    return calculate_angle_diff(start_angle - end_angle) > u_turn_angle


def projection_direction(direction, para_length=100):
    return [sin(radians(direction)) * para_length,
            cos(radians(direction)) * para_length]

def projection_direction_new(direction):
    para_length = 100.0
    if 0 <= direction < 90:
        return [sin(radians(direction)) * para_length,
                cos(radians(direction)) * para_length]
    if 90 <= direction < 180:
        return [sin(radians(direction)) * para_length,
                -cos(radians(direction)) * para_length]
    if 180 <= direction < 270:
        return [-sin(radians(direction)) * para_length,
                -cos(radians(direction)) * para_length]
    if 270 <= direction <= 360:
        return [-sin(radians(direction)) * para_length,
                cos(radians(direction)) * para_length]

# distance tool
def eucl_distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    # c = 2 * atan2(sqrt(a), sqrt(1-a))
    r = 6371
    return c * r * 1000


def judge_frechet(ca, i, j, p, q):
    """
    :param ca: The initial distance matrix
    :param i: traj_q list final index index
    :param j: traj_p list final index index
    :param p: traj_p mercator list
    :param q: traj_q mercator list
    :return:
    """
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = eucl_distance(p[0], q[0])
    elif i > 0 and j == 0:
        ca[i, j] = np.max((judge_frechet(ca, i - 1, 0, p, q), eucl_distance(p[i], q[0])))
    elif i == 0 and j > 0:
        ca[i, j] = np.max((judge_frechet(ca, 0, j - 1, p, q), eucl_distance(p[0], q[j])))
    elif i > 0 and j > 0:
        ca[i, j] = np.max((np.min((judge_frechet(ca, i - 1, j, p, q), judge_frechet(ca, i - 1, j - 1, p, q),
                                   judge_frechet(ca, i, j - 1, p, q))), eucl_distance(p[i], q[j])))
    else:
        ca[i, j] = float("inf")
    return ca[i, j]


def frechet_distance(traj_p, traj_q):
    return judge_frechet(np.multiply(np.ones((len(traj_p), len(traj_q))), -1), len(traj_p) - 1, len(traj_q) - 1, traj_p, traj_q)


# common tool
def get_by_index(pre, after_index):
    if len(pre) == len(after_index):
        return pre
    return [pre[e] for e in after_index]


def duplicate_removal(data):
    new_data = list()
    tmp_data = set()
    for item in data:
        tmp = tuple(item)
        if tmp not in tmp_data:
            new_data.append(item)
            tmp_data.add(tmp)
    return new_data


def filter_adjacent_redundant_index(coordinates):
    res_index = []
    pre = [coordinates[0]] + coordinates[:-1]
    for i, item in enumerate(coordinates):
        if item != pre[i]:
            res_index.append(i - 1)
    res_index.append(len(coordinates) - 1)
    return res_index


def get_velocity(coordinates, dt):
    dt = get_time_diff(dt)  # 把时间戳转换为间隔 list
    velocity_list = list()
    coordinate_compare = coordinates[1: ] + [coordinates[-1]]
    for i, item in enumerate(coordinates):
        temp_v = 0.0
        if dt[i] != 0:
            dis = eucl_distance([coordinate_compare[i][0], coordinate_compare[i][1]], [item[0], item[1]])
            temp_v = dis/dt[i]
        velocity_list.append(temp_v)
    return velocity_list

def get_major_point_index(directions):
    directions_cmp = [directions[0]] + directions[:-1]
    red_index = []
    for i, item in enumerate(directions):
        if i == len(directions) - 1:
            continue
        if abs(item - directions_cmp[i]) < 15 and abs(directions[i + 1] - item) < 15:
            red_index.append(i)
    return red_index


def get_direction(coordinates):
    directions = list()
    for i, item in enumerate(coordinates[:-1]):
        directions.append(get_angle_to_north(item[0], item[1], coordinates[i+1][0], coordinates[i+1][1]))
    directions.append(directions[-1])
    return directions


def get_distance_diff(coordinates_mercator):
    temp_result = []
    coordinates_mercator_pre = [coordinates_mercator[0]] + coordinates_mercator[:-1]
    for i, item in enumerate(coordinates_mercator):
        dis_temp = eucl_distance(np.array(item), np.array(coordinates_mercator_pre[i]))
        temp_result.append(dis_temp)
    return temp_result


def get_time_diff(times):
    time_diff = []
    times_pro = times[1: ] + [times[-1]]
    for i, item in enumerate(times):
        time_diff.append(times_pro[i] - item)
    return time_diff


def get_point_in_range(min_lon, min_lat, max_lon, max_lat, points):
    temp_result = []
    for i, item in enumerate(points):
        if min_lon <= item[0] <= max_lon and min_lat <= item[1] <= max_lat:
            temp_result.append(item)
    return temp_result


def get_b_spline(traj):  # 通过多项式使用差值法还原轨迹线段的曲线特征
    res = []
    len_ = len(traj)
    if len_ <= 4:
        for i, item in enumerate(traj):
            res.append([float(item[0]), float(item[1])])
    else:
        lon_list = []
        lat_list = []
        for e in traj:
            lon_list.append(e[0])
            lat_list.append(e[1])
        x_list = np.array(lon_list)
        y_list = np.array(lat_list)
        t = np.linspace(0, 1, len_ - 2, endpoint=True)
        t = np.append([0, 0, 0], t)
        t = np.append(t, [1, 1, 1])
        tck = [t, [x_list, y_list], 3]
        u3 = np.linspace(0, 1, 100, endpoint=True)
        out = interpolate.splev(u3, tck)
        lon_result = list(out[0])
        lat_result = list(out[1])
        for i, item in enumerate(lon_result):
            res.append([float(item), float(lat_result[i])])
    return res

def judge_coor_in_bounding(coor, bounding):
    lon, lat = coor
    if (bounding[0] < lon < bounding[2]) and (bounding[-1] < lat < bounding[1]):
        return 1
    return 0

def lcss(t0, t1, eps):
    """
    Usage
    -----
    The Longuest-Common-Subsequence distance between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float

    Returns
    -------
    lcss : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if eucl_distance(t0[i - 1], t1[j - 1]) < eps:
                C[i][j] = C[i - 1][j - 1] + 1
            else:
                C[i][j] = max(C[i][j - 1], C[i - 1][j])
    lcss = 1 - float(C[n0][n1]) / min([n0, n1])
    return lcss

def Distance(PointA, PointB, TypeC=0):
    # 精确的基于轨迹坐标的距离函数
    if TypeC == 0:
        latdis = fabs(PointA[1] - PointB[1]) * 111319.488
        lngdistemp = fabs(PointA[0] - PointB[0]) * 111319.488 * 0.5
        lngdis = lngdistemp * (cos(PointA[1] * pi/180.0) + cos(PointB[1] * pi/180.0))
        return sqrt(latdis ** 2 + lngdis ** 2)
    # haversine距离
    if TypeC == 1:
        return haversine(PointA, PointB, unit=Unit.METERS)
    # 墨卡托投影系下的欧式距离
    if TypeC == 2:
        pointa = wgs84_to_mercator(PointA[0], PointA[1])
        pointb = wgs84_to_mercator(PointB[0], PointB[1])
        return sqrt((pointa[0] - pointb[0])**2 + (pointa[1] - pointb[1])** 2)