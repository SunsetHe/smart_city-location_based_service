import os
import json

import utils.util as util
import numpy as np


def geo_json_generate_points(link_wkts, type_style="Multipoint"):
    res = {
        "type": "FeatureCollection",
        "features": []
    }

    for i, link_info in enumerate(link_wkts):
        t = {
            "type": "Feature",
            "geometry": {
                "type": type_style, "coordinates": util.transform_points_mercator_to_wgs84(link_info) if link_info[0][0] > 1000 else link_info
            },
            "properties": {
                'size': len(link_info),
                'order': i
            }

        }

        res["features"].append(t)

    return res


def geo_json_cluster_points(link_wkts, type_style="Multipoint"):
    res = {
        "type": "FeatureCollection",
        "features": []
    }

    for i, link_info in link_wkts.items():
        t = {
            "type": "Feature",
            "geometry": {
                "type": type_style,
                "coordinates": util.transform_points_mercator_to_wgs84(link_info) if link_info[0][
                                                                                               0] > 1000 else link_info
            },
            "properties": {
                'cluster_id': float(i)
            }

        }

        res["features"].append(t)

    return res


def geo_json_cluster_center(link_wkts, type_style="Point"):
    res = {
        "type": "FeatureCollection",
        "features": []
    }

    for i, link_info in link_wkts.items():
        t = {
            "type": "Feature",
            "geometry": {
                "type": type_style, "coordinates": np.mean(np.array(
                    util.transform_points_mercator_to_wgs84(link_info) if link_info[0][1] > 1000 else link_info),
                                                           axis=0).tolist()
            },
            "properties": {
                'cluster_id': float(i)
            }

        }

        res["features"].append(t)

    return res


def geo_json_points(link_wkts, type_style="Point"):
    res = {
        "type": "FeatureCollection",
        "features": []
    }

    for i, link_info in enumerate(link_wkts):
        t = {
            "type": "Feature",
            "geometry": {
                "type": type_style, "coordinates": link_info[0]
            },
            "properties": {
                'point_id': float(i),
                'rang': float(link_info[1]),
                'count_cons': float(link_info[2]),
                'count_stay': float(link_info[3]),
                'count_all': float(link_info[4]),
                '5_ave_speed': float(link_info[5]),
                '5_count_all': float(link_info[6]),
                '10_ave_speed': float(link_info[7]),
                '10_count_all': float(link_info[8]),
                '15_ave_speed': float(link_info[9]),
                '15_count_all': float(link_info[10]),
                '20_ave_speed': float(link_info[11]),
                '20_count_all': float(link_info[12]),
                '25_ave_speed': float(link_info[13]),
                '25_count_all': float(link_info[14]),
                '30_ave_speed': float(link_info[15]),
                '30_count_all': float(link_info[16]),
                '35_ave_speed': float(link_info[17]),
                '35_count_all': float(link_info[18]),
                '40_ave_speed': float(link_info[19]),
                '40_count_all': float(link_info[20]),
                '45_ave_speed': float(link_info[21]),
                '45_count_all': float(link_info[22]),
                '50_ave_speed': float(link_info[23]),
                '50_count_all': float(link_info[24]),
                '55_ave_speed': float(link_info[25]),
                '55_count_all': float(link_info[26]),
                '60_ave_speed': float(link_info[27]),
                '60_count_all': float(link_info[28]),
            }

        }

        res["features"].append(t)

    return res


def geo_json_generate_with_id(link_wkts, type_style="MultiLineString"):
    res = {
        "type": "FeatureCollection",
        "features": []
    }

    for i, link_info in enumerate(link_wkts):
        t = {
            "type": "Feature",
            "geometry": {
                "type": type_style, "coordinates": link_info
            },
            "properties": {
                "cluster_id": i
            }

        }

        res["features"].append(t)

    return res


def geo_multi_lines(link_wkts, type_style="Polygon"):
    res = {
        "type": "FeatureCollection",
        "features": []
    }
    # point_prop, ave_speed, std_dir, ave_speed_change, acc_speed_prop, low_speed_prop, speed_change_prop, ave_dir_change, weight_direction_change
    for i, link_info in enumerate(link_wkts):
        coors = link_info[0]

        t = {
            "type": "Feature",
            "geometry": {
                "type": type_style, "coordinates": [[i[0] for i in coors]]
            },
            "properties": {
                "cluster_id": i,
                "points_count": link_info[1][0],
                'point_prop': link_info[1][1],
                'ave_speed': link_info[1][2],
                'std_dir': str(link_info[1][3][0]) + ',' + str(link_info[1][3][1]),
                'ave_speed_change': link_info[1][4],
                'acc_speed_prop': link_info[1][5],
                'low_speed_prop': link_info[1][6],
                'speed_change_prop': link_info[1][7],
                'ave_dir_change_prop': link_info[1][8],
                'weight_dir_change': link_info[1][9],
                'turning_counts': link_info[1][10],
                'turning_nei_prop': link_info[1][11]
            }

        }

        res["features"].append(t)

    return res


def geo_json_generate_traj_with_id(link_wkts, type_style="LineString"):
    res = {
        "type": "FeatureCollection",
        "features": []
    }
    for item in link_wkts:
        for traj_id, traj in item[1].items():
            t = {
                "type": "Feature",
                "geometry": {
                    "type": type_style, "coordinates": traj
                },
                "properties": {
                    "order_id": traj_id,
                    "cluster_id": item[0]
                }

            }

            res["features"].append(t)

    return res


def geo_json_generate_traj_from_dict(link_wkts, type_style="LineString"):
    res = {
        "type": "FeatureCollection",
        "features": []
    }
    for k, v in link_wkts.items():
        t = {
            "type": "Feature",
            "geometry": {
                "type": type_style,
                "coordinates": util.transform_points_mercator_to_wgs84(v[1]) if v[1][0][0] > 1000 else v[1]
            },
            "properties": {
                "order_id": k,
                # /
            }

        }

        res["features"].append(t)

    return res

def geo_json_cluster_center_tkdd(link_wkts, type_style="Point"):
    res = {
        "type": "FeatureCollection",
        "features": []
    }

    for i, link_info in link_wkts.items():
        t = {
            "type": "Feature",
            "geometry": {
                "type": type_style, "coordinates": np.mean(np.array(
                    util.transform_points_mercator_to_wgs84([t[: 2] for t in link_info]) if link_info[0][1] > 1000 else link_info),
                                                           axis=0).tolist()
            },
            "properties": {
                'cluster_id': float(i)
            }

        }

        res["features"].append(t)

    return res

def geo_json_generate_traj_from_list(link_wkts, type_style="LineString"):
    res = {
        "type": "FeatureCollection",
        "features": []
    }
    for k, v in enumerate(link_wkts):
        t = {
            "type": "Feature",
            "geometry": {
                "type": type_style, "coordinates": v
            },
            "properties": {
                "order_id": k,
                "inter_count": str(len(v))
                # /
            }

        }

        res["features"].append(t)

    return res


def geo_json_generate_polygon_from_list(link_wkts, type_style="Polygon"):
    res = {
        "type": "FeatureCollection",
        "features": []
    }
    for k, v in enumerate(link_wkts):
        t = {
            "type": "Feature",
            "geometry": {
                "type": type_style, "coordinates": [v]
            },
            "properties": {
                "order_id": k,
                # "coor": f'{str(v[1][0])} {str(v[1][1])}',
                # "score": float(v[2]) if len(v) > 2 else 'None'
                # /
            }

        }

        res["features"].append(t)

    return res


def visual_result(traj, path, type_style="MultiLineString"):
    """
    Args:
        list traj
        String path
        String type_style(Multipoint, LineString, Point)
    Returns:
        .json
    """
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        json.dump(geo_json_generate_with_id(traj, type_style), f)


def visual_raw_traj(traj, path, type_style="LineString"):
    """
    Args:
        list traj
        String path
        String type_style(Multipoint, LineString, Point)
    Returns:
        .json
    """
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        json.dump(geo_json_generate_traj_from_dict(traj, type_style), f)


def visual_traj(traj, path, type_style="LineString"):
    """
    Args:
        list traj
        String path
        String type_style(Multipoint, LineString, Point)
    Returns:
        .json
    """
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        json.dump(geo_json_generate_traj_from_list(traj, type_style), f)


def visual_polygon(traj, path, type_style="Polygon"):
    """
    Args:
        list traj
        String path
        String type_style(Multipoint, LineString, Point)
    Returns:
        .json
    """
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        json.dump(geo_json_generate_polygon_from_list(traj, type_style), f)


def visual_key_points(traj, path, type_style="Multipoint"):
    """
    Args:
        list traj
        String path
        String type_style(Multipoint)
    Returns:
        .json
    """
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        json.dump(geo_json_generate_points(traj, type_style), f)


def visual_converge_points(points, path, type_style='Multipoint'):
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        json.dump(geo_json_generate_points(points, type_style), f)


def visual_cluster_points(points, path, type_style='Multipoint'):
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        json.dump(geo_json_cluster_points(points, type_style), f)


def visual_cluster_center(points, path, type_style='Point'):
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        json.dump(geo_json_cluster_center(points, type_style), f)


def visual_points(points, path, type_style='Point'):
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        json.dump(geo_json_points(points, type_style), f)


def visual_node_list(node_list, path, type_style='Polygon'):
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        json.dump(geo_multi_lines(node_list, type_style), f)


def visual_node_value(node_list, path, type_style='Polygon'):
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        json.dump(geo_multi_value(node_list, type_style), f)


def geo_multi_value(link_wkts, type_style="Polygon"):
    res = {
        "type": "FeatureCollection",
        "features": []
    }
    # point_prop, ave_speed, std_dir, ave_speed_change, acc_speed_prop, low_speed_prop, speed_change_prop, ave_dir_change, weight_direction_change
    for i, link_info in enumerate(link_wkts):
        t = {
            "type": "Feature",
            "geometry": {
                "type": type_style, "coordinates": [link_info[0]]
            },
            "properties": {
                "cluster_id": i,
                "value": float(link_info[1][-1])

            }

        }

        res["features"].append(t)

    return res

def visual_cluster_center_tkdd(points, path, type_style='Point'):
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        json.dump(geo_json_cluster_center_tkdd(points, type_style), f)