import datetime
from dataclasses import dataclass
from functools import reduce

import numpy as np


@dataclass(frozen=True, eq=True)
class Point:
    lat: float
    lon: float
    classification: str
    stay_time_minute: int
    is_start_point: bool
    is_end_point: bool


data = {
    'start_time': datetime.time(6, 0, 0),
    'end_time': datetime.time(16, 0, 0),
    'meal_time': [datetime.time(12, 0, 0)],

    'start_point': Point(10, 12, 'house', 0, True, False),
    'end_point': Point(10, 12, 'house', 0, False, True),

    'points': [
        Point(12, 12, 'travel', 60, False, False),
        Point(14, 14, 'food', 40, False, False),
        Point(12, 20, 'travel', 60, False, False),
        Point(9, 19, 'travel', 30, False, False),
        Point(15, 14, 'travel', 60, False, False),
        Point(8, 6, 'food', 40, False, False),
        Point(12, 20, 'travel', 60, False, False),
        Point(9, 19, 'travel', 30, False, False),
        Point(17, 13, 'travel', 60, False, False),
        Point(19, 6, 'travel', 30, False, False),
        Point(14, 15, 'travel', 60, False, False),
    ]
}


def meal_time_gaussian(x, mean):
    sigma = 1
    mean_value = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(0 ** 2 / (2 * sigma ** 2))
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) * 1 / mean_value


def end_time_gaussian(x, mean):
    sigma = 1
    mean_value = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(0 ** 2 / (2 * sigma ** 2))
    return (1 - (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) * (1 / mean_value)) * 100


def calc_time(p1, p2):
    return int((((p2.lon - p1.lon) ** 2 + (p2.lat - p1.lat) ** 2) ** 0.5) * 6)


def calc_weight(time, cost):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def time_func(x):
        sigX = (x - 30) / 30
        return -sigmoid(sigX) + 1

    def cost_func(x):
        sigX = (x - 10000) / 100000
        return -sigmoid(sigX) + 1

    return time_func(time) + cost_func(cost)


verbose = False
end_time_over_threshold_hour = 1.5

allPoints = data['points'][:]
allPoints.append(data['start_point'])
allPoints.append(data['end_point'])
weightMap = dict[Point, dict[Point, float]]()
timeMap = dict[Point, dict[Point, float]]()

for p in allPoints:
    weightMap[p] = dict()
    timeMap[p] = dict()


def plus_time(time, delta_minutes):
    hour = time.hour
    minute = time.minute + delta_minutes
    hour += int(minute / 60)
    minute = minute % 60
    return datetime.time(hour, minute, time.second)


def get_weight(p1, p2):
    if p2 not in weightMap[p1]:
        time = get_time(p1, p2)
        cost = 0
        weightMap[p1][p2] = calc_weight(time, cost)
    return weightMap[p1][p2]


def get_time(p1, p2):
    if p2 not in timeMap[p1]:
        timeMap[p1][p2] = calc_time(p1, p2)
    return timeMap[p1][p2]


def calc(data):
    visited, score = dfs([data['start_point']], 0, 0, 0, 0, 1)

    start_time_float = data['start_time'].hour + data['start_time'].minute / 60
    elapsed_time_minute = 0
    before_p = None

    for i, p in enumerate(visited):
        elapsed_time_minute += timeMap[before_p][p] if before_p is not None else 0
        print("#{0}: [{1:7.5f}, {2:7.5f}] {3} {4:3.2f}~{5:3.2f}".format(i + 1, p.lat, p.lon, p.classification,
                                                                        start_time_float + elapsed_time_minute / 60,
                                                                        start_time_float + (
                                                                                elapsed_time_minute + p.stay_time_minute) / 60))
        elapsed_time_minute += p.stay_time_minute
        before_p = p
    print("Visited {2}/{1} Points, Score: {0}".format(score, len(allPoints), len(visited)))


def dfs(visited, score, depth, elapsed_time_minute, meal_time_index, meal_time_weight):
    start_time = data['start_time']

    if verbose:
        print("{0}| Visited {1}, score: {2}".format(' ' * depth,
                                                    ["{2}[{0:0.5f}, {1:0.5f}]".format(p.lat, p.lon, p.classification)
                                                     for p in visited],
                                                    score))

    # 마지막 여행지를 만나면 끝나는 시간과 비교하여 최종 점수 반환
    if reduce(lambda acc, cur: acc or cur.is_end_point, visited, False):
        mean_end_time_hour = data['end_time'].hour + data['end_time'].minute / 60
        real_end_time_hour = start_time.hour + (start_time.minute + elapsed_time_minute) / 60
        end_time_weight = end_time_gaussian(real_end_time_hour, mean_end_time_hour)
        if verbose:
            print(
                "End time {0:0.2f}, Mean end time {1:0.2f}, Meal time weight: {4:0.5f} End time weight: {2:0.5f}, score: {3}".format(
                    real_end_time_hour,
                    mean_end_time_hour,
                    end_time_weight,
                    score * end_time_weight * meal_time_weight,
                    meal_time_weight))
        return visited, score * meal_time_weight * end_time_weight

    max_score = score
    visited_max_score = visited
    new_meal_time_weight = 1

    for nvp in (p for p in allPoints if p not in visited):
        point_end_time = start_time.hour + (elapsed_time_minute + get_time(visited[-1], nvp) + nvp.stay_time_minute) / 60
        if point_end_time > data['end_time'].hour + data['end_time'].minute / 60 + end_time_over_threshold_hour:
            continue
        nv = visited[:]
        nv.append(nvp)

        if nvp.classification == "food" and len(data['meal_time']) > 0:
            if meal_time_index >= len(data['meal_time']): continue
            food_start_time = (start_time.hour * 60 + start_time.minute + elapsed_time_minute + get_time(visited[-1],
                                                                                                         nvp)) / 60
            mean_time = data['meal_time'][meal_time_index].hour + data['meal_time'][meal_time_index].minute / 60
            new_meal_time_weight = meal_time_gaussian(food_start_time, mean_time)
            if verbose:
                print("{0}| Next is food! food time {2:0.2f}, mean time {3:0.2f}, Meal time weight: {1:0.5f}".format(
                    ' ' * depth,
                    new_meal_time_weight,
                    food_start_time,
                    mean_time))

        new_visited, new_score = dfs(
            visited=nv,
            score=score + get_weight(visited[-1], nvp),
            depth=depth + 1,
            elapsed_time_minute=elapsed_time_minute + get_time(visited[-1], nvp) + nvp.stay_time_minute,
            meal_time_index=meal_time_index + (1 if nvp.classification == "food" else 0),
            meal_time_weight=meal_time_weight * new_meal_time_weight
        )

        if new_score > max_score:
            max_score = new_score
            visited_max_score = new_visited

    return visited_max_score, max_score,


# 점수계산할때 이동시간 그대로 넣어버림 -> 수정
# visit time 고려

if __name__ == '__main__':
    calc(data)
