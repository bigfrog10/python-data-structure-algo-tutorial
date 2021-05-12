# LC1396 Design Underground System
import statistics
from collections import defaultdict
class UndergroundSystem:
    def __init__(self):
        self._checkins = dict()  # book keeping
        self._travels = defaultdict(list)  # stats
    def checkIn(self, id: int, stationName: str, t: int) -> None:
        if id in self._checkins: raise Exception('already checked in')
        self._checkins[id] = (stationName, t)
    def checkOut(self, id: int, stationName: str, t: int) -> None:
        checkin = self._checkins[id]
        del self._checkins[id]
        self._travels[(checkin[0], stationName)].append(t - checkin[1])
    def getAverageTime(self, startStation: str, endStation: str) -> float:
        return statistics.mean(self._travels[(startStation, endStation)])
