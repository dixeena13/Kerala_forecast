from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeralaStation:
    name: str
    district: str
    latitude: float
    longitude: float
    elevation_m: float
    coast_distance_km: float


KERALA_STATIONS: tuple[KeralaStation, ...] = (
    KeralaStation("Thiruvananthapuram", "Thiruvananthapuram", 8.52, 76.94, 10.0, 4.0),
    KeralaStation("Kollam", "Kollam", 8.89, 76.61, 3.0, 3.0),
    KeralaStation("Alappuzha", "Alappuzha", 9.50, 76.34, 1.0, 2.0),
    KeralaStation("Kochi", "Ernakulam", 9.97, 76.28, 5.0, 4.0),
    KeralaStation("Thrissur", "Thrissur", 10.53, 76.21, 10.0, 25.0),
    KeralaStation("Kozhikode", "Kozhikode", 11.26, 75.78, 5.0, 3.0),
    KeralaStation("Kannur", "Kannur", 11.87, 75.37, 12.0, 5.0),
    KeralaStation("Wayanad", "Wayanad", 11.69, 76.13, 760.0, 70.0),
    KeralaStation("Idukki", "Idukki", 9.85, 76.97, 900.0, 65.0),
)


def station_names() -> list[str]:
    return [station.name for station in KERALA_STATIONS]


def find_station(name: str) -> KeralaStation:
    normalized = name.strip().lower()
    for station in KERALA_STATIONS:
        if station.name.lower() == normalized or station.district.lower() == normalized:
            return station
    known = ", ".join(station_names())
    raise ValueError(f"Unknown Kerala station {name!r}. Known stations: {known}")
