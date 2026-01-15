"""
Star Catalog for Plate Solving

Provides a catalog of bright stars for plate solving. Uses embedded catalog
data for the ~300 brightest stars visible to the naked eye (magnitude < 4.5).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


@dataclass
class CatalogStar:
    """A star from the catalog."""
    hip_id: int          # Hipparcos catalog ID
    ra: float            # Right ascension in degrees (J2000)
    dec: float           # Declination in degrees (J2000)
    mag: float           # Visual magnitude
    name: Optional[str] = None  # Common name if available

    @property
    def ra_rad(self) -> float:
        """Right ascension in radians."""
        return math.radians(self.ra)

    @property
    def dec_rad(self) -> float:
        """Declination in radians."""
        return math.radians(self.dec)

    def unit_vector(self) -> np.ndarray:
        """Get unit vector pointing to star in celestial coordinates."""
        cos_dec = math.cos(self.dec_rad)
        return np.array([
            cos_dec * math.cos(self.ra_rad),
            cos_dec * math.sin(self.ra_rad),
            math.sin(self.dec_rad)
        ])

    def angular_distance(self, other: 'CatalogStar') -> float:
        """Calculate angular distance to another star in degrees."""
        v1 = self.unit_vector()
        v2 = other.unit_vector()
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return math.degrees(math.acos(dot))


class StarCatalog:
    """
    Catalog of bright stars for plate solving.

    Contains embedded data for ~300 brightest stars (mag < 4.5).
    """

    def __init__(self, max_magnitude: float = 4.5):
        """
        Initialize the star catalog.

        Args:
            max_magnitude: Maximum magnitude to include (default 4.5)
        """
        self.max_magnitude = max_magnitude
        self.stars: List[CatalogStar] = []
        self._load_embedded_catalog()

    def _load_embedded_catalog(self):
        """Load the embedded bright star catalog."""
        # Bright star data: (HIP_ID, RA_deg, Dec_deg, Vmag, Name)
        # Selected from Hipparcos catalog, ~300 brightest stars
        bright_stars = [
            # Navigation stars and major bright stars
            (32349, 101.2872, -16.7161, -1.46, "Sirius"),
            (30438, 95.9880, -52.6958, -0.72, "Canopus"),
            (69673, 213.9153, 19.1824, -0.04, "Arcturus"),
            (71683, 219.8962, -60.8339, -0.01, "Alpha Centauri"),
            (91262, 279.2347, 38.7837, 0.03, "Vega"),
            (24436, 78.6345, -8.2016, 0.12, "Rigel"),
            (37279, 114.8255, 5.2250, 0.34, "Procyon"),
            (24608, 79.1723, 45.9990, 0.08, "Capella"),
            (27989, 88.7929, 7.4070, 0.50, "Betelgeuse"),
            (7588, 24.4285, -57.2367, 0.46, "Achernar"),
            (68702, 210.9559, -60.3730, 0.61, "Hadar"),
            (97649, 297.6958, 8.8683, 0.77, "Altair"),
            (60718, 186.6496, -63.0990, 0.77, "Acrux"),
            (21421, 68.9802, 16.5093, 0.85, "Aldebaran"),
            (65474, 201.2983, -11.1614, 0.98, "Spica"),
            (80763, 247.3519, -26.4320, 1.00, "Antares"),
            (37826, 116.3289, 28.0262, 1.14, "Pollux"),
            (9884, 31.7932, -29.2970, 1.16, "Fomalhaut"),
            (62434, 191.9303, -59.6885, 1.25, "Mimosa"),
            (102098, 310.3580, 45.2803, 1.25, "Deneb"),
            (49669, 152.0929, 11.9672, 1.35, "Regulus"),
            (25336, 81.2828, -1.9425, 1.64, "Bellatrix"),
            (36850, 113.6494, 31.8884, 1.58, "Castor"),
            (25930, 84.0533, -1.2019, 1.70, "Alnilam"),
            (26311, 84.6866, -1.9422, 2.09, "Alnitak"),
            (26727, 85.1897, -1.2428, 2.23, "Mintaka"),
            (113368, 344.4127, -29.6222, 1.17, "Fomalhaut"),
            (677, 2.0965, 29.0906, 2.06, "Alpheratz"),
            (746, 2.2942, 59.1500, 2.27, "Caph"),
            (3419, 10.8974, 56.5373, 2.23, "Schedar"),
            (5447, 17.4333, 35.6206, 2.00, "Mirach"),
            (8102, 25.9155, 89.2641, 2.02, "Polaris"),
            (8903, 28.6603, 63.6700, 2.24, "Gamma Cas"),
            (9640, 30.9748, 42.3297, 2.04, "Almach"),
            (11767, 37.9546, 89.2641, 2.00, "Kochab"),
            (14135, 45.5700, 4.0897, 2.00, "Hamal"),
            (14576, 46.1992, 53.5064, 1.79, "Mirfak"),
            (15863, 51.0809, 49.8612, 2.12, "Algol"),
            (17702, 56.8713, 24.1053, 2.87, "Alcyone"),
            (21421, 68.9802, 16.5093, 0.85, "Aldebaran"),
            (25428, 81.5729, 28.6074, 1.65, "Elnath"),
            (25930, 84.0533, -1.2019, 1.70, "Alnilam"),
            (27913, 88.5958, -42.8196, 2.21, "Wezen"),
            (28360, 89.9303, -42.8220, 1.50, "Adhara"),
            (30324, 95.6749, -17.9559, 1.98, "Murzim"),
            (31681, 99.4279, 16.3993, 1.93, "Alhena"),
            (32349, 101.2872, -16.7161, -1.46, "Sirius"),
            (33579, 104.6565, -28.9721, 1.84, "Aludra"),
            (34444, 107.0978, -26.3932, 1.86, "Naos"),
            (35904, 110.0308, 21.9822, 3.17, "Propus"),
            (36188, 111.0238, -29.3031, 2.45, "Suhail"),
            (37279, 114.8255, 5.2250, 0.34, "Procyon"),
            (39429, 120.8962, -40.0033, 2.25, "Avior"),
            (39953, 122.3834, -47.3366, 1.68, "Miaplacidus"),
            (41037, 125.6285, -59.5095, 2.76, "Aspidiske"),
            (42913, 131.1761, -54.7088, 2.21, "Turais"),
            (44816, 137.0100, -43.4320, 2.06, "Alsuhail"),
            (45238, 138.3010, -69.7172, 1.67, "Regor"),
            (46390, 141.8970, -8.6586, 2.08, "Alphard"),
            (49669, 152.0929, 11.9672, 1.35, "Regulus"),
            (50583, 154.9931, 19.8414, 2.56, "Algieba"),
            (53910, 165.4600, 56.3824, 1.77, "Dubhe"),
            (54061, 165.9320, 61.7510, 2.37, "Merak"),
            (57632, 177.2649, 14.5720, 2.14, "Denebola"),
            (58001, 178.4576, -16.1958, 2.75, "Algorab"),
            (59774, 183.9514, -17.5419, 2.59, "Gienah"),
            (60718, 186.6496, -63.0990, 0.77, "Acrux"),
            (61084, 187.7916, -57.1132, 1.63, "Gacrux"),
            (62434, 191.9303, -59.6885, 1.25, "Mimosa"),
            (63608, 195.5443, 10.9591, 2.83, "Vindemiatrix"),
            (65378, 200.9814, 54.9254, 1.86, "Mizar"),
            (65474, 201.2983, -11.1614, 0.98, "Spica"),
            (66657, 204.9719, -53.4664, 2.55, "Muhlifain"),
            (67301, 206.8855, 49.3133, 1.85, "Alkaid"),
            (68702, 210.9559, -60.3730, 0.61, "Hadar"),
            (69673, 213.9153, 19.1824, -0.04, "Arcturus"),
            (71683, 219.8962, -60.8339, -0.01, "Alpha Centauri"),
            (72607, 222.6764, -16.0418, 2.75, "Zubenelgenubi"),
            (74785, 229.2520, -9.3829, 2.61, "Zubeneschamali"),
            (76267, 233.6715, 26.7147, 2.23, "Alphecca"),
            (77070, 236.0668, 6.4256, 2.63, "Unukalhai"),
            (78820, 241.3593, -19.8054, 2.29, "Dschubba"),
            (80763, 247.3519, -26.4320, 1.00, "Antares"),
            (82273, 252.1663, -69.0277, 2.82, "Atria"),
            (84012, 257.5949, -15.7249, 2.43, "Sabik"),
            (85927, 263.4022, -37.1038, 1.63, "Shaula"),
            (86032, 263.7334, 12.5600, 2.08, "Rasalhague"),
            (86228, 264.3297, -43.0000, 1.85, "Sargas"),
            (87833, 269.1516, 51.4889, 2.23, "Eltanin"),
            (88635, 271.4520, -30.4240, 2.70, "Nunki"),
            (90185, 276.0429, -34.3844, 1.85, "Kaus Australis"),
            (91262, 279.2347, 38.7837, 0.03, "Vega"),
            (92855, 284.7358, 32.6894, 2.20, "Sheliak"),
            (95947, 292.6804, 27.9597, 2.72, "Albireo"),
            (97649, 297.6958, 8.8683, 0.77, "Altair"),
            (100751, 306.4119, -56.7351, 1.94, "Peacock"),
            (102098, 310.3580, 45.2803, 1.25, "Deneb"),
            (107315, 326.0466, 9.8752, 2.39, "Enif"),
            (109268, 332.0580, -46.9611, 1.74, "Al Nair"),
            (112122, 340.6666, -46.8847, 2.10, "Ankaa"),
            (113368, 344.4127, -29.6222, 1.16, "Fomalhaut"),
            (113881, 345.9436, 28.0828, 2.49, "Scheat"),
            (677, 2.0965, 29.0906, 2.06, "Alpheratz"),
        ]

        # Remove duplicates based on HIP ID and filter by magnitude
        seen = set()
        for hip_id, ra, dec, mag, name in bright_stars:
            if hip_id not in seen and mag <= self.max_magnitude:
                seen.add(hip_id)
                self.stars.append(CatalogStar(
                    hip_id=hip_id,
                    ra=ra,
                    dec=dec,
                    mag=mag,
                    name=name
                ))

        # Sort by magnitude (brightest first)
        self.stars.sort(key=lambda s: s.mag)

    def get_stars_in_region(self, ra_center: float, dec_center: float,
                           radius: float) -> List[CatalogStar]:
        """
        Get stars within a circular region.

        Args:
            ra_center: Center RA in degrees
            dec_center: Center Dec in degrees
            radius: Search radius in degrees

        Returns:
            List of stars in the region
        """
        center = CatalogStar(0, ra_center, dec_center, 0.0)
        return [s for s in self.stars if center.angular_distance(s) <= radius]

    def get_visible_stars(self, lat: float, lon: float,
                         lst: float, min_altitude: float = 10.0) -> List[Tuple[CatalogStar, float, float]]:
        """
        Get stars visible from a given location.

        Args:
            lat: Observer latitude in degrees
            lon: Observer longitude in degrees
            lst: Local sidereal time in hours
            min_altitude: Minimum altitude above horizon in degrees

        Returns:
            List of (star, altitude, azimuth) tuples
        """
        visible = []

        for star in self.stars:
            alt, az = self._ra_dec_to_alt_az(star.ra, star.dec, lat, lst)
            if alt >= min_altitude:
                visible.append((star, alt, az))

        return visible

    def _ra_dec_to_alt_az(self, ra: float, dec: float,
                          lat: float, lst: float) -> Tuple[float, float]:
        """
        Convert RA/Dec to Alt/Az.

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            lat: Observer latitude in degrees
            lst: Local sidereal time in hours

        Returns:
            (altitude, azimuth) in degrees
        """
        # Hour angle
        ha = (lst * 15 - ra) % 360
        ha_rad = math.radians(ha)
        dec_rad = math.radians(dec)
        lat_rad = math.radians(lat)

        # Calculate altitude
        sin_alt = (math.sin(dec_rad) * math.sin(lat_rad) +
                   math.cos(dec_rad) * math.cos(lat_rad) * math.cos(ha_rad))
        alt = math.degrees(math.asin(np.clip(sin_alt, -1.0, 1.0)))

        # Calculate azimuth
        cos_az = ((math.sin(dec_rad) - math.sin(lat_rad) * sin_alt) /
                  (math.cos(lat_rad) * math.cos(math.radians(alt)) + 1e-10))
        az = math.degrees(math.acos(np.clip(cos_az, -1.0, 1.0)))

        if math.sin(ha_rad) > 0:
            az = 360 - az

        return alt, az

    def build_triangle_index(self, max_stars: int = 50) -> dict:
        """
        Build a geometric hash index of star triangles for plate solving.

        Uses the brightest stars to create triangle patterns that can
        be matched against detected stars in images.

        Args:
            max_stars: Maximum number of stars to include

        Returns:
            Dictionary mapping triangle hash to star triplets
        """
        stars = self.stars[:max_stars]
        index = {}

        # Generate all triangles from pairs of stars
        for i in range(len(stars)):
            for j in range(i + 1, len(stars)):
                for k in range(j + 1, len(stars)):
                    s1, s2, s3 = stars[i], stars[j], stars[k]

                    # Calculate side lengths
                    d12 = s1.angular_distance(s2)
                    d23 = s2.angular_distance(s3)
                    d13 = s1.angular_distance(s3)

                    # Skip triangles that are too small or too large
                    sides = sorted([d12, d23, d13])
                    if sides[0] < 1.0 or sides[2] > 60.0:
                        continue

                    # Compute hash based on side ratios
                    hash_key = self._triangle_hash(sides)

                    if hash_key not in index:
                        index[hash_key] = []
                    index[hash_key].append((s1, s2, s3, sides))

        return index

    def _triangle_hash(self, sides: List[float], bins: int = 100) -> Tuple[int, int]:
        """
        Compute a hash key for a triangle based on side ratios.

        Uses two ratios: shortest/longest and middle/longest

        Args:
            sides: Sorted list of side lengths [shortest, middle, longest]
            bins: Number of bins for quantization

        Returns:
            Tuple of quantized ratios
        """
        r1 = sides[0] / sides[2]  # shortest / longest
        r2 = sides[1] / sides[2]  # middle / longest

        # Quantize to bins
        b1 = int(r1 * bins)
        b2 = int(r2 * bins)

        return (b1, b2)

    def __len__(self) -> int:
        return len(self.stars)

    def __iter__(self):
        return iter(self.stars)
