"""
This module defines the fuel models defined by the Rothermel fire spread model.

The basis for the fire simulation module is the rate of spread provided by the Rothermel model. In this module,
various classes defining this model are defined. The FuelBase class defines the fire parameters that may be
calculated solely from the fuel model properties provided. The various parameters provided by this model are
implemented as class properties using only standard Python and, more rarely, Numpy functions and structures. This was
purposely done to help the user identify and modify the various model parameters. Nevertheless,
for most applications, the definition of these parameters (and, subsequently, the respective properties) may remain
unchanged; as such, the usage of caching methodologies may be advisable as to reduce the computational overhead
resulting from this approach. The Spread class allows for obtaining additional fire propagation parameters,
such as the rate of spread, which are dependent on the fuel model selected and other environmental parameters,
such as the wind speed and terrain slope.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import Iterable, Dict, Tuple, Optional

import numpy as np

from ..utils.cached import Cached
from ..utils.custom_types import WindCellLike, TerrainCellLike


class FuelClass(Enum):
    """Defines the two types of fuel components (dead and alive) used in the Rothermel model."""
    DEAD = auto()
    ALIVE = auto()


class TimeClass(Enum):
    """Defines the three time classes (1H, 10H, and 100H) considered for several parameters of the Rothermel model."""
    T1H = auto()
    T10H = auto()
    T100H = auto()


@dataclass
class FuelModel(Cached):
    """Defines the fuel parameters as described by the Rothermel model."""
    # Units: kJ.kg^-1
    heat_content_dead: float
    heat_content_alive: float

    # Units: kg.m^-2
    oven_dry_fuel_load_alive: float
    oven_dry_fuel_load_dead_1h: float
    oven_dry_fuel_load_dead_10h: float
    oven_dry_fuel_load_dead_100h: float

    # Units: m
    fuel_depth_alive: float
    fuel_depth_dead: float

    # Units: fraction [0, 1]
    fuel_moisture_alive: float
    fuel_moisture_dead_extinction: float
    fuel_moisture_dead_1h: float
    fuel_moisture_dead_10h: float
    fuel_moisture_dead_100h: float

    # Units: m
    flame_length: float

    # Units: cm^-1
    surface_area_to_volume_ratio_alive: float
    surface_area_to_volume_ratio_dead_1h: float

    # Units: m^-1
    surface_area_to_volume_ratio_dead_10h: float = 260
    surface_area_to_volume_ratio_dead_100h: float = 80

    # Units: fraction
    mineral_content_total: float = 0.0555
    mineral_content_effective: float = 0.01

    # Units: kg.m^-3
    oven_dry_particle_density: float = 500

    name: str = ""

    @classmethod
    def from_dict(cls, arg_dict: Dict) -> FuelModel:
        """Constructor using a dictionary of parameters."""
        args = {k: v for k, v in arg_dict.items()
                if (k in inspect.signature(cls).parameters and k != "name")}
        if "name" in arg_dict:
            args["name"] = str(arg_dict["name"])
        return cls(**args)

    @property
    def fuel_varieties(self) -> Iterable:
        """Combinations of fuel categories considered in the Rothermel model."""
        varieties = [(FuelClass.DEAD, TimeClass.T1H),
                     (FuelClass.DEAD, TimeClass.T10H),
                     (FuelClass.DEAD, TimeClass.T100H),
                     (FuelClass.ALIVE, None)]
        return iter(varieties)

    # Units: m
    def fuel_depth(self, i: FuelClass) -> float:
        """Returns the fuel bed depth of either dead-fuel or alive-fuel.

        This function is not strictly needed for the Rothermel model but it is helpfull through out the code to use this
        this parameters.
        """
        if i is FuelClass.DEAD:
            return self.fuel_depth_dead
        elif i is FuelClass.ALIVE:
            return self.fuel_depth_alive

    # Units: kg.m^-2
    def oven_dry_fuel_load(self, i: FuelClass, j: Optional[TimeClass] = None) -> float:
        """Returns the oven-dry fuel load of either dead-fuel based on its time-class or alive-fuel.

        This function is not strictly needed for the Rothermel model but it is helpfull through out the code to use this
        parameters.
        """
        if i is FuelClass.DEAD:
            if j is TimeClass.T1H:
                return self.oven_dry_fuel_load_dead_1h
            elif j is TimeClass.T10H:
                return self.oven_dry_fuel_load_dead_10h
            elif j is TimeClass.T100H:
                return self.oven_dry_fuel_load_dead_100h
        elif i is FuelClass.ALIVE:
            return self.oven_dry_fuel_load_alive
        raise NotImplementedError

    # Units: m^-1
    def surface_area_to_volume_ratio(self, i: FuelClass, j: Optional[TimeClass] = None) -> float:
        """Returns the various Surface-Area-to-Volume ratios of either dead-fuel based on its time-class or alive-fuel.

        This function is not strictly needed for the Rothermel model but it is helpfull through out the code to use this
        parameters.
        """
        if i is FuelClass.DEAD:
            if j is TimeClass.T1H:
                return self.surface_area_to_volume_ratio_dead_1h
            elif j is TimeClass.T10H:
                return self.surface_area_to_volume_ratio_dead_10h
            elif j is TimeClass.T100H:
                return self.surface_area_to_volume_ratio_dead_100h
        elif i is FuelClass.ALIVE:
            return self.surface_area_to_volume_ratio_alive
        raise NotImplementedError

    # Units: fraction
    def fuel_moisture(self, i: FuelClass, j: Optional[TimeClass] = None) -> float:
        """Returns the various fuel moisture percentages of either dead-fuel based on its time-class or alive-fuel.

        This function is not strictly needed for the Rothermel model but it is helpfull through out the code to use this
        parameters.
        """
        if i is FuelClass.DEAD:
            if j is TimeClass.T1H:
                return self.fuel_moisture_dead_1h
            elif j is TimeClass.T10H:
                return self.fuel_moisture_dead_10h
            elif j is TimeClass.T100H:
                return self.fuel_moisture_dead_100h
        elif i is FuelClass.ALIVE:
            return self.fuel_moisture_alive
        raise NotImplementedError

    # Units: m^2
    def mean_surface_area(self, i: FuelClass, j: Optional[TimeClass] = None) -> float:
        """Mean surface area per unit fuel cell for given fuel and time class."""
        return self.surface_area_to_volume_ratio(i, j) * self.oven_dry_fuel_load(i, j) / self.oven_dry_particle_density

    @cached_property
    # Units: m^2
    def total_dead_mean_surface_area(self) -> float:
        """Total mean surface area per unit fuel cell of only dead fuel class."""
        # Check this sum (prev. was it over TimeClass, so NONE was considered)
        return sum([self.mean_surface_area(i, j) for i, j in self.fuel_varieties
                    if i is FuelClass.DEAD])

    @cached_property
    # Units: m^2
    def total_mean_surface_area(self) -> float:
        """Total mean surface area per unit fuel cell for both dead an alive fuel classes."""
        return self.total_dead_mean_surface_area + self.mean_surface_area(FuelClass.ALIVE)

    # Units: kJ.kg^-1
    def heat_content(self, i: FuelClass) -> float:
        """Returns the heat content of either dead or alive-fuel.

        This function is not strictly needed for the Rothermel model but it is helpfull through out the code to use this
        parameters.
        """
        if i is FuelClass.DEAD:
            return self.heat_content_dead
        elif i is FuelClass.ALIVE:
            return self.heat_content_alive

    # Units: fraction
    def factor(self, i: FuelClass, j: TimeClass) -> float:
        """Returns the weighting factors of the different types of fuel: dead and alive."""
        if i is FuelClass.DEAD:
            a_t = self.total_dead_mean_surface_area
        elif i is FuelClass.ALIVE:
            a_t = self.total_mean_surface_area
        else:
            raise ValueError("WRONG!")  # TODO: Fix this

        return self.mean_surface_area(i, j) / a_t

    @cached_property
    # Units: m^-1
    def total_surface_area_to_volume_ratio_dead(self) -> float:
        """Total Surface-Area-to-Volume(SAV) ratio of dead-fuel based on a weighted average.

        Each time-class (1h, 10h and 100h) of the SAV contributes to its Dead-SAV average with a certain weight given by
        "factor" (notice that this factor only has in account the dead-fuel contribution, not alive). Thus we can simply
        iterate for all SAV time-classes(only dead-fuel has a time-class) with the respective weight to obtain the full
        dead average.
        """
        return sum([self.factor(fuel_class, time_class) * self.surface_area_to_volume_ratio(fuel_class, time_class)
                    for fuel_class, time_class in self.fuel_varieties
                    if fuel_class is FuelClass.DEAD])

    @cached_property
    # Units: m^-1
    def total_surface_area_to_volume_ratio(self) -> float:
        """Total Surface-Area-to-Volume(SAV) ratio of both alive and dead fuel based on a weighted average.

        For each time-class (1h, 10h and 100h) of the Dead-SAV and for the Alive-SAV is attributed a weight which is the
        ratio A/A_total. And by iterating for all types of SAV we compute its average, with the respective weight, to
        obtain its average.
        """
        return sum([(self.surface_area_to_volume_ratio(fuel_class, time_class) *
                     self.mean_surface_area(fuel_class, time_class) /
                     self.total_mean_surface_area)
                    for fuel_class, time_class
                    in self.fuel_varieties])

    # Units: fraction
    def optimum_packing_ratio(self, i: FuelClass) -> float:
        """Optimum packing ratio for given fuel class."""
        if i is FuelClass.DEAD:
            surface_area_to_volume_ratio = self.total_surface_area_to_volume_ratio_dead
        elif i is FuelClass.ALIVE:
            surface_area_to_volume_ratio = self.surface_area_to_volume_ratio_alive
        else:
            raise ValueError("WRONG!")  # TODO: Fix this

        return 0.20395 * (surface_area_to_volume_ratio / 100) ** (-0.8189)

    @cached_property
    # Units: fraction
    def total_dead_fuel_moisture(self) -> float:
        """Total fuel moisture ratio of dead-fuel based on a weighted average.

        Each time-class (1h, 10h and 100h) of fuel moisture contributes to its average with a certain weight given by
        "factor" (notice that this factor only has in account the dead-fuel contribution, not alive). Thus we iterate
        for all fuel moisture time-classes(only dead-fuel has a time-class) with the respective weight to obtain the
        full dead average.
        """
        return sum([self.factor(fuel_class, time_class) * self.fuel_moisture(fuel_class, time_class)
                    for fuel_class, time_class in self.fuel_varieties
                    if fuel_class is FuelClass.DEAD])

    # Units: kg.m^-3
    def oven_dry_bulk_density(self, i: FuelClass) -> float:
        """Oven-dry bulk density for respective fuel class (alive or dead)."""
        return sum([self.oven_dry_fuel_load(fuel_class, time_class) / self.fuel_depth(i)
                    for fuel_class, time_class in self.fuel_varieties
                    if fuel_class is i])

    # Units: fraction
    def packing_ratio(self, i: FuelClass) -> float:
        """Real packing ratio (not the optimum) for given fuel class."""
        return self.oven_dry_bulk_density(i) / self.oven_dry_particle_density

    # Units: kg.m^-2
    def net_fuel_load(self, i: FuelClass) -> float:
        """Net fuel load of either dead or alive fuel class.

        This is obtained by subtracting the mineral content contribution to the weighted oven dry fuel load average.
        Notice that alive fuel class has a total weight of 1.
        """
        if i is FuelClass.DEAD:
            oven_dry_fuel_load = sum([(self.factor(fuel_class, time_class)
                                       * self.oven_dry_fuel_load(fuel_class, time_class))
                                      for fuel_class, time_class in self.fuel_varieties
                                      if fuel_class is FuelClass.DEAD])
        else:
            oven_dry_fuel_load = self.oven_dry_fuel_load(FuelClass.ALIVE)
        return oven_dry_fuel_load * (1 - self.mineral_content_total)

    # Units: fraction
    def moisture_damping_coefficient(self, i: FuelClass) -> float:
        """Moisture damping coefficient for either dead or alive fuel class.

        This is the coefficient that damps the reaction velocity due to the moisture content of the fuel.
        """
        if i is FuelClass.DEAD:
            fuel_moist = self.total_dead_fuel_moisture
            fuel_moist_ext = self.fuel_moisture_dead_extinction
            # TODO: meter um aviso caso seja rm = 1
        elif i is FuelClass.ALIVE:
            fuel_moist = self.fuel_moisture_alive
            fuel_moist_ext = max(self.fuel_moisture_alive_extintion, self.fuel_moisture_dead_extinction)
        else:
            raise ValueError("WRONG!")  # TODO: Fix this

        r_m = min(fuel_moist / fuel_moist_ext, 1)
        return 1 - 2.59 * r_m + 5.11 * r_m ** 2 - 3.52 * r_m ** 3

    # Units: s^-1
    def max_reaction_velocity(self, i: FuelClass) -> float:
        """Maximum reaction "velocity" for either dead or alive fuel class."""
        if i is FuelClass.DEAD:
            surface_area_to_volume_ratio = self.total_surface_area_to_volume_ratio_dead
        elif i is FuelClass.ALIVE:
            surface_area_to_volume_ratio = self.surface_area_to_volume_ratio_alive
        else:
            raise NotImplementedError

        return ((30.48 * surface_area_to_volume_ratio / 100) ** 1.5 /
                (0.0594 * (30.48 * surface_area_to_volume_ratio / 100) ** 1.5 + 495) * (1 / 60))

    # Units: s^-1
    def optimum_reaction_velocity(self, i: FuelClass) -> float:
        """Optimum reaction "velocity" for either dead or alive fuel class.

        This is the reaction velocity if there were no mineral content nor moisture content which damps this "velocity".
        """
        if i is FuelClass.DEAD:
            surface_area_to_volume_ratio = self.total_surface_area_to_volume_ratio_dead
        elif i is FuelClass.ALIVE:
            surface_area_to_volume_ratio = self.surface_area_to_volume_ratio_alive
        else:
            raise NotImplementedError

        return (self.max_reaction_velocity(i) *
                ((self.packing_ratio(i) / self.optimum_packing_ratio(i)) *
                 np.exp(1 - (self.packing_ratio(i) / self.optimum_packing_ratio(i)))) **
                (8.9033 * (surface_area_to_volume_ratio / 100) ** (-0.7913)))

    # Units: kJ.s^-1.m^-2
    def reaction_intensity(self, i: FuelClass) -> float:
        """Reaction intensity for either dead or alive fuel.

        This is the total heat release rate per unit area of fire front given fuel moisture and load restraint.
        """
        check_case_alive = (i is FuelClass.ALIVE and
                            (self.fuel_moisture_alive > 1 or self.oven_dry_fuel_load(FuelClass.ALIVE) == 0))
        check_case_dead = (i is FuelClass.DEAD and self.total_dead_fuel_moisture > 1)
        if check_case_alive or check_case_dead:
            return 0
        else:
            return (self.optimum_reaction_velocity(i) * self.net_fuel_load(i) * self.heat_content(i) *
                    self.moisture_damping_coefficient(i) * self.mineral_damping_coefficient)

    @cached_property
    # Units: fraction
    def dead_to_alive_load_ratio(self):
        """Ratio between oven-dry dead fuel load and oven-dry alive fuel load.

        This ratio is not a simple division but a ratio between relevant loads of each fuel class.
        """
        w_dead = sum([(self.oven_dry_fuel_load(fuel_class, time_class)
                       * np.exp(-4.528 / (self.surface_area_to_volume_ratio(fuel_class, time_class) / 100)))
                      for fuel_class, time_class in self.fuel_varieties if fuel_class is FuelClass.DEAD])
        w_alive = (self.oven_dry_fuel_load(FuelClass.ALIVE)
                   * np.exp(-16.406 / (self.surface_area_to_volume_ratio(FuelClass.ALIVE) / 100)))
        return w_dead / w_alive

    @cached_property
    # Units: fraction
    def total_fine_dead_fuel_moisture(self):
        """Dead fine-fuel moisture."""
        a = sum([(self.fuel_moisture(fuel_class, time_class) * self.oven_dry_fuel_load(fuel_class, time_class)
                  * np.exp(-4.528 / (self.surface_area_to_volume_ratio(fuel_class, time_class) / 100)))
                 for fuel_class, time_class in self.fuel_varieties if fuel_class is FuelClass.DEAD])
        b = sum([(self.oven_dry_fuel_load(fuel_class, time_class)
                  * np.exp(-4.528 / (self.surface_area_to_volume_ratio(fuel_class, time_class) / 100)))
                 for fuel_class, time_class in self.fuel_varieties if fuel_class is FuelClass.DEAD])
        return a / b

    @cached_property
    # Units: fraction
    def fuel_moisture_alive_extintion(self):
        """Alive fuel moisture of extinction.

        This is the moisture required to extinguish the ignition. A fuel with this moisture content will never ignite.
        """
        fuel_moisture_dead_extinction = max(0.3, self.fuel_moisture_dead_extinction)
        return (2.9 * self.dead_to_alive_load_ratio *
                (1 - self.total_fine_dead_fuel_moisture / fuel_moisture_dead_extinction) - 0.226)

    @cached_property
    # Units: m
    def total_fuel_depth(self):
        """Total fuel bed depth of both dead and alive fuels."""
        oven_dry_fuel_load_dead = sum([self.oven_dry_fuel_load(fuel_class, time_class)
                                       for fuel_class, time_class in self.fuel_varieties
                                       if fuel_class is FuelClass.DEAD])
        oven_dry_fuel_load_alive = self.oven_dry_fuel_load(FuelClass.ALIVE)
        return (self.fuel_depth(FuelClass.ALIVE) * oven_dry_fuel_load_alive + self.fuel_depth(FuelClass.DEAD) *
                oven_dry_fuel_load_dead) / (oven_dry_fuel_load_alive + oven_dry_fuel_load_dead)

    @cached_property
    # Units: kg.m^-3
    def total_oven_dry_bulk_density(self):
        """Total oven-dry bulk density.

         This is a mean of all fuel and time classes of oven-dry fuel loads of equal weight( the total fuel depth).
         """
        return sum([self.oven_dry_fuel_load(fuel_class, time_class)
                    for fuel_class, time_class in self.fuel_varieties]) / self.total_fuel_depth

    @cached_property
    # Units: fraction
    def total_optimum_packing_ratio(self) -> float:
        """Total optimum packing ratio.

        It represents the optimum packing ratio for all fuel and time classes."""
        return 0.20395 * (self.total_surface_area_to_volume_ratio / 100) ** (-0.8189)

    @property
    # Units: fraction
    def total_packing_ratio(self) -> float:
        """Total packing ratio.

        It represents the packing ratio for all fuel and time classes.
        """
        return self.total_oven_dry_bulk_density / self.oven_dry_particle_density

    @cached_property
    # Units: dimensionaless
    def effective_heating_number(self) -> float:
        """Effective heating number.

        This a number between zero and one that one multiplies by the total bulk density to obtain the effective
        total bulk density.
        """
        return np.exp(-4.5276 / (self.total_surface_area_to_volume_ratio / 100))

    @cached_property
    # Units: kJ.kg^-1
    def preignition_heat(self) -> float:
        """Pre-ignition heat.

        This is the heat required to ignite the fuel given a fuel moisture."""
        return 581.0 + 2594.0 * self.total_dead_fuel_moisture

    @cached_property
    # Units: fraction
    def mineral_damping_coefficient(self):
        """Mineral damping coefficient.

        This is coefficient that damps the reaction velocity due to the mineral content of the fuel."""
        return 0.174 * self.mineral_content_effective ** -0.19

    @cached_property
    # Units: fraction
    def propagating_flux_ratio(self) -> float:
        """Propagation flux ratio.

        This is the ratio of the total reaction intensity that actually ignites adjacent matter"""
        return (np.exp((0.792 + 3.7597 * (self.total_surface_area_to_volume_ratio / 100) ** 0.5) * (
                self.total_packing_ratio + 0.1)) / (192.0 + 7.9095 * (self.total_surface_area_to_volume_ratio / 100)))

    # Units: m.s^-1
    def rate_of_spread_o_i(self, i: FuelClass) -> float:
        """Rate of Spread of the fire front for no wind, no slope contribution for given fuel class."""
        return (self.reaction_intensity(i) * self.propagating_flux_ratio /
                (self.total_oven_dry_bulk_density * self.effective_heating_number * self.preignition_heat))

    # @property
    @cached_property
    # Units: m.s^-1
    def rate_of_spread_o(self) -> float:
        """Total Rate of Spread for no wind, no slope contribution.

        This is the rate of spread of the fire front that accounts for the live and dead fuel contributions.
        """
        if self.total_dead_mean_surface_area == 0:
            return 0
        return ((self.reaction_intensity(FuelClass.DEAD) + self.reaction_intensity(FuelClass.ALIVE)) *
                self.propagating_flux_ratio /
                (self.total_oven_dry_bulk_density * self.effective_heating_number * self.preignition_heat))

    @cached_property
    # Units: s
    def residence_time(self) -> float:
        """Resident time.

        This is the time that the fire front takes to transverse a given point of the fuel."""
        if self.total_dead_mean_surface_area != 0 and self.total_surface_area_to_volume_ratio != 0:
            return 384 * 60 / (self.total_surface_area_to_volume_ratio / 100 * 30.478)
        return 0


@dataclass
class Spread:
    """Defines various fire spread related properties based on the Rothermel model and environmental parameters."""
    fuel: FuelModel
    wind: WindCellLike
    terrain: TerrainCellLike

    @property
    # Units: fraction
    def logarithmic_factor(self) -> float:
        """Logarithmic factor.

        This is the factor that interpolates the wind velocity from ground to mid-flame height.
        """
        fuel_depth_dead = self.fuel.fuel_depth(FuelClass.DEAD)
        flame_length = self.fuel.flame_length
        return (1 + 0.36 * fuel_depth_dead / flame_length) * \
               (np.log((flame_length / fuel_depth_dead + 0.36) / 0.13) - 1) / (
                   np.log((self.wind.height / 2 + 0.36 * fuel_depth_dead) / (0.13 * fuel_depth_dead)))

    @property
    # Units: m.s^-1
    def spread_wind_vel_x(self) -> float:
        """X wind component at mid-flame height."""
        return self.wind.vel_x * self.logarithmic_factor

    @property
    # Units: m.s^-1
    def spread_wind_vel_y(self) -> float:
        """Y wind component at mid-flame height."""
        return self.wind.vel_y * self.logarithmic_factor

    @property
    # Units: m.s^-1
    def spread_wind_vel_z(self) -> float:
        """Z wind component at mid-flame height."""
        return self.wind.vel_z * self.logarithmic_factor

    @property
    # Units: m.s^-1
    def wind_speed(self) -> float:
        """Wind velocity norm at mid-flame height."""
        return float(np.linalg.norm((self.spread_wind_vel_x, self.spread_wind_vel_y, self.spread_wind_vel_z)))

    @property
    def slope_flux(self) -> float:
        """Slope contribution to the total flux."""
        return 5.275 * self.fuel.total_packing_ratio ** (-0.3) * np.tan(self.terrain.slope) ** 2

    # TODO: wind_flux and equivalent_wind seem to be almost equal. Maybe simplify. They serve different purposes
    @property
    def wind_flux(self) -> float:
        """Wind contribution to the total flux."""
        c = 7.47 * np.exp(-0.8711 * (self.fuel.total_surface_area_to_volume_ratio / 100) ** 0.55)
        b = 0.15988 * (self.fuel.total_surface_area_to_volume_ratio / 100) ** 0.54
        e = 0.715 * np.exp(-0.01094 * (self.fuel.total_surface_area_to_volume_ratio / 100))
        return c * (3.2808 * 60 * self.wind_speed) ** b * (
                self.fuel.total_packing_ratio / self.fuel.total_optimum_packing_ratio) ** (-e)

    @property
    # Units: m.s^-1
    def equivalent_wind(self) -> float:
        """Norm of the equivalent wind of the flux.

         This is the wind norm that would produce the same effect on the total flux as the normal wind plus the terrain
         slope combined.
         """
        if self.fuel.total_dead_mean_surface_area != 0:
            C = 7.47 * np.exp(-0.8711 * (self.fuel.total_surface_area_to_volume_ratio / 100) ** 0.55)
            B = 0.15988 * (self.fuel.total_surface_area_to_volume_ratio / 100) ** 0.54
            E = 0.715 * np.exp(-0.01094 * (self.fuel.total_surface_area_to_volume_ratio / 100))

            return ((np.linalg.norm(self.total_flux) / C *
                     (self.fuel.total_packing_ratio / self.fuel.total_optimum_packing_ratio) ** E) ** (1 / B) /
                    (3.281 * 60))
        else:
            return 0.1

    @property
    def total_flux(self) -> Tuple[float, float]:
        """Total flux.

         This flux is a vector that has wind and slope fluxes contributing to it depending on the angle that each flux
         makes respective to the wind reference frame.
         Note that the wind reference frame should be equivalent to the terrain reference frame (the origin can be
         shifted and the angles still will match but one needs to be careful with index matching). Otherwise, one might
         be cautious for the need of some reference frame transformation (example, rotations and/or translations on one
         side or the other if both, at least, agree on the type of lattice).
         """
        if (self.spread_wind_vel_x or self.spread_wind_vel_y) and self.fuel.total_dead_mean_surface_area:
            angle_w = np.arctan2(self.spread_wind_vel_y, self.spread_wind_vel_x) % (2 * np.pi)
            phi_x = self.wind_flux * np.cos(angle_w) + self.slope_flux * np.cos(self.terrain.slope_dir)
            phi_y = self.wind_flux * np.sin(angle_w) + self.slope_flux * np.sin(self.terrain.slope_dir)
            return phi_x, phi_y
        return 0.0, 0.0

    @property
    # Units: m.s^-1
    def rate_of_spread(self) -> float:
        """Total Rate of Spread with wind and terrain slope contributions.

        This is the rate of spread of the fire front that accounts for the live and dead fuel contributions, as well for
        the terrain and wind contribution appearing as an addition to the unit flux.
        """
        return self.fuel.rate_of_spread_o * (1 + float(np.linalg.norm(self.total_flux)))

    @property
    # Units: kJ.m^-2
    def heat_per_unit_area(self) -> float:
        """Heat per unit area.

        This is the heat released by the reaction intensity given that the a residence time has passed. So, it's the
        heat released, per unit area, by the flaming front.
        """
        return (self.fuel.residence_time *
                (self.fuel.reaction_intensity(FuelClass.DEAD) + self.fuel.reaction_intensity(FuelClass.ALIVE)))

    @property
    # Units: kJ.m^-1.s^-1
    def fire_line_intensity(self) -> float:
        """Byram's Fire line intensity."""
        return self.rate_of_spread * self.heat_per_unit_area

    @property
    # Units: m
    def flame_length(self) -> float:
        """Byram's flame length."""
        return 0.077476 * self.fire_line_intensity ** 0.46
